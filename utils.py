import math
import pickle
import random
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
import os

from PIL import Image
from einops import rearrange
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from tqdm import tqdm
from torchvision.utils import save_image
from torch import distributed as dist
from loguru import logger

logging = logger


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


def set_seed(seed: int):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


def get_optimizer(params, name, **kwargs):
    if name == 'adam':
        from torch.optim import Adam
        return Adam(params, **kwargs)
    elif name == 'adamw':
        from torch.optim import AdamW
        return AdamW(params, **kwargs)
    else:
        raise NotImplementedError(name)


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1

    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name='customized', args=None, **kwargs):
    if name == 'customized':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from timm.scheduler.cosine_lr import CosineLRScheduler
        return CosineLRScheduler(
            optimizer,
            t_initial=args.train_steps,
            warmup_t=args.warmup_steps,
        )
    else:
        raise NotImplementedError(name)


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


class TrainState(object):
    def __init__(self, optimizer, lr_scheduler, step, nnet=None, nnet_ema=None, args=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.nnet = nnet
        self.nnet_ema = nnet_ema
        self.args = args

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key not in ['step', 'args'] and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))

    def load(self, path, ignored_keys=tuple()):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'), map_location='cpu')
        ignored_keys += ('step', 'args')
        for key, val in self.__dict__.items():
            if key not in ignored_keys and val is not None:
                ckpt = torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu')
                if key in ['nnet', 'nnet_ema']:
                    ckpt = {k: v for k, v in ckpt.items() if 'position_ids' not in k}
                    # execute consume_prefix_in_state_dict_if_present
                    consume_prefix_in_state_dict_if_present(ckpt, 'module.')
                val.load_state_dict(ckpt)
            else:
                logging.info(f'<<load state dict>>: ignore {key}')

    def resume(self, ckpt_root, step=None, ignored_keys=tuple()):
        if not os.path.exists(ckpt_root):
            logger.warning('ckpt_root does not exist when resuming.')
            return
        if ckpt_root.endswith('.ckpt'):
            ckpt_path = ckpt_root
        else:
            if step is None:
                ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
                if not ckpts:
                    return
                steps = map(lambda x: int(x.split(".")[0]), ckpts)
                step = max(steps)
            ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path, ignored_keys=ignored_keys)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def cnt_params(model):
    return sum(param.numel() for param in model.parameters())


def initialize_train_state(device, args):
    params = []

    from libs.models import DiT
    nnet = DiT(
           input_size=args.latent_size,
           num_classes=1000,
           hidden_size=args.embed_dim,
           num_heads=args.num_heads,
           args=args
    )

    params += nnet.parameters()
    nnet_ema = deepcopy(nnet)
    nnet_ema.eval()
    for p in nnet_ema.parameters():
        p.requires_grad = False

    optimizer = get_optimizer(params, name='adamw', lr=args.lr, weight_decay=args.wd, betas=(0.99, 0.99))
    lr_scheduler = get_lr_scheduler(optimizer, args=args, name='customized', warmup_steps=args.warmup_steps)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema, args=args)
    train_state.ema_update(0)
    train_state.to(device)
    return train_state


def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


def sample2dir(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None, dist=True):
    if path:
        os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes if dist else mini_batch_size

    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process,
                            desc='sample2dir'):
        samples = unpreprocess_fn(sample_fn())
        if dist:
            samples = accelerator.gather(samples.contiguous())[:_batch_size]
        if accelerator.is_main_process:
            for sample in samples:
                save_image(sample, os.path.join(path, f"{idx}.png"))
                idx += 1


def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


from collections import defaultdict, deque


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter=" "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if torch.__version__.startswith('2'):
        from torch import inf
    else:
        from torch._six import inf
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def adm_transform(pil_image, is_train=True, args=None):
    pil_image = pil_image.convert("RGB")
    arr = center_crop_arr(pil_image, 256)

    if is_train and random.random() < 0.5:
        arr = arr[:, ::-1]

    arr = arr.astype(np.float32) / 255.
    arr = np.transpose(arr, [2, 0, 1])
    return arr


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def l2_loss(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """
  Args:
    u: (N, T, D) tensor.
    v: (N, T, D) tensor.
  Returns:
    l1_loss: (N,) tensor of summed L1 loss.
  """
  assert u.shape == v.shape, (u.shape, v.shape)
  return ((u - v) ** 2).sum(dim=-1) ** 0.5

def all_gather(data, accelerator):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = accelerator.num_processes
    device = accelerator.device
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device)

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to(device)
    size_list = [torch.LongTensor([0]).to(device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to(device))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to(device)
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list
