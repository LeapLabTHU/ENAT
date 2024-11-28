import argparse
import numpy as np
from accelerate import DistributedDataParallelKwargs
from torch._C import _distributed_c10d

_distributed_c10d.set_debug_level(_distributed_c10d.DebugLevel.INFO)
import os
import time

from torch.utils._pytree import tree_map

import accelerate
import torch

from dataset import ImageNetFeatures
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from libs.nat_misc import NATSchedule
import taming.models.vqgan

import utils
from torch.nn.functional import adaptive_avg_pool2d
from libs.inception import InceptionV3


def get_args():
    parser = argparse.ArgumentParser()
    # basics
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--mixed_precision', type=str, default='fp16')
    parser.add_argument('--input_res', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--class_num', type=int, default=1000)
    # training
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--wd', type=float, default=0.03)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--train_steps', type=int, default=500000)
    parser.add_argument('--ema_rate', type=float, default=0.9999)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--reference_image_path', type=str, default='assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz')
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    # evaluation
    parser.add_argument('--eval_n', type=int, default=50000, help='number of samples for evaluation')
    parser.add_argument('--test_bsz', type=int, default=25)
    parser.add_argument('--pretrained_path', type=str, default=None)  # for finetuning
    parser.add_argument('--gen_steps', type=int, default=8)
    parser.add_argument('--cfg_scale', type=float)
    parser.add_argument('--mask_temp', type=float)
    parser.add_argument('--samp_temp', type=float)
    # enat model configuration
    parser.add_argument('--enc_dec', type=int, nargs='+')
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--num_heads', type=int)
    # enat reuse
    parser.add_argument('--reuse_prob', type=float, default=0.5)
    parser.add_argument('--ema_for_cond', type=int, default=1)
    args = parser.parse_args()

    args.latent_size = args.input_res // 16
    args.seq_len = args.latent_size ** 2
    return args


def mask_by_confidence(mask_len, confidence_map=None, ignore_map=None, temp=None):
    if confidence_map is None:
        confidence_map = torch.rand(mask_len.shape[0], args.seq_len, device=device)
        assert temp is None
    if ignore_map is not None:
        confidence_map = torch.where(ignore_map, +np.inf, confidence_map)
    sorted_inds = torch.argsort(confidence_map, dim=-1, descending=False)
    mask_len = torch.clamp(mask_len, 0, args.seq_len - 1)
    assert mask_len.allclose(mask_len[0])
    mask_len = mask_len[0]
    inds_to_mask = sorted_inds[:, :mask_len.long()]  # mask out low confidence
    masking = torch.zeros_like(confidence_map, dtype=torch.bool)
    masking.scatter_(1, inds_to_mask, True)
    return masking

def LSimple(x0, nnet, schedule, nnet_ema=None, no_reuse=False, **kwargs):
    mask_ratios, labels, xn = schedule.sample(x0)
    if torch.rand(1).item() < args.reuse_prob and not no_reuse:
        # use ema model or not
        if not args.ema_for_cond:
            nnet_ema = nnet
        # prepare prev_features
        with torch.no_grad():
            masking = (xn == schedule.mask_ind)
            visible_len = (~masking).sum(dim=-1)
            assert visible_len.allclose(visible_len[0])
            vis2mask_len = (visible_len * torch.rand(1, device=device))
            vis2mask_len = vis2mask_len.round().long()
            prev_xn_mask = mask_by_confidence(mask_len=vis2mask_len, ignore_map=masking)
            assert not torch.any(prev_xn_mask & masking)
            _xn = torch.where(prev_xn_mask, schedule.mask_ind, xn)
            prev_dict = nnet_ema(_xn, **kwargs, return_dict=True)
    else:
        prev_dict = None
    pred = nnet(xn, **kwargs, prev_dict=prev_dict)
    loss = schedule.loss(pred, labels)
    masked_token_ratio = xn.eq(schedule.mask_ind).sum().item() / xn.shape[0] / xn.shape[1]
    return loss, {'masked_token_ratio': masked_token_ratio}


@logger.catch()
def train():
    # prepare for fid calc
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device)
    inception.eval()
    inception.requires_grad_(False)
    # load npz file
    with np.load(args.reference_image_path) as f:
        m2, s2 = f['mu'][:], f['sigma'][:]
        m2, s2 = torch.from_numpy(m2).to(device), torch.from_numpy(s2).to(device)

    autoencoder = taming.models.vqgan.get_model()
    autoencoder.to(device)
    # load npy dataset & dataloader
    dataset = ImageNetFeatures(path=f'assets/imagenet{args.input_res}_vq_features', cfg=True, p_uncond=0.15)
    train_dataset = dataset.get_split(split='train', labeled=True)
    train_dataset_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True,
                                      num_workers=8, pin_memory=True, persistent_workers=True
                                      )
    # initialize train state
    train_state = utils.initialize_train_state(device, args)
    nnet, nnet_ema, optimizer, train_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader)
    assert len(optimizer.param_groups) == 1
    lr_scheduler = train_state.lr_scheduler
    if args.pretrained_path is None:
        train_state.resume(ckpt_root)

    @torch.cuda.amp.autocast(enabled=(accelerator.mixed_precision == 'fp16'))
    def decode(_batch):
        return autoencoder.decode_code(_batch)

    def get_data_generator():
        while True:
            for data in train_dataset_loader:
                yield data

    data_generator = get_data_generator()

    schedule = NATSchedule(codebook_size=autoencoder.n_embed, device=device, args=args)

    def train_step(_batch):
        optimizer.zero_grad()
        with torch.no_grad():
            _z = _batch[0]
            context = _batch[1]
        loss, lsimple_metrics = LSimple(_z, nnet, schedule, nnet_ema=nnet_ema, context=context)
        metric_logger.update(loss=accelerator.gather(loss.detach()).mean())
        metric_logger.update(**lsimple_metrics)
        accelerator.backward(loss.mean())
        optimizer.step()
        lr_scheduler.step(train_state.step)
        train_state.ema_update(args.ema_rate)
        metric_logger.update(max_mem=torch.cuda.max_memory_allocated() / 1024 / 1024)
        metric_logger.update(loss_scaler=accelerator.scaler.get_scale() if accelerator.scaler is not None else 1.)
        metric_logger.update(grad_norm=utils.get_grad_norm_(optimizer.param_groups[0]['params']))
        train_state.step += 1

    @torch.no_grad()
    def eval_step(n_samples, sample_steps, **kwargs):
        logger.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}'
                     f'mini_batch_size={args.test_bsz}')
        batch_size = args.test_bsz * accelerator.num_processes

        class_label_gen_world = np.arange(0, args.class_num).repeat(args.eval_n // args.class_num)
        class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(args.eval_n)])
        world_size = accelerator.num_processes
        local_rank = accelerator.process_index

        idx = 0
        pred_tensor = torch.empty((n_samples, 2048), device=device)
        for i, _batch_size in enumerate(tqdm(utils.amortize(n_samples, batch_size), disable=not accelerator.is_main_process,
                                   desc='sample2dir')):
            contexts = class_label_gen_world[world_size * args.test_bsz * i + local_rank * args.test_bsz:
                                             world_size * args.test_bsz * i + (local_rank + 1) * args.test_bsz]
            contexts = torch.Tensor(contexts).long().cuda()
            samples = schedule.generate(sample_steps, len(contexts), nnet_ema, decode, context=contexts, latent_size=args.latent_size, **kwargs)
            samples = samples.clamp_(0., 1.)

            pred = inception(samples.float())[0]

            # Apply global spatial average pooling if needed
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2)
            pred_tensor[idx:idx + pred.shape[0]] = pred

            idx = idx + pred.shape[0]

        pred_tensor = pred_tensor[:idx].to(device)
        pred_tensor = accelerator.gather(pred_tensor)

        pred_tensor = pred_tensor[:n_samples]

        m1 = torch.mean(pred_tensor, dim=0)
        pred_centered = pred_tensor - pred_tensor.mean(dim=0)
        s1 = torch.mm(pred_centered.T, pred_centered) / (pred_tensor.size(0) - 1)

        m1 = m1.double()
        s1 = s1.double()

        a = (m1 - m2).square().sum(dim=-1)
        b = s1.trace() + s2.trace()
        c = torch.linalg.eigvals(s1 @ s2).sqrt().real.sum(dim=-1)

        _fid = (a + b - 2 * c).item()

        if accelerator.is_main_process:
            logger.info(f'FID{n_samples}={_fid}, global_step={train_state.step}')

    logger.info(f'Start fitting, step={train_state.step}')

    metric_logger = utils.MetricLogger()
    if args.pretrained_path:
        nnet_ema.load_state_dict(torch.load(args.pretrained_path, map_location='cpu'))
        eval_step(n_samples=args.eval_n, sample_steps=args.gen_steps)
        return

    nnet.train()
    while train_state.step < args.train_steps:
        data_time_start = time.time()
        batch = next(data_generator)
        if isinstance(batch, list):
            batch = tree_map(lambda x: x.to(device), next(data_generator))
        else:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        metric_logger.update(data_time=time.time() - data_time_start)
        train_step(batch)

        if train_state.step % args.save_interval == 0 or train_state.step == args.train_steps:
            torch.cuda.empty_cache()
            logger.info(f'Save checkpoint {train_state.step}...')
            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(ckpt_root, f'{train_state.step}.ckpt'))
        accelerator.wait_for_everyone()

        if accelerator.is_main_process and train_state.step % args.log_interval == 0:
            logger.info(f'step: {train_state.step} {metric_logger}')

    logger.info(f'Finish fitting, step={train_state.step}')


if __name__ == "__main__":
    args = get_args()
    ckpt_root = os.path.join(args.output_dir, 'ckpts')

    # prepare accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=args.mixed_precision)
    logger.add(os.path.join(args.output_dir, 'output.log'), level='INFO')
    device = accelerator.device
    accelerate.utils.set_seed(args.seed, device_specific=True)
    logger.info(f'Process {accelerator.process_index} using device: {device}')
    logger.info('Using mixed precision: {}'.format(accelerator.mixed_precision))
    assert args.batch_size % accelerator.num_processes == 0
    mini_batch_size = args.batch_size // accelerator.num_processes
    args.batch_size = mini_batch_size
    logger.info(f'Using mini-batch size {mini_batch_size} per device')

    if accelerator.is_main_process:
        os.makedirs(ckpt_root, exist_ok=True)

    train()
