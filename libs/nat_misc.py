import numpy as np
import torch
import math
from einops import rearrange
from torch.nn import functional as F

def add_gumbel_noise(t, temperature, device):
    return (t + torch.Tensor(temperature * np.random.gumbel(size=t.shape)).to(device))


class NATSchedule(object):
    def __init__(self, codebook_size, device, ignore_ind=-1, smoothing=0.1, beta_alpha_beta=(12, 3), args=None):
        self.mask_ind = codebook_size  # for input masking
        self.ignore_ind = ignore_ind  # for ce loss, excluding visible
        self.device = device
        self.smoothing = smoothing
        self.beta_a, self.beta_b = beta_alpha_beta
        self.args = args

    @staticmethod
    def cosine_schedule(t):
        return torch.cos(t * math.pi * 0.5)

    def sample(self, x0):
        N, L, device = *x0.shape, self.device
        beta_dist = torch.distributions.Beta(self.beta_a, self.beta_b)
        rand_mask_probs = beta_dist.sample((1,)).to(device).float()
        rand_mask_probs = torch.full((N,), rand_mask_probs[0], device=device)
        batch_randperm = torch.rand(N, L, device=device).argsort(dim=-1)
        num_token_masked = (L * rand_mask_probs).round().clamp(min=1, max=self.args.seq_len - 1)
        mask = batch_randperm < rearrange(num_token_masked, 'b -> b 1')
        masked_ids = torch.where(mask, self.mask_ind, x0)
        labels = torch.where(mask, x0, self.ignore_ind)
        return rand_mask_probs, labels, masked_ids  # timestep is not needed for nnet

    def loss(self, pred, label):  # pred: N, L, C
        return F.cross_entropy(pred.transpose(1, 2), label.long(),
                               ignore_index=self.ignore_ind, label_smoothing=self.smoothing)

    @torch.no_grad()
    def generate(self, sample_steps, _n_samples, nnet, decode_fn, latent_size, **kwargs):
        _sample_steps, device = sample_steps, self.device

        fmap_size = (latent_size, latent_size)

        seq_len = fmap_size[0] * fmap_size[1]

        ids = torch.full((_n_samples, seq_len), self.mask_ind, dtype=torch.long, device=device)
        empty_ctx = torch.full((_n_samples, 1), 1000, dtype=torch.long, device=device)

        cfg_scale = 0.
        cond_state = uncond_state = None
        for step in range(_sample_steps):
            # get plain ratio
            ratio = 1. * (step + 1) / _sample_steps
            # get mask ratio
            mask_ratio = np.cos(ratio * math.pi * 0.5)
            # scaling temp
            annealed_temp = self.args.mask_temp * (1 - ratio)
            annealed_samp_temp = max(self.args.samp_temp * (1-ratio), 1e-3)
            # sampling & scoring
            cond_state = nnet(ids, **kwargs, return_dict=True, prev_dict=cond_state)
            uncond_state = nnet(ids, context=empty_ctx, return_dict=True, prev_dict=uncond_state)
            logits = cond_state['logits'] + cfg_scale * (cond_state['logits'] - uncond_state['logits'])
            logits = torch.log_softmax(logits, dim=-1)
            is_mask = (ids == self.mask_ind)
            sampled_ids = torch.where(is_mask,
                                      torch.distributions.Categorical(logits=logits / annealed_samp_temp).sample(),
                                      ids)
            sampled_logits = torch.where(is_mask,
                                         torch.gather(logits, dim=-1, index=sampled_ids.unsqueeze(-1)).squeeze(-1),
                                         +np.inf).float()
            # masking
            confidence = add_gumbel_noise(sampled_logits, annealed_temp, device)
            sorted_inds = confidence.argsort(dim=-1, descending=False)
            inds_to_mask = sorted_inds[:, :math.floor(seq_len * mask_ratio)]  # mask out low confidence
            new_masking = torch.zeros_like(ids, dtype=torch.bool)
            new_masking.scatter_(1, inds_to_mask, True)
            ids = torch.where(new_masking, self.mask_ind, sampled_ids)
            cfg_scale = ratio * self.args.cfg_scale

        _z = rearrange(sampled_ids, 'b (i j) -> b i j', i=fmap_size[0], j=fmap_size[1])
        out = decode_fn(_z)

        return out
