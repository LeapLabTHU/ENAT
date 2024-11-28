# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math

from libs.modules import BertEmbeddings, MlmLayer
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class PrevProj(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # norm
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        # mlp
        mlp_hidden_dim = int(hidden_size * 4)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x):
        identity = x
        x = self.mlp(self.norm(x))
        x = x + identity
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.embed_dim = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm1_ = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, multiway_kv=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 8 * hidden_size, bias=True)
        )

    def forward(self, x, c, y=None, prev=None):
        shift_msa, scale_msa, shift_msa_, scale_msa_, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(8, dim=1)
        assert (prev is not None and y is None) or (y is not None and prev is None) or (y is None and prev is None)
        if prev is not None:
            y = torch.cat([prev, x], dim=1)
        if y is not None:
            self.attn.kv.split_position = y.shape[1] - x.shape[1]
            modulated_cond_y = modulate(self.norm1_(y[:, :self.attn.kv.split_position]), shift_msa_, scale_msa_)
            modulated_self_y = modulate(self.norm1(y[:, self.attn.kv.split_position:]), shift_msa, scale_msa)
            modulated_y = torch.cat([modulated_cond_y, modulated_self_y], dim=1)
        else:  # enc, without reuse
            self.attn.kv.split_position = 0
            modulated_y = None
        x = x + gate_msa.unsqueeze(1) * self.attn(x=modulate(self.norm1(x), shift_msa, scale_msa),
                                                  y=modulated_y)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        codebook_size=1024,
        args=None
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads
        self.embed_dim = hidden_size
        self.args = args

        vocab_size = codebook_size + 1
        self.num_vis_tokens = int((input_size) ** 2)
        self.codebook_size = codebook_size

        if args.enc_dec:
            depth_enc, depth_dec = args.enc_dec
            depth = depth_enc + depth_dec

        self.x_embedder = BertEmbeddings(vocab_size=vocab_size,
                                         hidden_size=hidden_size,
                                         max_position_embeddings=self.num_vis_tokens,
                                         dropout=0.1)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.prev_embedder = Mlp(in_features=hidden_size, hidden_features=int(4 * hidden_size),
                                 act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)

        self.prev_proj = PrevProj(hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.mlm_layer = MlmLayer(feat_emb_dim=hidden_size, word_emb_dim=hidden_size, vocab_size=vocab_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def rearranger(self, action, masking, x=None, visible_x=None, masked_x=None):
        assert masking.sum(-1).allclose(masking.sum(-1)[0])
        if action == 'split':
            assert x is not None
            visible_x = x[~masking].reshape(masking.shape[0], -1, self.embed_dim)
            masked_x = x[masking].reshape(masking.shape[0], -1, self.embed_dim)
            split_position = visible_x.shape[1]
            return visible_x, masked_x, split_position
        elif action == 'restore':
            assert visible_x is not None and masked_x is not None
            x = torch.empty(*masking.shape, self.embed_dim, dtype=visible_x.dtype, device=visible_x.device)
            x = x.masked_scatter(~masking.unsqueeze(-1), visible_x)
            x = x.masked_scatter(masking.unsqueeze(-1), masked_x.to(x.dtype))
            return x
        else:
            raise NotImplemented

    def forward(self, masked_ids, context=None, prev_dict=None, return_dict=False):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        return_d = {}

        # embed
        x = self.x_embedder(masked_ids)  # (N, T, D)
        c = self.y_embedder(context.reshape(-1), self.training)  # (N, D)

        # split
        masking = (masked_ids == self.codebook_size)
        return_d['masking'] = masking
        visible_x, masked_x, split_position = self.rearranger('split', masking, x=x)

        # process prev features
        layer_prev_enc = None
        if prev_dict is not None:
            # prepare features
            prev_features = prev_dict['feat'].detach()
            # split & select prev features
            assert torch.all(prev_dict['masking'][masking])
            layer_prev_enc, _, _ = self.rearranger('split', prev_dict['masking'], x=prev_features)
            layer_prev_enc = self.prev_proj(layer_prev_enc).to(x.dtype)
            # for kv cache
            visible_x = x[(~masking) & prev_dict['masking']].reshape(masking.shape[0], -1, self.embed_dim)
            assert torch.all((~masking)[(~prev_dict['masking'])])

        ##############################forward##############################
        enc_depth, dec_depth = self.args.enc_dec
        for enc_block in self.blocks[:enc_depth]:
            visible_x = enc_block(visible_x, c, prev=layer_prev_enc) if visible_x.numel() != 0 else visible_x
        if visible_x.shape[1] != split_position:
            x_pad = torch.empty_like(x)
            x_pad[((~masking) & prev_dict['masking'])] = visible_x.reshape(-1, self.embed_dim)
            x_pad[(~masking) & (~prev_dict['masking'])] = layer_prev_enc.reshape(-1, self.embed_dim)
            visible_x = x_pad[~masking].reshape(masking.shape[0], -1, self.embed_dim)
        # dec forward
        for i, dec_block in enumerate(self.blocks[enc_depth:]):
            masked_x = dec_block(masked_x, c, y=torch.cat([visible_x, masked_x], dim=1))
        ######################################################################

        # restore
        x = self.rearranger('restore', masking, visible_x=visible_x, masked_x=masked_x)

        return_d['feat'] = x
        x = self.norm(x)

        # mlm output
        word_embeddings = self.x_embedder.word_embeddings.weight.data.detach()
        x = self.mlm_layer(x, word_embeddings)
        x = x[..., :self.codebook_size]
        return_d['logits'] = x

        return return_d if return_dict else x


    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
