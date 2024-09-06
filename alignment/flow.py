# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

# Adapted from https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/fe88a163f4661b4ddabba0751ff645e2e620746e/simple_inference.py
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

import sys
if sys.version_info < (3, 9):
    from importlib_resources import files
else:
    from importlib.resources import files
ASSETS_PATH = files("alignment.assets")


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, GaussianFourierProjection
from diffusers.models.unets.unet_2d_blocks import get_down_block, DownBlock2D, CrossAttnDownBlock2D


# https://github.com/huggingface/diffusers/blob/v0.17.1-patch/src/diffusers/models/unet_2d_condition.py
class ConditionalFlow(torch.nn.Module):
    def __init__(self,
        # sample_size: Optional[int] = None,
        in_channels: int = 4,
        # center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        # dual_cross_attention: bool = False,
        # use_linear_projection: bool = False,
        # class_embed_type: Optional[str] = None,
        # addition_embed_type: Optional[str] = None,
        # num_class_embeds: Optional[int] = None,
        # upcast_attention: bool = False,
        # resnet_time_scale_shift: str = "default",
        # resnet_skip_time_act: bool = False,
        # resnet_out_scale_factor: int = 1.0,
        # time_embedding_type: str = "positional",
        # time_embedding_dim: Optional[int] = None,
        # time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        # conv_out_kernel: int = 3,
        # projection_class_embeddings_input_dim: Optional[int] = None,
        class_embeddings_concat: bool = False,
        # mid_block_only_cross_attention: Optional[bool] = None,
        # cross_attention_norm: Optional[str] = None,
        ):

        super().__init__()

        timestep_input_dim = block_out_channels[0]
        self.time_proj = Timesteps(block_out_channels[0],
               flip_sin_to_cos=flip_sin_to_cos, downscale_freq_shift=freq_shift)
        time_embed_dim = block_out_channels[0] * 4
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim,
                act_fn=act_fn, post_act_fn=timestep_post_act, cond_proj_dim=time_cond_proj_dim)

        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )
        self.encoder_hid_proj = None

        self.down_blocks = nn.ModuleList([])
        # only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if class_embeddings_concat:
            # The time embeddings are concatenated with the class embeddings. The dimension of the
            # time embeddings passed to the down, middle, and up blocks is twice the dimension of the
            # regular time embeddings
            blocks_time_embed_dim = time_embed_dim * 2
        else:
            blocks_time_embed_dim = time_embed_dim

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            # is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                # add_downsample=not is_final_block,
                add_downsample=True,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim[i],
                # attn_num_head_channels=attention_head_dim[i], # old diffusers version
                num_attention_heads=attention_head_dim[i],
                attention_head_dim=attention_head_dim[i], # can be annotated
                downsample_padding=downsample_padding,
                # dual_cross_attention=dual_cross_attention,
                # use_linear_projection=use_linear_projection,
                # only_cross_attention=only_cross_attention[i],
                # upcast_attention=upcast_attention,
                # resnet_time_scale_shift=resnet_time_scale_shift,
                # resnet_skip_time_act=resnet_skip_time_act,
                # resnet_out_scale_factor=resnet_out_scale_factor,
                # cross_attention_norm=cross_attention_norm,
            )
            self.down_blocks.append(down_block)

        self.pool = nn.AvgPool2d(4, stride=4) # (bs, 4, 64, 64) -> downsample 4 times -> (bs, ..., 4, 4)
        self.fc = nn.Linear(block_out_channels[-1], 1)

    def forward(self, sample, timesteps, encoder_hidden_states,
                attention_mask: Optional[torch.Tensor] = None,
                cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                ):
        # bs = sample.shape[0]
        dtype = next(self.down_blocks.parameters()).dtype
        # device = next(self.down_blocks.parameters()).device

        # timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=dtype)
        emb = self.time_embedding(t_emb)

        sample = self.conv_in(sample)  # (bs, 320, 64, 64)
        # down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            # down_block_res_samples += res_samples

        sample = self.pool(sample)
        sample = sample.view(sample.size(0), -1)
        sample = self.fc(sample).squeeze()
        return sample
