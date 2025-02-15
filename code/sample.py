#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import matplotlib.pyplot as plt
import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional
from PIL import Image
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, LMSDiscreteScheduler
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from TBC_dataset import TBC_Bench_Single_Story
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0
)
import torch.nn as nn
if is_wandb_available():
    import wandb
from typing import Optional, Union, Tuple, List, Callable, Dict
attn_maps = {}
from train import Autok_encoder, validation
from dataclasses import dataclass
from diffusers.utils import deprecate, BaseOutput, is_torch_version, logging
from einops import rearrange
import cv2
is_cond = True
from glob import glob
import random


import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def th2image(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.28.0.dev0")

logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def validation(example, tokenizer, text_encoder, unet, vae, k_encoder,scheduler,device, guidance_scale, seed=None, num_steps=100):
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    uncond_input = tokenizer(
        [''],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
        ).input_ids[0]
    uncond_input = uncond_input.unsqueeze(0).repeat(len(example['text']),1)

    uncond_embeddings = text_encoder(uncond_input.to(device),return_dict=False)[0]
    uncond_indice = example["character_desc_index"]
    if seed is None:
        latents = torch.randn(
            (len(example["text"]), unet.config.in_channels, 64, 112)
        )
    else:
        generator = torch.manual_seed(seed)
        latents = torch.randn(
            (len(example["text"]), unet.config.in_channels, 64, 112), generator=generator,
        )

    latents = latents.to(uncond_embeddings)
    scheduler.set_timesteps(num_steps)
    latents = latents * scheduler.init_noise_sigma

    uncond_mask = torch.ones((len(example['text']),512,896))
    uncond_mask = [uncond_mask, uncond_mask]
    uncond_encoder_hidden_states =  {
        'text_feat': uncond_embeddings,
        'character_indice': uncond_indice,
        'mask': uncond_mask,
    }


    for t in tqdm(scheduler.timesteps):
        # print(t)
        global is_cond,is_last_few_layers
        if t <= 0:
            is_last_few_layers=True
        else:
            is_last_few_layers=False
        is_cond = True
        cond_mask, all_mean, all_cor, all_sum_mean = k_encoder(example['character_list'], example['character_desc_index'], example['mean'], example['cor'], text_encoder, t.unsqueeze(0).long().repeat(len(example['text'])))
        # print(cond_mask[0].shape)
        latent_model_input = scheduler.scale_model_input(latents, t)
        noise_pred_text = unet(
            latent_model_input,
            t,
            encoder_hidden_states={
        'text_feat': text_encoder(example["input_ids"].to(text_encoder.device), return_dict=False)[0],
        'character_indice': example["character_desc_index"],
        'mask': cond_mask
    },
        ).sample
        latent_model_input = scheduler.scale_model_input(latents, t)
        is_cond = False
        noise_pred_uncond = unet(
            latent_model_input,
            t,
            encoder_hidden_states=uncond_encoder_hidden_states,
        ).sample
        noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    _latents = 1 / 0.18215 * latents.clone()
    images = vae.decode(_latents).sample
    ret_pil_images = [th2image(image) for image in images]
    return ret_pil_images, cond_mask


def get_attention_scores(
        self, query: torch.Tensor, key: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
    r"""
    Compute the attention scores.

    Args:
        query (`torch.Tensor`): The query tensor.
        key (`torch.Tensor`): The key tensor.
        attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

    Returns:
        `torch.Tensor`: The attention probabilities/scores.
    """
    dtype = query.dtype
    if self.upcast_attention:
        query = query.float()
        key = key.float()

    if attention_mask is None:
        baddbmm_input = torch.empty(
            query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        )
        beta = 0
    else:
        baddbmm_input = attention_mask
        beta = 1

    attention_scores = torch.baddbmm(
        baddbmm_input,
        query,
        key.transpose(-1, -2),
        beta=beta,
        alpha=self.scale,
    )
    del baddbmm_input

    if self.upcast_softmax:
        attention_scores = attention_scores.float()

    attention_probs = attention_scores.softmax(dim=-1)

    attention_probs = attention_probs.to(dtype)

    return attention_probs, attention_scores
@dataclass
class Transformer2DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: torch.FloatTensor

def Transformer2DModelForward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    added_cond_kwargs: Dict[str, torch.Tensor] = None,
    class_labels: Optional[torch.LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
):
    """
    The [`Transformer2DModel`] forward method.

    Args:
        hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
            Input `hidden_states`.
        encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
            Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
            self-attention.
        timestep ( `torch.LongTensor`, *optional*):
            Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
        class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
            Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
            `AdaLayerZeroNorm`.
        cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        attention_mask ( `torch.Tensor`, *optional*):
            An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
            is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
            negative values to the attention scores corresponding to "discard" tokens.
        encoder_attention_mask ( `torch.Tensor`, *optional*):
            Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                * Mask `(batch, sequence_length)` True = keep, False = discard.
                * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

            If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
            above. This bias will be added to the cross-attention scores.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
            tuple.

    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """
    if cross_attention_kwargs is not None:
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
    # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
    #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
    #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
    # expects mask of shape:
    #   [batch, key_tokens]
    # adds singleton query_tokens dimension:
    #   [batch,                    1, key_tokens]
    # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
    #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
    #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
    if attention_mask is not None and attention_mask.ndim == 2:
        # assume that mask is expressed as:
        #   (1 = keep,      0 = discard)
        # convert mask into a bias that can be added to attention scores:
        #       (keep = +0,     discard = -10000.0)
        attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # 1. Input
    if self.is_input_continuous:
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim) # hidden_states.shape == (2,320,96,96) -> (2,9216,320)
            hidden_states = self.proj_in(hidden_states)

    elif self.is_input_vectorized:
        hidden_states = self.latent_image_embedding(hidden_states)
    elif self.is_input_patches:
        height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
        hidden_states = self.pos_embed(hidden_states)

        if self.adaln_single is not None:
            if self.use_additional_conditions and added_cond_kwargs is None:
                raise ValueError(
                    "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                )
            batch_size = hidden_states.shape[0]
            timestep, embedded_timestep = self.adaln_single(
                timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )

    ####################################################################################################
    if cross_attention_kwargs is None:
        cross_attention_kwargs = {}
    cross_attention_kwargs['height'] = height
    cross_attention_kwargs['width'] = width
    ####################################################################################################

    # # 2. Blocks
    # if self.caption_projection is not None:
    #     batch_size = hidden_states.shape[0]
    #     encoder_hidden_states = self.caption_projection(encoder_hidden_states)
    #     encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

    for block in self.transformer_blocks:
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                timestep,
                cross_attention_kwargs,
                class_labels,
                **ckpt_kwargs,
            )
        else:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )

    # 3. Output
    if self.is_input_continuous:
        if not self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual
    elif self.is_input_vectorized:
        hidden_states = self.norm_out(hidden_states)
        logits = self.out(hidden_states)
        # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
        logits = logits.permute(0, 2, 1)

        # log(p(x_0))
        output = F.log_softmax(logits.double(), dim=1).float()

    if self.is_input_patches:
        if self.config.norm_type != "ada_norm_single":
            conditioning = self.transformer_blocks[0].norm1.emb(
                timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
            shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            hidden_states = self.proj_out_2(hidden_states)
        elif self.config.norm_type == "ada_norm_single":
            shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states)
            # Modulation
            hidden_states = hidden_states * (1 + scale) + shift
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.squeeze(1)

        # unpatchify
        if self.adaln_single is None:
            height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
        )

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)

def attn_call(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
        height=None,
        width=None,
    ):
    assert attention_mask is None
    residual = hidden_states

    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states["text_feat"].shape
    )
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    # query = attn.to_q(hidden_states, scale=scale)
    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        context_tensor = hidden_states
    elif encoder_hidden_states is not None:
        context_tensor = encoder_hidden_states["text_feat"]
    if  encoder_hidden_states is not None and attn.norm_cross:
        context_tensor = attn.norm_encoder_hidden_states(context_tensor)

    # key = attn.to_k(encoder_hidden_states, scale=scale)
    key = attn.to_k(context_tensor)
    # value = attn.to_v(encoder_hidden_states, scale=scale)
    value = attn.to_v(context_tensor)

    query = attn.head_to_batch_dim(query)
    key = attn.head_to_batch_dim(key)
    value = attn.head_to_batch_dim(value)

    attention_probs, attention_scores = attn.get_attention_scores(query, key, attention_mask)

    with torch.autograd.set_detect_anomaly(True):
    ####################################################################################################
    # (20,4096,77) or (40,1024,77)
        if encoder_hidden_states is not None:
            new_attention_scores = attention_scores.clone()
            B = attention_probs.shape[0]
            HW = attention_probs.shape[1]
            mini_num = math.sqrt(HW/4/7)
            mini_num = int(mini_num)
            H = mini_num * 4
            W = mini_num * 7
            char_num = len(encoder_hidden_states['mask'])
            for idx in range(char_num):
                for bsz in range(batch_size):
                    per_char_mask = encoder_hidden_states['mask'][idx][[bsz],:,:].unsqueeze(0)
                    per_char_mask = F.interpolate(per_char_mask, size=(H, W), mode='nearest').to(attention_scores.device)
                    per_char_mask = per_char_mask.reshape(HW,1)
                    if per_char_mask.min() == 1 and per_char_mask.max() == 1:
                        continue

                    class_start_index = encoder_hidden_states['character_indice'][idx][2][bsz]
                    if encoder_hidden_states['character_indice'][idx][1][bsz] <= 77:
                        class_end_index = encoder_hidden_states['character_indice'][idx][3][bsz]
                    else:
                        class_end_index = 77
                    class_indices = torch.arange(class_start_index, class_end_index)
                    selected_class = new_attention_scores[bsz,:,class_indices].clone()
                    selected_class = selected_class.clone() + per_char_mask                  
                    new_attention_scores[bsz,:,class_indices] = selected_class

                    desc_start_index = encoder_hidden_states['character_indice'][idx][0][bsz]
                    if encoder_hidden_states['character_indice'][idx][1][bsz] <= 77:
                        desc_end_index = encoder_hidden_states['character_indice'][idx][1][bsz]
                    else:
                        desc_end_index = 77
                    desc_indices = torch.arange(desc_start_index, desc_end_index)
                    selected_desc = new_attention_scores[bsz,:,desc_indices].clone()
                    selected_desc = selected_desc.clone() + per_char_mask                   
                    new_attention_scores[bsz,:,desc_indices] = selected_desc
            attention_probs = new_attention_scores.softmax(dim=-1)
            global is_cond
            if is_cond == True:
                self.attn_map = rearrange(attention_probs, 'b (h w) d -> b d h w', h=height) # (10,9216,77) -> (10,77,96,96)
            
    ####################################################################################################
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    # hidden_states = attn.to_out[0](hidden_states, scale=scale)
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states




def hook_fn(name):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            # attn_maps[name] = module.processor.attn_map
            global attn_maps,is_last_few_layers
            attn_maps[name] = module.processor.attn_map
            # attn_maps[name] = attn_maps.get(name, torch.zeros_like(module.processor.attn_map)) + module.processor.attn_map
            # global cnt
            # if cnt < 32*3:
            #     attn_maps[name] = attn_maps.get(name, torch.zeros_like(module.processor.attn_map)) + module.processor.attn_map
            # else:
            #     attn_maps[name] = attn_maps.get(name, torch.zeros_like(module.processor.attn_map)) 
            # cnt += 1
            del module.processor.attn_map

    return forward_hook


def register_cross_attention_hook(unet):
    for name, module in unet.named_modules():
        if not name.split('.')[-1].startswith('attn2'):
            continue

        if isinstance(module.processor, AttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, AttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor2_0):
            module.processor.store_attn_map = True

        hook = module.register_forward_hook(hook_fn(name))
    
    return unet


def set_layer_with_name_and_path(model, target_name="attn2", current_path=""):
    from diffusers.models import Transformer2DModel
    for name, layer in model.named_children():
        if layer.__class__ == Transformer2DModel:
            layer.forward = Transformer2DModelForward.__get__(layer, Transformer2DModel)

        new_path = current_path + '.' + name if current_path else name
        if name.endswith(target_name):
            layer.processor = AttnProcessor2_0()
        
        set_layer_with_name_and_path(layer, target_name, new_path)
    
    return model


def prompt2tokens(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    tokens = []
    for text_input_id in text_input_ids[0]:
        token = tokenizer.decoder[text_input_id.item()]
        tokens.append(token)
    return tokens


def preprocess(max_height=256, max_width=256):
    # max_height, max_width = 0, 0
    global attn_maps
    for k,v in attn_maps.items():
        v = torch.mean(v.cpu(),axis=0).squeeze(0)
        _, h, w = v.shape
        max_height = max(h, max_height)
        max_width = max(w, max_width)
        v = F.interpolate(
            v.to(dtype=torch.float32).unsqueeze(0),
            size=(max_height, max_width),
            mode='bilinear',
            align_corners=False
        ).squeeze(0) # (77,64,64)
        attn_maps[k] = v
    
    attn_map = torch.stack(list(attn_maps.values()), axis=0)
    attn_map = torch.mean(attn_map, axis=0)

    return attn_map


def visualize_and_save_attn_map(attn_map, tokenizer, prompt,save_path = 'attn_maps'):
    # match with tokens
    tokens = prompt2tokens(tokenizer, prompt)
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    pad_token = tokenizer.pad_token
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    # to_pil = transforms.ToPILImage()
    for i, (token, token_attn_map) in enumerate(zip(tokens, attn_map)):
        if token == bos_token:
            continue
        if token == eos_token:
            break
        token = token.replace('</w>','')
        token = f'{i}_<{token}>.jpg'

        # low quality
        # to_pil(255 * token_attn_map).save(os.path.join(save_path, token))
        # to_pil(255 * (token_attn_map - torch.min(token_attn_map)) / (torch.max(token_attn_map) - torch.min(token_attn_map))).save(os.path.join(save_path, token))

        token_attn_map = token_attn_map.cpu().numpy()
        normalized_token_attn_map = (token_attn_map - np.min(token_attn_map)) / (np.max(token_attn_map) - np.min(token_attn_map)) * 255
        normalized_token_attn_map = normalized_token_attn_map.astype(np.uint8)
        image = Image.fromarray(normalized_token_attn_map)
        image.save(os.path.join(save_path, token))


def visualize_and_save_attn_map_v2(attn_map, tokenizer, prompt,save_path = 'attn_maps.png', ):
    # match with tokens
    tokens = prompt2tokens(tokenizer, prompt)
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    pad_token = tokenizer.pad_token
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    images = []
    # to_pil = transforms.ToPILImage()
    for i, (token, token_attn_map) in enumerate(zip(tokens, attn_map)):
        token = token.replace('</w>','')
        token_attn_map = token_attn_map.cpu().numpy()
        normalized_token_attn_map = (token_attn_map - np.min(token_attn_map)) / (np.max(token_attn_map) - np.min(token_attn_map)) * 255
        normalized_token_attn_map = np.array(Image.fromarray(normalized_token_attn_map).resize((128, 128)))
        normalized_token_attn_map = np.repeat(np.expand_dims(normalized_token_attn_map, axis=2), 3, axis=2).astype(np.uint8)
        image = text_under_image(normalized_token_attn_map,token)
        images.append(image)

    image_ = view_images(np.stack(images, axis=0))
    attn_image = Image.fromarray(image_)
    attn_image.save(os.path.join(save_path))

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img

def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    return image_

def cross_attn_init():
    Attention.get_attention_scores = get_attention_scores
    AttnProcessor.__call__ = attn_call
    AttnProcessor2_0.__call__ = attn_call # attn_call is faster

def th2image(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)

def save_model_card(
    args,
    repo_id: str,
    images: list = None,
    repo_folder: str = None,
):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(args.validation_prompts))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    model_description = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {args.validation_prompts}: \n
{img_str}

## Pipeline usage

You can use the pipeline like so:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
prompt = "{args.validation_prompts[0]}"
image = pipeline(prompt).images[0]
image.save("my_image.png")
```

## Training info

These are the key hyperparameters used during training:

* Epochs: {args.num_train_epochs}
* Learning rate: {args.learning_rate}
* Batch size: {args.train_batch_size}
* Gradient accumulation steps: {args.gradient_accumulation_steps}
* Image resolution: {args.resolution}
* Mixed-precision: {args.mixed_precision}

"""
    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
"""

    model_description += wandb_info

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=args.pretrained_model_name_or_path,
        model_description=model_description,
        inference=True,
    )

    tags = ["stable-diffusion", "stable-diffusion-diffusers", "text-to-image", "diffusers", "diffusers-training"]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--k_encoder_dir",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default='Pororo',
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--style_dir",
        type=str,
        default=None,
        help=(
           "The txt file of the style description."
        ),
    )
    parser.add_argument(
        "--caption_dir",
        type=str,
        default=None,
        help=(
            "The file dir to the character caption."
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default='Pororo',
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--ckpt_rank",
        type=str,
        default=None,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--max_char", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--seed_num", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    os.makedirs(args.output_dir, exist_ok = True)
    return args


def main():
    cross_attn_init()
    args = parse_args()
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", 
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder",
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", 
        )
    unet = UNet2DConditionModel.from_pretrained(
        args.ckpt_rank, subfolder="unet", 
    )
    k_encoder = torch.load(os.path.join(args.ckpt_rank, 'k_encoder.pt')).to(args.device)
    # Freeze vae and text_encoder and set unet to trainable
    k_encoder.set_scale(args.scale) if args.scale is not None else k_encoder.set_scale()
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    k_encoder.requires_grad_(False)
    # Preprocessing the datasets.
    # Eval dataset creation

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(args.device, dtype=weight_dtype)
    vae.to(args.device, dtype=weight_dtype)
    unet.to(args.device, dtype=weight_dtype)
    unet = set_layer_with_name_and_path(unet)
    unet = register_cross_attention_hook(unet)
    story_path = args.dataset_dir
    os.makedirs(args.output_dir,exist_ok = True)
    random_seeds = random.sample(range(1, 1024), args.seed_num)
    for each_seed in random_seeds:
        per_output_dir = os.path.join(args.output_dir, f'seed_{str(each_seed)}')
        os.makedirs(per_output_dir,exist_ok = True)
        os.makedirs(os.path.join(per_output_dir, f'image'),exist_ok = True)
        os.makedirs(os.path.join(per_output_dir, f'text'),exist_ok = True)
        eval_dataset = TBC_Bench_Single_Story(
            data_root=story_path,
            tokenizer=tokenizer,
            resolution=512,
            caption_dir=args.caption_dir,
            max_char = args.max_char,
            style_dir=args.style_dir,
            type=args.data_name,
            )

        # Eval DataLoaders creation:
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            shuffle=False,
            batch_size=1,
            num_workers=args.dataloader_num_workers,
        )
        
        for step, batch in enumerate(eval_dataloader):
            global attn_maps
            attn_maps = {}
            batch['input_ids'] = batch['input_ids'].to(args.device)
            img_list = []
            syn_images,_ = validation(batch, tokenizer, text_encoder, unet, vae, k_encoder, noise_scheduler,batch["input_ids"].device, 5 ,each_seed, 100)
            for syn in syn_images:
                img_list.append(np.array(syn))
            img_list = np.concatenate(img_list, axis=1)
            Image.fromarray(img_list).save(os.path.join(per_output_dir, f"image/{str(step).zfill(5)}.jpg"))
            with open(os.path.join(per_output_dir, f"text/{str(step).zfill(5)}.txt") , 'w') as f:
                for each_line in batch['text']:
                    f.writelines(each_line + '\n')


if __name__ == "__main__":
    main()
