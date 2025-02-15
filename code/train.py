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

import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
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
from TBC_dataset import BenchMark_Each_ClassConstrains, Benchmark_Eval_ClassConstrains
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0
)
if is_wandb_available():
    import wandb
import torch.nn as nn
import itertools
import matplotlib.pyplot as plt
import cv2
from einops import rearrange
is_cond = False
from typing import Optional, Union, Tuple, List, Callable, Dict
attn_maps = {}
from dataclasses import dataclass
from diffusers.utils import deprecate, BaseOutput, is_torch_version
from typing import Any, Dict, Optional

def hook_fn(name):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            # attn_maps[name] = module.processor.attn_map
            global attn_maps
            attn_maps[name] = attn_maps.get(name, torch.zeros_like(module.processor.attn_map)) + module.processor.attn_map
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

def set_tensor_values_to_zero(tensor0, tensor1):
    mask = torch.all(torch.eq(tensor0, tensor1), dim=-1)
    tensor0[mask] = 0.0
    return tensor0

def calculate_cos_sim(mask_list):
    similarity = []
    for i in range(1,len(mask_list)):
        sim = torch.cosine_similarity(mask_list[0], mask_list[i], dim=-1)
        similarity.append(sim)
    similarity = torch.stack(similarity)
    return similarity.mean(dim=0).mean()

def th2image(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)

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
        global is_cond
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
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.28.0.dev0")

class Autok_encoder(nn.Module):
    def __init__(self, in_dim, out_dim,Height=512, Width=896, scale=1.0):
        super(Autok_encoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.H = Height
        self.W = Width
        self.scale = scale
        image_size = (Width, Height)

        x = torch.linspace(-1, 1, image_size[1])
        y = torch.linspace(-1, 1, image_size[0])
        grid_x, grid_y = torch.meshgrid(x, y)
        self.grid = torch.stack((grid_x, grid_y), dim=2)
        setattr(self, f'mean_predictor', nn.Sequential(nn.Linear(in_dim, 64),
                                         nn.LeakyReLU(),
                                         nn.LayerNorm(64),
                                         nn.Linear(64, out_dim),
                                         nn.Tanh(),),                                         
                                         )

        setattr(self, f'cor_predictor', nn.Sequential(nn.Linear(in_dim, 64),
                                         nn.LeakyReLU(),
                                         nn.LayerNorm(64),
                                         nn.Linear(64, out_dim),
                                         nn.Tanh()))
    def set_scale(self, scale=1.0):
        self.scale = scale
    def forward(self, char_input_ids, char_indice_list, prior_means, prior_cors, text_encoder,timesteps):
        mask = []
        all_mean = []
        all_cor = []
        all_sum_mean = []

        prior_means = prior_means.to(text_encoder.device)
        prior_cors = prior_cors.to(text_encoder.device)
        time_aware_strength = self.scale * (torch.log(timesteps + 1) + 1)
        # print(time_aware_strength)
        for idx, (each_ids, each_indice) in enumerate(zip(char_input_ids, char_indice_list)):
            bs = char_input_ids[0].shape[0]
            each_feat = text_encoder(each_ids.to(text_encoder.device), return_dict=False)[1]
            per_mean = self.mean_predictor(each_feat)
            per_cor =self.cor_predictor(each_feat)
            all_mean.append(per_mean)
            all_cor.append(per_cor)

            per_cor = (per_cor+1.).unsqueeze(2) * torch.eye(2).to(each_feat.device)
            mean = per_mean + prior_means[:,idx,:]
            cor = (per_cor) * prior_cors[:,idx,:,:]
            all_sum_mean.append(mean)
            gaussian = torch.distributions.MultivariateNormal(mean, cor)
            grids = self.grid.unsqueeze(-2).repeat(1,1,bs,1).to(each_feat.device)
            probabilities = gaussian.log_prob(grids).exp()
            new_probabilities = []
            for i in range(probabilities.shape[-1]):
                per_char_mask = probabilities[:,:,[i]].squeeze(-1).clone()
                enhance_indice =  per_char_mask >= (per_char_mask.max() *0.85)
                decrease_indice = per_char_mask < (per_char_mask.max() *0.85)
                per_char_mask[[enhance_indice]] = time_aware_strength[i]
                per_char_mask[[decrease_indice]] = -time_aware_strength[i]
                new_probabilities.append(per_char_mask)
            new_probabilities = torch.stack(new_probabilities)
            mask.append(new_probabilities)
        return mask, all_mean, all_cor, all_sum_mean

def compute_similarity(tensor1, tensor2):
    similarity_list = []
    for i in range(tensor1.size(0)):
        if torch.all(tensor1[i] == 1) or torch.all(tensor2[i] == 1):
            continue
        dot_product = torch.sum(tensor1[i] * tensor2[i])
        norm1 = torch.norm(tensor1[i])
        norm2 = torch.norm(tensor2[i])
        similarity = dot_product / (norm1 * norm2)
        similarity_list.append(similarity.item())
    return similarity_list.mean()


logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}

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


def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images = []
    for i in range(len(args.validation_prompts)):
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(accelerator.device.type)

        with autocast_ctx:
            image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]

        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images


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
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default='Pororo',
    )
    parser.add_argument(
        "--test_dataset_dir",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--character_dir",
        type=str,
        default=None,
        help=(
            "The file path to the whole customize character"
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
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
        "--max_char",
        type=int,
        default=2,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--use_high_res",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--dream_detail_preservation",
        type=float,
        default=1.0,
        help="Dream detail preservation factor p (should be greater than 0; default=1.0, as suggested in the paper)",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank


    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    cross_attn_init()
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            args.sample_dir = os.path.join(args.output_dir, 'sample')
            os.makedirs(args.sample_dir, exist_ok=True)
            args.test_dir = os.path.join(args.output_dir, 'test')
            os.makedirs(args.test_dir, exist_ok=True)
            args.pipeline_save = os.path.join(args.output_dir, 'pipeline')
            os.makedirs(args.pipeline_save, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler_eval = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
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
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, 
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, 
        )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()
    # unet.requires_grad_(False)
    k_encoder = Autok_encoder(in_dim = 768, out_dim = 2, Height = args.resolution, Width= int(args.resolution/9) * 16, scale=args.scale)
    k_encoder.train()

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, 
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    if type(model).__name__ == 'Autok_encoder':
                        torch.save(model,os.path.join(output_dir, "k_encoder.pt"))
                    else:
                        model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                if type(model).__name__ == 'Autok_encoder':
                    k_encoder = torch.load(f'{input_dir}/k_encoder.pt')
                else:
                # load diffusers style into model
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    params_to_opt = itertools.chain(unet.parameters(),  k_encoder.parameters())
    # params_to_opt = itertools.chain(k_encoder.parameters())
    # total = sum([param.nelement() for param in params_to_opt])
    # print('Number of parameter: % .4fM' % (total / 1e6))
    # assert 1==2
    optimizer = optimizer_cls(
        params_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.

    # Preprocessing the datasets.
    # Train dataset creation
    train_dataset = BenchMark_Each_ClassConstrains(
        data_root=args.dataset_dir,
        tokenizer=tokenizer,
        resolution=512,
        center_crop=True,
        random_flip=False,
        max_char = args.max_char,
        type=args.data_name,
        )

    # Train DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Eval dataset creation
    eval_dataset = BenchMark_Each_ClassConstrains(
        data_root=args.dataset_dir,
        tokenizer=tokenizer,
        resolution=512,
        center_crop=True,
        random_flip=False,
        max_char = args.max_char,
        type=args.data_name,
        )

    # Eval DataLoaders creation:
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    test_dataset = Benchmark_Eval_ClassConstrains(
        data_root=args.test_dataset_dir,
        character_root=args.character_dir,
        tokenizer=tokenizer,
        resolution=512,
        max_char = 2,
        type=args.data_name,
        )

    # Eval DataLoaders creation:
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    unet = set_layer_with_name_and_path(unet)
    unet = register_cross_attention_hook(unet)
    # Prepare everything with our `accelerator`.
    unet, k_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, k_encoder, optimizer, train_dataloader, lr_scheduler
    )
    def make_data_yielder(dataloader):
        while True:
            for batch in dataloader:
                yield batch
            accelerator.wait_for_everyone()
    val_data_yielder = make_data_yielder(eval_dataloader)
    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, k_encoder):
            # with accelerator.accumulate(k_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                # print('latents:',latents.shape)
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                # Get mask by mask maker
                mask, all_mean, all_cor, _ = k_encoder(batch['character_list'], batch['character_desc_index'], batch['mean'], batch['cor'], text_encoder, timesteps)
                # print(mask[0].shape)
                # debug
                
                # Get the text embedding for conditioning
                encoder_hidden_states = {
                    'text_feat': text_encoder(batch["input_ids"], return_dict=False)[0],
                    'character_indice': batch["character_desc_index"],
                    'mask': mask,
                }

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                loss_reg = torch.mean(torch.abs(torch.stack(all_mean))) * 0.01 + torch.mean(torch.abs(torch.stack(all_cor))) * 0.001

                _mask = [i.clone().detach().squeeze(1) for i in mask]
                for i in range(0, len(_mask)-1):
                    _mask[i] = set_tensor_values_to_zero(_mask[i], _mask[-1])
                loss_dif = calculate_cos_sim(_mask) * 0.001
                # loss_dif = 0
                if args.snr_gamma is None:
                    loss_mse = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss_mse = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss_mse = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss_mse = loss.mean()
                loss = loss_mse + loss_reg + loss_dif
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)


                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        # validation
                        val_batch = next(val_data_yielder)
                        pipeline = StableDiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            text_encoder=text_encoder,
                            vae=vae,
                            unet=unwrap_model(unet),
                            revision=args.revision,
                        )
                        pipeline.save_pretrained(args.pipeline_save)
                        torch.save(unwrap_model(k_encoder),os.path.join(args.pipeline_save, "k_encoder.pt"))
                        # Run a final round of inference.
                        
                        gt_images = []
                        logger.info("Running inference for collecting generated images...")
                        pipeline = pipeline.to(accelerator.device)
                        pipeline.torch_dtype = weight_dtype
                        pipeline.set_progress_bar_config(disable=True)

                        if args.enable_xformers_memory_efficient_attention:
                            pipeline.enable_xformers_memory_efficient_attention()

                        if args.seed is None:
                            generator = None
                        else:
                            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                        global attn_maps
                        attn_maps = {}
                        syn_images, _ = validation(val_batch, tokenizer, text_encoder, unwrap_model(unet), vae, unwrap_model(k_encoder), noise_scheduler_eval, text_encoder.device, 5, args.seed)
                        img_list = []
                        for syn in syn_images:
                            img_list.append(np.array(syn))
                        img_list = np.concatenate(img_list, axis=1)
                        
                        Image.fromarray(img_list).save(os.path.join(args.sample_dir, f"{str(global_step).zfill(5)}.jpg"))

                        with open(os.path.join(args.sample_dir, f"{str(global_step).zfill(5)}.txt") , 'w') as f:
                            for each_line in val_batch['text']:
                                f.writelines(each_line + '\n')
                        
                        test_dir = os.path.join(args.test_dir, f'{str(global_step).zfill(5)}')
                        os.makedirs(test_dir, exist_ok = True)
                        for test_step, test_batch in enumerate(test_dataloader):
                            attn_maps = {}
                            test_batch['input_ids'] = test_batch['input_ids'].to(text_encoder.device)

                            img_list = []
                            syn_images, cond_mask = validation(test_batch, tokenizer, text_encoder,unwrap_model(unet), vae, unwrap_model(k_encoder), noise_scheduler_eval, text_encoder.device, 5, args.seed)
                            for syn in syn_images:
                                img_list.append(np.array(syn))
                            img_list = np.concatenate(img_list, axis=1)
                            Image.fromarray(img_list).save(os.path.join(test_dir, f"{str(test_step).zfill(5)}.jpg"))
                            with open(os.path.join(test_dir, f"{str(test_step).zfill(5)}.txt") , 'w') as f:
                                for each_line in test_batch['text']:
                                    f.writelines(each_line + '\n')
                        logger.info(f"Saved state to {save_path}")
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            if global_step >= args.max_train_steps:
                break
        if accelerator.is_main_process:
            if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                log_validation(
                    vae,
                    text_encoder,
                    tokenizer,
                    unet,
                    args,
                    accelerator,
                    weight_dtype,
                    global_step,
                )
                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.pipeline_save)
        torch.save(unwrap_model(k_encoder),os.path.join(args.pipeline_save, "k_encoder.pt"))
        # Run a final round of inference.
        logger.info("Running inference for collecting generated images...")
        pipeline = pipeline.to(accelerator.device)
        pipeline.torch_dtype = weight_dtype
        pipeline.set_progress_bar_config(disable=True)

        if args.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()
        val_batch = next(val_data_yielder)
        img_list = []
        if args.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        attn_maps = {}
        syn_images,_ = validation(val_batch, tokenizer, text_encoder, unet, vae, unwrap_model(k_encoder), noise_scheduler_eval, batch["input_ids"].device, 5, args.seed)
        for syn in syn_images:
            img_list.append(np.array(syn))
        img_list = np.concatenate(img_list, axis=1)
        Image.fromarray(img_list).save(os.path.join(args.sample_dir, f"final.jpg"))

        with open(os.path.join(args.sample_dir, f"final.txt") , 'w') as f:
            for each_line in val_batch['text']:
                f.writelines(each_line + '\n')

        test_dir = os.path.join(args.test_dir, 'final')
        os.makedirs(test_dir, exist_ok = True)
        for test_step, test_batch in enumerate(test_dataloader):
            attn_maps = {}
            test_batch['input_ids'] = test_batch['input_ids'].to(text_encoder.device)

            img_list = []
            syn_images, cond_mask = validation(test_batch, tokenizer, text_encoder,unwrap_model(unet), vae, unwrap_model(k_encoder), noise_scheduler_eval, text_encoder.device, 5, args.seed)
            for syn in syn_images:
                img_list.append(np.array(syn))
            img_list = np.concatenate(img_list, axis=1)
            Image.fromarray(img_list).save(os.path.join(test_dir, f"{str(test_step).zfill(5)}.jpg"))
            with open(os.path.join(test_dir, f"{str(test_step).zfill(5)}.txt") , 'w') as f:
                for each_line in test_batch['text']:
                    f.writelines(each_line + '\n')

        if args.push_to_hub:
            save_model_card(args, repo_id, images, repo_folder=args.output_dir)
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
