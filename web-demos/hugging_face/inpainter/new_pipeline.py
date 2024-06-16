# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from tqdm import tqdm
import numpy as np
import PIL.Image
import PIL
import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AsymmetricAutoencoderKL, AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.attention_processor import FusedAttnProcessor2_0
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import torch.nn.functional as F
from core.utils import to_tensors
from utils.cfa import AttnState, CrossFrameAttnProcessor
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def get_occlusion_from_path():
    import os
    image_dir = '../../Downloads/MPI-Sintel-training_extras/training/occlusions/bandage_2'

    # Get a sorted list of image file names
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') and f.startswith('frame_')])
    image_files = image_files[:19]

    # Initialize an empty list to store the processed images
    image_list = []

    # Process each image
    for image_file in image_files:
        # Read the image
        img_path = os.path.join(image_dir, image_file)
        img = Image.open(img_path)
        
        # Resize the image to 512x512
        img_resized = img.resize((512, 512))
        
        # Convert to grayscale
        img_gray = img_resized.convert('1')
        
        # Define a transform to convert PIL  
        # image to a Torch tensor 
        transform = transforms.Compose([ 
            transforms.ToTensor() 
        ]) 
        
        # transform = transforms.PILToTensor() 
        # Convert the PIL image to Torch tensor 
        img_tensor = transform(img_gray) 
        image_list.append(img_tensor)

    # Convert the numpy array to a torch tensor
    image_tensor = torch.cat(image_list)
    # image_numpy = image_tensor[0].cpu().numpy() * 255
    # image_numpy = image_numpy.astype(np.uint8)
    # pil_image = Image.fromarray(image_numpy)
    # # If the tensor is a binary image and you want to save it as 1-bit image
    # pil_image = pil_image.convert('1')

    # # Save or show the image
    # pil_image.save("output_image.png")
    # pil_image.show()
    return image_tensor


def resize_flow(flow, new_height, new_width):
    batch_size, num_maps, channels, old_height, old_width = flow.shape
    resized_flow = torch.zeros((batch_size, num_maps, channels, new_height, new_width), device=flow.device)

    # Resizing each flow map individually
    for i in range(num_maps):
        # Extract the individual flow map
        single_flow = flow[:, i]

        # Resizing
        resized_single_flow = F.interpolate(single_flow, size=(new_height, new_width), mode='bilinear', align_corners=True)

        # Adjusting the flow values
        resized_single_flow[:, 0, :, :] *= (new_width / old_width)
        resized_single_flow[:, 1, :, :] *= (new_height / old_height)

        resized_flow[:, i] = resized_single_flow

    return resized_flow


def binary_mask(mask, th=0.1):
    mask[mask>th] = 1
    mask[mask<=th] = 0
    # return mask.float()
    return mask.to(mask)

def flow_warp(x,
              flow,
              #interpolation='bilinear',
              interpolation='nearest',
              padding_mode='zeros',
              align_corners=True,
              time = 1):
    """Warp an image or a feature map with optical flow.
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.
    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    device = flow.device
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, device=device), torch.arange(0, w, device=device))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
    grid.requires_grad = False
    
    # flow change by time
    flow = flow * time

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow_x = grid_flow_x
    grid_flow_y = grid_flow_y
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(x,
                           grid_flow,
                           mode=interpolation,
                           padding_mode=padding_mode,
                           align_corners=align_corners)
    return output


#coords_grid, bilinear_sample, flow_warp1, forward_backward_consistency_check come from unimatch
#https://github.com/ernestchu/unimatch/blob/master/unimatch/geometry.py

def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid

def bilinear_sample(img, sample_coords, mode='nearest', padding_mode='zeros', return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]
    grid = grid.to(torch.float16)

    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)  # [B, H, W]

        return img, mask

    return img


def flow_warp1(feature, flow, mask=False, padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature, grid, padding_mode=padding_mode,
                           return_mask=mask)


def forward_backward_consistency_check(fwd_flow, bwd_flow,
                                       alpha=0.01,
                                       beta=0.5
                                       ):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = flow_warp1(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = flow_warp1(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ, bwd_occ


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class VideoDiffusionInpaintPipeline(
    DiffusionPipeline
):
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"

    def __init__(
        self,
        vae: Union[AutoencoderKL, AsymmetricAutoencoderKL],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        feature_extractor: CLIPImageProcessor,
    ):
        
        

        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
    
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def encode_prompt(
        self,
        prompt,
        device,
        do_classifier_free_guidance,
        negative_prompt=None,
        clip_skip: Optional[int] = None,
    ):
        batch_size = 1
        

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        if clip_skip is None:
            prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
            prompt_embeds = prompt_embeds[0]
        else:
            prompt_embeds = self.text_encoder(
                text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
            )
            # Access the `hidden_states` first, that contains a tuple of
            # all the hidden states from the encoder layers. Then index into
            # the tuple to access the hidden states from the desired layer.
            prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
            # We also need to apply the final LayerNorm here to not mess with the
            # representations. The `last_hidden_states` that we typically use for
            # obtaining the final prompt representations passes through the LayerNorm
            # layer.
            prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)

        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * 1, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds


    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values
        
        image = image.to(device=device, dtype=dtype)

        image_embeds = self.image_encoder(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)

        return image_embeds, uncond_image_embeds
    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    def prepare_latents(
        self,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,
    ):
        shape = (1, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)


        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        if return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)

            if image.shape[1] == 4:
                image_latents = image
            else:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            image_latents = image_latents.repeat(1 // image_latents.shape[0], 1, 1, 1)

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs
    
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents
    
    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = self._encode_vae_image(masked_image, generator=generator)


        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt
    

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        images: List[PIL.Image.Image] = None,
        mask_images: List[PIL.Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 1.0,
        num_inference_steps: int = 50,
        flows = None,
        do_multi_diffusion: bool = False,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: int = None,
        use_cf = False,
        updated_frames_tensor: Optional[torch.HalfTensor] = None,
        updated_masks_tensor: Optional[torch.HalfTensor] = None,
        encoded_pixels: Optional[torch.IntTensor] = None,
        backward_coding: bool = True,
        output_path = '',
        **kwargs,
    ):
        
        print("start inpaint...")
        print("prompt: ", prompt)
        print("negative prompt: ", negative_prompt)
        self.attn_state = AttnState()
        if use_cf:
            attn_processor_dict = {}
            for k in self.unet.attn_processors.keys():
                if k.startswith("up") or k.startswith("down") or k.startswith("mid"):
                    attn_processor_dict[k] = CrossFrameAttnProcessor(self.attn_state)
            
            self.attn_state.reset()
            self.attn_state.set_timestep(num_inference_steps)
            self.unet.set_attn_processor(attn_processor_dict)

        # 0. Initialize
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        print(f"Unet Height: {height}, Width: {width}")
        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        batch_size = 1
        vae_batch_size = 1
        device = self._execution_device

        # 1. Encode input prompt
        print("Encode input prompt . . . ")

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            self.do_classifier_free_guidance,
            negative_prompt,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        # 2. set timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )

        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(1)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 3. Preprocess mask and image

        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels

        init_images = []
        for image in images:
            init_image = self.image_processor.preprocess(
                image, height=height, width=width
            )
            init_image = init_image.to(dtype=torch.float32)
            init_images.append(init_image)

        # 4. Prepare latent variables
        
        latents = []
        for init_image in init_images:
            latent_output = self.prepare_latents(
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents=None,
                image=init_image,
                timestep=latent_timestep,
                is_strength_max=is_strength_max,
                return_noise=True,
                return_image_latents=False,
            )
            latent, noise = latent_output
            latents.append(latent)

        # 5. Prepare mask latent variables
        print("Prepare mask latent variables . . .")
        masks = []
        masked_image_latents = []
        for init_image, mask_image in tqdm(zip(init_images, mask_images)):
            mask_condition = self.mask_processor.preprocess(
                mask_image, height=height, width=width
            )
            masked_image = init_image * (mask_condition < 0.5)
            mask, masked_image_latent = self.prepare_mask_latents(
                mask_condition,
                masked_image,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                self.do_classifier_free_guidance
                )
            masks.append(mask)
            masked_image_latents.append(masked_image_latent)
        # prepare downscaling mask and flow
        resize_masks = []
        for mask_image in mask_images:
            mask_image = mask_image.resize((64, 64), PIL.Image.NEAREST)
            resize_masks.append(mask_image)
        resize_masks = to_tensors()(resize_masks).unsqueeze(0).to(device).half()

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        self._num_timesteps = len(timesteps)
        # print("flows[0] shape here ", flows[0].shape) #torch.Size([1, t, 2, 512, 512])
        # flows_f = resize_flow(flows[0], 64, 64).half().to(device)
        # print("flow forward shape ", flows_f.shape)
        # flows_b = resize_flow(flows[1], 64, 64).half().to(device)
        # flows = (flows_f, flows_b)
        # 6. Denoising loop
        print("Denoising loop . . .")
        self._num_timesteps = len(timesteps)

        pred_occs, _ = forward_backward_consistency_check(flows[0][0], flows[1][0])
        # print("before dtype ", pred_occs.dtype)
        # # pred_occs = get_occlusion_from_path()
        # print("after dtype ", pred_occs.dtype)
        # print("pred occs shape ", pred_occs.shape)
        
        # import os
        # if not os.path.isdir(f'{output_path}/occlusion'):
        #     os.mkdir(f'{output_path}/occlusion')
        # for i in range(pred_occs.shape[0]):
        #     first_channel = pred_occs[i].cpu().numpy() *255 # Convert to numpy array
        #     first_channel = first_channel.astype(np.uint8)
        #     pil_image = Image.fromarray(first_channel)
        #     pil_image = pil_image.convert('1')
        #     pil_image.save(f'{output_path}/occlusion/{i}.png')

        # image_numpy = image_tensor[0].cpu().numpy() * 255
        # image_numpy = image_numpy.astype(np.uint8)
        # pil_image = Image.fromarray(image_numpy)
        # # If the tensor is a binary image and you want to save it as 1-bit image
        # pil_image = pil_image.convert('1')

        

        occlusions = (pred_occs > 0.5) #forward flow occlusion mask
        print("occlusions device ", occlusions.device)

        permuted_flow = flows[0].permute(0, 1, 3, 4, 2).cpu()
        occlusions = occlusions.cpu()
        del flows
        print("permte_flow.shape ", permuted_flow.shape)
        print(occlusions.shape)
        mixer = FlowCoding(permuted_flow, occlusions[None], backward_coding, encoded_pixels)
        mixer.to(device=device)


        """
        latent0 = flow_warp(latents[2], 
            flows[0][:,1, :, :, :].permute(0, 2, 3, 1), interpolation='bilinear', time=2)
        b_mask = binary_mask(resize_masks[:,0, :, :, :])
        latent0 = (latents[0] * (1 - b_mask) + latent0 * b_mask)

        latent1 = flow_warp(latents[2],
            flows[0][:,1, :, :, :].permute(0, 2, 3, 1), interpolation='bilinear', time=1)
        b_mask = binary_mask(resize_masks[:,1, :, :, :])
        latent1 = (latents[1] * (1 - b_mask) + latent1 * b_mask)

        latent2 = latents[2]

        latent3 = flow_warp(latents[2],
            flows[1][:,2, :, :, :].permute(0, 2, 3, 1), interpolation='bilinear', time=1)
        b_mask = binary_mask(resize_masks[:,3, :, :, :])
        latent3 = (latents[3] * (1 - b_mask) + latent3 * b_mask)

        latent4 = flow_warp(latents[2],
            flows[1][:,2, :, :, :].permute(0, 2, 3, 1), interpolation='bilinear', time=2)
        b_mask = binary_mask(resize_masks[:,4, :, :, :])
        latent4 = (latents[4] * (1 - b_mask) + latent4 * b_mask)
        latents = [latent0, latent1, latent2, latent3, latent4]"""

        updated_frames_tensor = (updated_frames_tensor) * 2 - 1   
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            
            for i, t in enumerate(timesteps):
                pred_clean_latents_list = latents.copy()
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                if i == 0:
                    print("unet info")
                    for key, value in self.unet.attn_processors.items():
                        print(key, value)
                
                #print(t)
                if self.interrupt:
                    continue
                #for index in range(2):
                npreds = latents.copy()
                latents_list = torch.cat(latents)
                for index in range(len(latents)):
                    # if index == 0:
                    #     self.attn_state.reset_wo_timestep()
                    # else:
                    #     self.attn_state.to_load()

                    latent, mask, masked_image_latent = latents[index], masks[index], masked_image_latents[index]
                    
                    latent_model_input = torch.cat([latent] * 2) if self.do_classifier_free_guidance else latent
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latent], dim=1)
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        return_dict=False,
                    )[0]

                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    npreds[index] = noise_pred

                    pred_clean_latents_list[index] = (latents[index] - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

                    if t == 1:
                        latents[index] = self.scheduler.step(noise_pred, t, latent, **extra_step_kwargs, return_dict=False)[0]
                    # latents[index] = self.scheduler.step(noise_pred, t, latent, **extra_step_kwargs, return_dict=False)[0]
                    
                # for index in range(len(latents)):
                #     latents[index] = self.scheduler.step(npreds[index:index+1], t, latents_list[index:index + 1], **extra_step_kwargs, return_dict=False)[0]
                    
                # mix the shared pixels, no mix in the last denoising step
                if t > 1:
                    pred_clean_latents_list = torch.cat(pred_clean_latents_list, dim=0).to(device, dtype=torch.float16)
                    # print("now pred clean latent list shape ", pred_clean_latents_list.shape) #TCHW
                    harmonized_pred_clean_latents_list = pred_clean_latents_list.clone()
                    latent_shape = harmonized_pred_clean_latents_list.shape
                    pred_clean_latents_observed = torch.zeros(
                        *latent_shape[:-3], 3, latent_shape[-2] * 8, latent_shape[-1] * 8,
                        device=harmonized_pred_clean_latents_list.device,
                        dtype=harmonized_pred_clean_latents_list.dtype)
                    
                    #change from latnet space to image space
                    #vae_batch_size = 1 here
                    harmonized_pred_clean_latents_list /= self.vae.config.scaling_factor
                    for frame_id in range(0, len(harmonized_pred_clean_latents_list), vae_batch_size):
                        pred_clean_latents_observed[frame_id:frame_id+vae_batch_size] = self.vae.decode(
                            harmonized_pred_clean_latents_list[frame_id:frame_id+vae_batch_size], return_dict=False)[0]

                    pred_clean_latents_observed = pred_clean_latents_observed * updated_masks_tensor + updated_frames_tensor * (1 - updated_masks_tensor)
                    
                    # mask = (updated_masks_tensor == 1)
                    # Use torch.nonzero to find the indices of elements that are 1
                    # indices = torch.nonzero(mask)
                    # print("ind ")
                    # print(indices)
                        
                    # for k in range(len(latents)):
                    #         im = self.image_processor.postprocess(pred_clean_latents_observed[k:k + 1].to(torch.float), output_type='pil')
                    #         im[0].save(f'decode/before/zzzbefore{i}_{k}.jpg')
                    
                    # harmonize the video
                    pred_clean_latents_observed = mixer(pred_clean_latents_observed)

                    # for k in range(len(latents)):
                    #         im = self.image_processor.postprocess(pred_clean_latents_observed[k:k + 1].to(torch.float), output_type='pil')
                    #         im[0].save(f'decode/after/zzzafter{i}_{k}.jpg')

                    #cast `pred_clean_latents_observed' to the latent space
                    for frame_id in range(0, len(harmonized_pred_clean_latents_list), vae_batch_size):
                        harmonized_pred_clean_latents_list[frame_id:frame_id+vae_batch_size] = self.vae.encode(
                            pred_clean_latents_observed[frame_id:frame_id+vae_batch_size]).latent_dist.mode()
                    harmonized_pred_clean_latents_list *= self.vae.config.scaling_factor

                ##################################################
                # original x_0 preds: pred_clean_latents_list
                # harmonized x_0 preds: harmonized_pred_clean_latents_list
                # Now we have derived the harmonized x_0 preds, we re-derive the noise again (x_0 -> noise)
                
                harmonization_scale = 1.
                if (t > 1 and harmonization_scale > 0):
                    # latents_list = torch.cat(latents, dim=0)
                    # pred_clean_latents_list = torch.cat(pred_clean_latents_list, dim=0).to(device, dtype=torch.float16)
                    noise_pred_ = pred_clean_latents_list.clone()
                    noise_pred_ = (latents_list - alpha_prod_t ** (0.5) * pred_clean_latents_list) / beta_prod_t ** (0.5)
                    noise_pred_harmonized = (latents_list - alpha_prod_t ** (0.5) *
                                                harmonized_pred_clean_latents_list) / beta_prod_t ** (0.5)
                    noise_pred_ = noise_pred_ + harmonization_scale * (noise_pred_harmonized - noise_pred_)                 
                    # compute the previous noisy sample x_t -> x_t-1
                    for index in range(len(latents)):
                        latents[index] = self.scheduler.step(noise_pred_[index:index+1], t, latents_list[index:index+1], **extra_step_kwargs, return_dict=False)[0]
                progress_bar.update()
        """
        latent0 = flow_warp(latents[2], 
            flows[0][:,1, :, :, :].permute(0, 2, 3, 1), interpolation='bilinear', time=2)
        b_mask = binary_mask(resize_masks[:,0, :, :, :])
        latent0 = (latents[0] * (1 - b_mask) + latent0 * b_mask)

        latent1 = flow_warp(latents[2],
            flows[0][:,1, :, :, :].permute(0, 2, 3, 1), interpolation='bilinear', time=1)
        b_mask = binary_mask(resize_masks[:,1, :, :, :])
        latent1 = (latents[1] * (1 - b_mask) + latent1 * b_mask)

        latent2 = latents[2]

        latent3 = flow_warp(latents[2],
            flows[1][:,2, :, :, :].permute(0, 2, 3, 1), interpolation='bilinear', time=1)
        b_mask = binary_mask(resize_masks[:,3, :, :, :])
        latent3 = (latents[3] * (1 - b_mask) + latent3 * b_mask)

        latent4 = flow_warp(latents[2],
            flows[1][:,2, :, :, :].permute(0, 2, 3, 1), interpolation='bilinear', time=2)
        b_mask = binary_mask(resize_masks[:,4, :, :, :])
        latent4 = (latents[4] * (1 - b_mask) + latent4 * b_mask)
        latents = [latent0, latent1, latent2, latent3, latent4]"""

        images = []
        # for latent in latents:
        #     image = self.vae.decode(
        #         latent / self.vae.config.scaling_factor, return_dict=False, generator=generator
        #     )[0]
        #     image = image * updated_masks_tensor + updated_frames_tensor * (1 - updated_masks_tensor)
        #     image = self.image_processor.postprocess(image, output_type=output_type)
        #     images.append(image)

        for i in range(len(latents)):
            image = self.vae.decode(
                latents[i] / self.vae.config.scaling_factor, return_dict=False, generator=generator
            )[0]
            image = image * updated_masks_tensor[i:i+1] + updated_frames_tensor[i:i+1] * (1 - updated_masks_tensor[i:i+1])
            image = self.image_processor.postprocess(image, output_type=output_type)
            images.append(image)

        # Offload all models
        self.maybe_free_model_hooks()
        return images

class FlowCoding(torch.nn.Module):
    def __init__(self, flows, occlusions, backward_coding, encoded_pixels):
        '''
        backward flow coding:
            required: optical flow map, occlusion mask, target shape
            return: number of unique pixels, encoded pixels (index to the unique pixels)
        '''
        super().__init__()
        flows = flows.to(torch.float32)
        if encoded_pixels is None:
            # unbatched (batched inference has not implemented)
            flows = flows[0]
            occlusions = occlusions[0]
            
            # flip along temporal dimension for backward coding
            if backward_coding:
                flows = flows.flip(0)
                occlusions = occlusions.flip(0)
            
            # coordinate matrix
            shape = occlusions.shape[1:]
            meshgrid = torch.stack(torch.where(torch.ones(shape))).T.view(*shape, -1)
            
            # start encoding
            encoded_pixels = []
            n_pixels = shape[0] * shape[1]
            
            enc = torch.arange(n_pixels).view(shape)
            encoded_pixels.append(enc)
            
            for i in range(len(flows)):
                flow = flows[i]
                occlusion = occlusions[i]
                # prepare ingredients
                prev_enc = enc
                enc = torch.zeros_like(enc)

                unfulfilled_mask = torch.ones_like(occlusion)

                dest_float = flow + meshgrid
                dest = dest_float.round().to(int)

                # discard out-of-range
                valid_mask = ((dest >= 0) & (dest < torch.tensor(dest.shape[:2]))).all(-1)
                # maskout where long range flow already handled
                valid_mask = torch.logical_and(valid_mask, unfulfilled_mask)
                v_dest = dest[valid_mask]
                v_grid = meshgrid[valid_mask]

                # get the common pixels from the previous frame
                enc[v_grid[:, 0], v_grid[:, 1]] = prev_enc[v_dest[:, 0], v_dest[:, 1]]
                # set the warped pixels to fulfilled
                unfulfilled_mask = torch.logical_and(unfulfilled_mask, ~valid_mask)
                # unset the occluded pixels
                unfulfilled_mask = torch.logical_or(unfulfilled_mask, occlusion)

                # novel pixels = occlusions + invalid meshgrid (long and short)
                # add novel pixels to global pixels
                novel_px = meshgrid[unfulfilled_mask]

                offset = n_pixels
                enc[novel_px[:, 0], novel_px[:, 1]] = torch.arange(len(novel_px)) + offset
                
                n_pixels += len(novel_px)
                encoded_pixels.append(enc)


            encoded_pixels = torch.stack(encoded_pixels)
            if backward_coding:
                # flip back to original temporal order
                encoded_pixels = encoded_pixels.flip(0)

            # add a dummy batch dimension for future batched implementation
            self.encoded_pixels = encoded_pixels[None]

        else:
            self.encoded_pixels = encoded_pixels
        
        self.shape = self.encoded_pixels.shape[-2:]
        # n_pixels is the sum of the number of unique pixels of the entire batch
        self.n_pixels = int(self.encoded_pixels.max() + 1)

        # init computation buffers
        self.register_buffer('values', torch.zeros(self.n_pixels, 3))
        self.register_buffer('counts', torch.zeros(self.n_pixels, 1, dtype=torch.int64))

        # percompute `counts' since it does not depend on the input
        self.counts.index_put_((self.encoded_pixels,), torch.tensor(1), accumulate=True)
        print("unique pixels ", self.n_pixels)


    def forward(self, samples):
        original_shape = samples.shape[-2:]
        samples = torch.nn.functional.interpolate(
            samples, self.shape, mode='bilinear')
        samples = samples.permute(0, 2, 3, 1) # TCHW -> THWC

        self.values.zero_()
        self.values = self.values.to(torch.float16)
        self.values.index_put_((self.encoded_pixels,), samples, accumulate=True)
        pixels = torch.where(self.counts > 0, self.values / self.counts, self.values)
        mixed = pixels[self.encoded_pixels][0].permute(0, 3, 1, 2) # BTHWC -> TCHW

        mixed = torch.nn.functional.interpolate(
            mixed, original_shape, mode='bilinear')

        return mixed
   