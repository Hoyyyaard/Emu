# -*- coding: utf-8 -*-

import json
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
from typing import List, Union, Tuple

import torch
import torch.nn as nn
from torchvision import transforms as TF

from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor

from .modeling_emu_pretrain import Emu as Emu_pretrain
from .modeling_emu import Emu


class EmuGenerationPipeline(nn.Module):

    def __init__(
        self,
        emu_encoder,
        multimodal_model: str,
        feature_extractor: str,
        safety_checker: str,
        scheduler: str,
        unet: str,
        vae: str,
        eva_size=224,
        eva_mean=(0.48145466, 0.4578275, 0.40821073),
        eva_std=(0.26862954, 0.26130258, 0.27577711),
        **kwargs,
    ):
        super().__init__()

        self.unet = UNet2DConditionModel.from_pretrained(
            unet,
        )
        
        
        self.vae = AutoencoderKL.from_pretrained(
            vae,
        )
        
        
        
        self.scheduler = PNDMScheduler.from_pretrained(
            scheduler,
        )

        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            safety_checker,
        )
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            feature_extractor,
        )

        # self.emu_encoder = self.prepare_emu("Emu-14B", multimodal_model, **kwargs)
        self.emu_encoder = emu_encoder


        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.eval()

        self.transform = TF.Compose([
            TF.Resize((eva_size, eva_size), interpolation=TF.InterpolationMode.BICUBIC),
            TF.ToTensor(),
            TF.Normalize(mean=eva_mean, std=eva_std),
        ])
        
        self.gt_transform = TF.Compose([
            TF.Resize((512, 512), interpolation=TF.InterpolationMode.BICUBIC),
            TF.ToTensor(),
            TF.Normalize(mean=eva_mean, std=eva_std),
        ])

        self.args = kwargs['args']

        if self.args.gckpt:
            self.unet.enable_gradient_checkpointing()
            self.vae.enable_gradient_checkpointing()
            self.emu_encoder.set_grad_checkpointing()

    @torch.no_grad()
    def batch_forward(
        self,
        inputs: List[Union[Image.Image, str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> Tuple[Image.Image, bool]:

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self.emu_encoder.ln_visual.weight.device
        dtype = self.emu_encoder.ln_visual.weight.dtype

        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. Encode input prompt
        batch_size = self.args.batch_size

        prompt_embeds = self._prepare_and_encode_inputs_batch(
            inputs,
            device,
            dtype,
            do_classifier_free_guidance,
        )

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        
        # 3. Prepare latent variables
        # Bx4xHxW
        shape = (
            batch_size,
            self.unet.config.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor
        )
        latents = torch.randn(shape, device=device, dtype=dtype)

        # 4. Denoising loop
        # for t in tqdm(timesteps):
        for t in timesteps:
            # expand the latents if we are doing classifier free guidance
            # 2B x 4 x H x W
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        has_nsfw_concept = None
        # image, has_nsfw_concept = self.run_safety_checker(
        #     image,
        #     device,
        #     dtype
        # )

        # 10. Convert to PIL
        # image = self.numpy_to_pil(image)
        return image, has_nsfw_concept if has_nsfw_concept is not None else has_nsfw_concept
    
    def batch_visual_decoding(
        self,
        inputs: List[Union[Image.Image, str]],
        batch_tgt_images,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> Tuple[Image.Image, bool]:

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self.emu_encoder.ln_visual.weight.device
        dtype = self.emu_encoder.ln_visual.weight.dtype

        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. Encode input prompt
        batch_size = batch_tgt_images.shape[0]

        prompt_embeds = self._prepare_and_encode_inputs_batch(
            inputs,
            device,
            dtype,
            do_classifier_free_guidance,
        )

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        # timesteps = self.scheduler.timesteps

        latents = self.vae.encode(batch_tgt_images.to(torch.bfloat16)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        
        # Sample a random timestep for each image
        timesteps = torch.randint(0, num_inference_steps, (batch_size,), device=latents.device)
        timesteps = timesteps.long()
        
        # Add noise of timesteps to latents
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # Prepare noise label
        target = noise
        
        # Predict the noise residual and compute loss
        latent_model_input = torch.cat([noisy_latents] * 2) if do_classifier_free_guidance else noisy_latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, timesteps)
        timesteps =  torch.cat([timesteps] * 2)
        noise_pred = self.unet(latent_model_input, timesteps, prompt_embeds).sample
        
        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        
        # Compute loss
        import torch.nn.functional as F
        loss = F.mse_loss(noise_pred, target, reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape))))
        loss = loss.mean()
        
        
        # loss = F.mse_loss(noise_pred, target)
        
        return loss


    
    @torch.no_grad()
    def forward(
        self,
        inputs: List[Union[Image.Image, str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> Tuple[Image.Image, bool]:

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self.emu_encoder.ln_visual.weight.device
        dtype = self.emu_encoder.ln_visual.weight.dtype

        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. Encode input prompt
        batch_size = 1

        prompt_embeds = self._prepare_and_encode_inputs(
            inputs,
            device,
            dtype,
            do_classifier_free_guidance,
        )

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 3. Prepare latent variables
        # Bx4xHxW
        shape = (
            batch_size,
            self.unet.config.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor
        )
        latents = torch.randn(shape, device=device, dtype=dtype)

        # 4. Denoising loop
        # for t in tqdm(timesteps):
        for t in timesteps:
            # expand the latents if we are doing classifier free guidance
            # 2B x 4 x H x W
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(
            image,
            device,
            dtype
        )

        # 10. Convert to PIL
        image = self.numpy_to_pil(image)
        return image[0], has_nsfw_concept[0] if has_nsfw_concept is not None else has_nsfw_concept
            

    @torch.no_grad()
    def _prepare_and_encode_inputs(
        self,
        inputs: List[Union[str, Image.Image]],
        device: torch.device = "cpu",
        dtype: str = torch.float32,
        do_classifier_free_guidance: bool = False,
        placeholder: str = "[<IMG_PLH>]"
    ) -> torch.Tensor:
        text_prompt = ""
        image_prompt = []
        for x in inputs:
            if isinstance(x, str):
                text_prompt += x
            else:
                text_prompt += placeholder
                image_prompt.append(self.transform(x))

        # Nx3x224x224
        if len(image_prompt) == 0:
            image_prompt = None
        else:
            image_prompt = torch.stack(image_prompt)
            image_prompt = image_prompt.type(dtype).to(device)

        if do_classifier_free_guidance:
            text_prompt = [text_prompt, ""]
        else:
            text_prompt = [text_prompt]

        prompt = self.emu_encoder.generate_image(
            text=text_prompt,
            image=image_prompt,
            placeholder=placeholder,
        )

        return prompt

    @torch.no_grad()
    def _prepare_and_encode_inputs_batch(
        self,
        inputs: List[Union[str, Image.Image]],
        device: torch.device = "cpu",
        dtype: str = torch.float32,
        do_classifier_free_guidance: bool = False,
        placeholder: str = "[<IMG_PLH>]"
    ) -> torch.Tensor:
    
        batch_text_promot = []
        batch_image_prompt = []
        for bi in range(len(inputs)):
            text_prompt = ""
            image_prompt = []
            for x in inputs[bi]:
                if isinstance(x, str):
                    text_prompt += x
                else:
                    text_prompt += placeholder
                    image_prompt.append(self.transform(x))

            # Nx3x224x224
            if len(image_prompt) == 0:
                image_prompt = None
            else:
                image_prompt = torch.stack(image_prompt)
                image_prompt = image_prompt.type(dtype).to(device)

            if do_classifier_free_guidance:
                text_prompt = [text_prompt, ""]
            else:
                text_prompt = [text_prompt]
            batch_text_promot.extend(text_prompt)
            batch_image_prompt.append(image_prompt)

        prompt = self.emu_encoder.generate_image_batch(
            text=batch_text_promot,
            image=batch_image_prompt,
            placeholder=placeholder,
        )

        return prompt

    def decode_latents(self, latents: torch.Tensor) -> np.ndarray:
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def numpy_to_pil(self, images: np.ndarray) -> List[Image.Image]:
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def run_safety_checker(
        self,
        image: List[Image.Image],
        device: str,
        dtype: str,
    ):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    @classmethod
    def prepare_emu(
        cls,
        model_name: str,
        model_path: str,
        args,
        **kwargs,
    ) -> nn.Module:
        with open(f'models/{model_name}.json', "r", encoding="utf8") as f:
            model_cfg = json.load(f)

        # model = Emu(**model_cfg, cast_dtype=torch.float, **kwargs)
        model = Emu(**model_cfg, args=args) if args.instruct or args.lora else Emu_pretrain(**model_cfg, args=args)
        
        ckpt = torch.load(model_path, map_location="cpu")
        if args.lora:
            print('Patching LoRA...')
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model.decoder.lm = get_peft_model(model.decoder.lm, lora_config)
            msg = model.load_state_dict(ckpt, strict=False)
            print(msg)
        elif "module" in ckpt:
            # model.load_state_dict(ckpt["module"], strict=True)
            model.load_state_dict(ckpt["module"], strict=False)
        else:
            # model.load_state_dict(ckpt, strict=True)
            model.load_state_dict(ckpt, strict=False)

        return model

    @classmethod
    def from_pretrained(cls, emu_encoder, path: str, **kwargs):
        multimodal_model = kwargs.pop("multimodal_model", None)
        feature_extractor = kwargs.pop("feature_extractor", None)
        safety_checker = kwargs.pop("safety_checker", None)
        scheduler = kwargs.pop("scheduler", None)
        unet = kwargs.pop("unet", None)
        vae = kwargs.pop("vae", None)

        check_if_none = lambda x, y: y if x is None else x

        multimodal_model = check_if_none(multimodal_model, f"{path}/multimodal_encoder/pytorch_model.bin")
        feature_extractor = check_if_none(feature_extractor, f"{path}/feature_extractor")
        safety_checker = check_if_none(safety_checker, f"{path}/safety_checker")
        scheduler = check_if_none(scheduler, f"{path}/scheduler")
        unet = check_if_none(unet, f"{path}/unet")
        vae = check_if_none(vae, f"{path}/vae")

        return cls(
            emu_encoder=emu_encoder,
            multimodal_model=multimodal_model,
            feature_extractor=feature_extractor,
            safety_checker=safety_checker,
            scheduler=scheduler,
            unet=unet,
            vae=vae,
            **kwargs,
        )

    def wrap_fsdp(self, wrapper_kwargs):
        self.emu_encoder.wrap_fsdp()
        self.vae = self.vae.to(torch.cuda.current_device())
        self.unet = self.unet.to(torch.cuda.current_device())
        self.safety_checker = self.safety_checker.to(torch.cuda.current_device())
        if torch.cuda.current_device() == 0:
            print(f'FSDP model takes torch.cuda.memory_reserved : {torch.cuda.memory_reserved(torch.cuda.current_device())/1024**3.:3f} GB')