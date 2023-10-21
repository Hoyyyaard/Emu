from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
import os
import json
import numpy as np
import pickle
from PIL import Image
import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from typing import Callable, List, Optional, Union

import tqdm
import argparse

from PIL import Image
from models.pipeline import EmuGenerationPipeline
import argparse

import json
import time
import torch
from models.modeling_emu import Emu
from utils import process_img, process_video
import functools
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default='val_data/',

    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default='val_data/',

    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default='results/debug/',

    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--classify_scale",
        type=float,
        default=10.,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--lr_base",
        type=float,
        default=1e-7,
    )
    
    parser.add_argument(
        "--gckpt",
        action='store_true',
        default=False,
    )
    
    parser.add_argument(
        "--train",
        action='store_true',
        default=False,
    )
    
    args = parser.parse_args()
    return args

args = parse_args()

import logging

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
# 获取文件日志句柄并设置日志级别，第二层过滤
handler = logging.FileHandler(f"{args.log_dir}/log.log", encoding='UTF-8')
handler.setLevel(logging.INFO)
# 生成并设置文件日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
# console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# 为logger对象添加句柄
logger.addHandler(handler)
logger.addHandler(console)

from torch.utils.tensorboard import SummaryWriter  
writer = SummaryWriter(f'{args.log_dir}/log')


class InstrP2P_Finetune_Dataset(Dataset):
    
    def __init__(self, _dataset_path='val_data/'):
        self._dataset_path = _dataset_path
        self._episodes = []
        
        self._parse_dataset()
        
    def _parse_dataset(self):
        for task in os.listdir(self._dataset_path):
            p1 = os.path.join(self._dataset_path, task)
            for epi in os.listdir(p1):
                episodes = []
                p2 = os.path.join(p1, epi)

                with open(f'{p2}/info.json', 'r') as f:
                    task_info = json.load(f)
                
                for si,subtask in enumerate(list(task_info.values())[1:]):
                    image_cond_p = f'{p2}/subtask{si}_rgb.png' if si > 0 else f'{p2}/origin_rgb.png'
                    episodes.append({'text_cond':subtask,
                                     'image_cond_p':image_cond_p,
                                     'gt_image_p':f'{p2}/subtask{si+1}_rgb.png'})
                
                self._episodes.extend(episodes)


    def __getitem__(self, index):
        return self._episodes[index]
    
    def __len__(self):
        return len(self._episodes)


import PIL.Image
import PIL.ImageOps
from packaging import version
if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }


class InstrP2P_Finetune_Pipeline(StableDiffusionInstructPix2PixPipeline):
    
    def __init__(self,
                vae: AutoencoderKL,
                text_encoder: CLIPTextModel,
                tokenizer: CLIPTokenizer,
                unet: UNet2DConditionModel,
                scheduler,
                safety_checker,
                feature_extractor: CLIPImageProcessor,
                requires_safety_checker: bool = True,):
        
        super().__init__(vae, 
                        text_encoder,
                        tokenizer,
                        unet,
                        scheduler,
                        safety_checker,
                        feature_extractor,
                        requires_safety_checker)
    
    def preprocess(self, image):
        if isinstance(image, torch.Tensor):
            return image
        elif isinstance(image, PIL.Image.Image):
            image = [image]

        if isinstance(image[0], PIL.Image.Image):
            # w, h = image[0].size
            # w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
            w, h = 224, 224
            image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
            image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(0, 3, 1, 2)
            image = 2.0 * image - 1.0
            image = torch.from_numpy(image)
        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, dim=0)
        return image
    
    def finetune_forward(self,
                        prompt: Union[str, List[str]] = None,
                        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
                        gt_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
                        num_inference_steps: int = 100,
                        guidance_scale: float = 7.5,
                        image_guidance_scale: float = 1.5,
                        negative_prompt: Optional[Union[str, List[str]]] = None,
                        num_images_per_prompt: Optional[int] = 1,
                        eta: float = 0.0,
                        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                        latents: Optional[torch.FloatTensor] = None,
                        prompt_embeds: Optional[torch.FloatTensor] = None,
                        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                        output_type: Optional[str] = "pil",
                        return_dict: bool = True,
                        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                        callback_steps: int = 1,):
        
        batch_size = len(prompt)
        do_classifier_free_guidance = guidance_scale > 1.0 and image_guidance_scale >= 1.0
        # import random
        # do_classifier_free_guidance = do_classifier_free_guidance if random.random() <= 0.1 else False
        device = self._execution_device

        # 2. Encode input prompt
        # prompt_embeds = [prompt_embeds, negative_prompt_embeds, negative_prompt_embeds]
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        
        # 3. Preprocess image
        image = self.preprocess(image)
        gt_image = self.preprocess(gt_image)
        height, width = image.shape[-2:]
        
        # 4. set timesteps
        # self.scheduler.set_timesteps(num_inference_steps, device=device)
        # timesteps = self.scheduler.timesteps
        
        # 5. Prepare Image latents
        gt_image = gt_image.to(device=device, dtype=prompt_embeds.dtype)
        gt_image_latents = self.vae.encode(gt_image).latent_dist.mode()
        
        init_image_latents = self.prepare_image_latents(
            image,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            do_classifier_free_guidance,
            generator,
        )

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        
        # 7. Check that shapes of latents and image match the UNet channels
        num_channels_image = gt_image_latents.shape[1]
        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents+num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )
        
        # 8. Sample timestep within num_inference_steps and compute noise target
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        timesteps_index = torch.randint(0, len(timesteps), (1,))
        t = timesteps[timesteps_index].to(device)
        
        # Add noise of timesteps to latents
        target = latents
        noisy_latents = self.scheduler.add_noise(gt_image_latents, latents, t)
        
        latent_model_input = torch.cat([noisy_latents] * 3) if do_classifier_free_guidance else noisy_latents
        # t = torch.cat([t] * 3) if do_classifier_free_guidance else t
        # concat latents, image_latents in the channel dimension
        scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        scaled_latent_model_input = torch.cat([scaled_latent_model_input, init_image_latents], dim=1)
        # predict the noise residual
        noise_pred = self.unet(scaled_latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
        
        scheduler_is_in_sigma_space = hasattr(self.scheduler, "sigmas")
        if scheduler_is_in_sigma_space:
            if isinstance(t, torch.Tensor):
                t = t.to(self.scheduler.timesteps.device)
            step_index = (self.scheduler.timesteps == t).nonzero().item()
            sigma = self.scheduler.sigmas[step_index]
            noise_pred = latent_model_input - sigma * noise_pred
                    
        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_image)
                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )
        
        import torch.nn.functional as F
        loss = F.mse_loss(noise_pred, target, reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape))))
        loss = loss.mean()
        
        return loss

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '55567'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    train_dataset = InstrP2P_Finetune_Dataset(_dataset_path=args.data_path)
    train_sampler = DistributedSampler(train_dataset, 
                                rank=rank, 
                                num_replicas=torch.cuda.device_count(), 
                                shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                sampler=train_sampler,
                                batch_size=args.batch_size ,
                                num_workers=16,
                                drop_last=True)
    model_id = "ckpts/instrp2p"
    pipe = InstrP2P_Finetune_Pipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    pipe.to(torch.cuda.current_device())
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    optimizer = optim.AdamW(pipe.unet.parameters(), lr = args.lr_base, betas=(0.9,0.999), weight_decay=1e-2, eps=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    global_step = 0
    
    pipe.vae.eval()
    pipe.vae.requires_grad_(False)
    pipe.unet.train()
    pipe.unet.requires_grad_(True)
    # pipe.scheduler.config.num_train_timesteps = 50
    
    # total_train_param = 0
    # for n,p in pipe.named_parameters():
    #     if p.requires_grad == True:
    # #         # print(n)
    #         total_train_param += p.numel()
    # print("trainable params: ", total_train_param)
    
    # import accelerator
    # train_loader, _, pipe, optimizer = accelerator.prepare(
    #     train_loader, None, pipe, optimizer
    # )

    for epoch in range(args.epochs):
        
        batch_prompt = None
        batch_image_cond = None
        batch_gt_image = None
        
        train_sampler.set_epoch(epoch)
        if rank == 0:
            pbar = tqdm.tqdm(total=len(train_loader), desc=f'Epoch {epoch}')
        # for bi, batch in enumerate(tqdm.tqdm(train_loader, desc=f'Epoch {epoch}')):
        for bi ,batch in enumerate(train_loader):
            if rank == 0:
                pbar.update(1)
                

            global_step += args.batch_size * torch.cuda.device_count()

            batch_prompt = batch['text_cond']
            batch_image_cond = []
            batch_gt_image = []
            for bii in range(args.batch_size):
                batch_image_cond.append(PIL.Image.open(batch['image_cond_p'][bii]))
                batch_gt_image.append(PIL.Image.open(batch['gt_image_p'][bii]))
            
            if bi % 20 == 0:
                # batch_image_cond[0].save(args.log_dir + f"/vis/origin.png")
                # print(batch_prompt[0])
                vis_images = pipe(batch_prompt[:10], image=batch_image_cond[:10], num_inference_steps=50, guidance_scale=3., image_guidance_scale=3.).images
                if not os.path.exists(p:=(args.log_dir + f"/vis")):
                    os.makedirs(p)
                for iii, vi in enumerate(vis_images):
                    vi.save(args.log_dir + f"/vis/{epoch}_{global_step}_{rank}_{iii}.png")
            
            loss = pipe.finetune_forward(
                prompt = batch_prompt,
                image=batch_image_cond,
                gt_image=batch_gt_image,
                num_inference_steps=50, 
                guidance_scale=3., 
                image_guidance_scale=3.
            )

            if (rank == 0):
                logger.info('Train Epoch: {}\t Step: {}\tBatch: {}\t Loss: {:.6f} \t  Lr: {}'.format(epoch, global_step, bi, loss, optimizer.state_dict()['param_groups'][0]['lr']))
                writer.add_scalar("train_loss_rank0", loss.item(), global_step=global_step)

            loss.backward()
                
            nono_zero_grad = 0
            for n,p in pipe.unet.named_parameters():
                if not torch.sum(p.grad) == 0:
                    nono_zero_grad += 1
            assert nono_zero_grad > 0
            
            optimizer.step()
            # scheduler.step()
            torch.cuda.empty_cache()
            optimizer.zero_grad()
        

                    
                
if __name__ == '__main__':

    WORLD_SIZE = torch.cuda.device_count()
    logger.info(f"world size : {WORLD_SIZE}")
    
    mp.spawn(main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)