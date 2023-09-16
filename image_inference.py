# -*- coding: utf-8 -*-

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instruct",
        action='store_true',
        default=False,
        help="Load Emu-I",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default='ckpts',
        help="Emu ckpt path",
    )
    args = parser.parse_args()

    return args

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def fsdp_main(rank, world_size, args, emu_encoder):
    
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    wrapper_kwargs = dict(
        process_group=None,
        cpu_offload=CPUOffload(offload_params=False),
        device_id=torch.cuda.current_device(),
        auto_wrap_policy=size_based_auto_wrap_policy,

    )
    pipeline = EmuGenerationPipeline.from_pretrained(
        emu_encoder=emu_encoder,
        path=args.ckpt_path,
        args=args,
    )
    # pipeline = pipeline.bfloat16().cuda()
    pipeline = pipeline.to(torch.float16)
    pipeline.wrap_fsdp(wrapper_kwargs)

    # image blend case
    # image_1 = Image.open("examples/sunflower.png")
    # image_2 = Image.open("examples/oil_sunflower.jpg")
    
    # image_1 = Image.open("examples/cat.jpg")
    # image_2 = Image.open("examples/tiger.jpg")
    # image, safety = pipeline(
    #     [image_1, image_2],
    #     height=512,
    #     width=512,
    #     guidance_scale=7.5,
    # )

    # if safety is None or not safety:
    #     image.save("image_blend_result.jpg")
    # else:
    #     print("ImageBlend Generated Image Has Safety Concern!!!")

    # # text-to-image case
    # text = "An image of a dog wearing a pair of glasses."
    # image, safety = pipeline(
    #     [text],
    #     height=512,
    #     width=512,
    #     guidance_scale=7.5,
    # )

    # if safety is None or not safety:
    #     image.save("text2image_result.jpg")
    # else:
    #     print("T2I Generated Image Has Safety Concern!!!")

    # in-context generation
    # image_1 = Image.open("examples/dog.png")
    image_2 = Image.open("examples/arm.png").convert("RGB") 

    image, safety = pipeline(
        [
            image_2,
            "Image the image that the blue box is on top of the yellow box:",
        ],
        height=512,
        width=512,
        guidance_scale=10.,
    )

    if safety is None or not safety:
        image.save("results/arm.jpg")
    else:
        print("In-context Generated Image Has Safety Concern!!!")


if __name__ == "__main__":
    args = parse_args()

    # NOTE
    # Emu Decoder Pipeline only supports pretrain model
    # Using instruct tuning model as image encoder may cause unpredicted results
    assert args.instruct is False, "Image Generation currently do not support instruct tuning model"

    
     
    WORLD_SIZE = torch.cuda.device_count()
    
    # total_params = sum(p.numel() for p in pipeline.parameters())
    # print(f"total params: {total_params*2/(1024**3):.3f} GB")
     
    # print(f"unet params : {(sum(p.numel() for p in pipeline.unet.parameters()))*2/(1024**3):.3f} GB")
    # print(f"vae params : {(sum(p.numel() for p in pipeline.vae.parameters()))*2/(1024**3):.3f} GB")
    # print(f"safety_checker params : {(sum(p.numel() for p in pipeline.safety_checker.parameters()))*2/(1024**3):.3f} GB")
    # print(f"emu_encoder params : {(sum(p.numel() for p in pipeline.emu_encoder.parameters()))*2/(1024**3):.3f} GB")

    emu_encoder = EmuGenerationPipeline.prepare_emu("Emu-14B", "ckpts/multimodal_encoder/pytorch_model.bin", args=args)
    
    # WORLD_SIZE = 8
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args, emu_encoder),
        nprocs=WORLD_SIZE,
        join=True)