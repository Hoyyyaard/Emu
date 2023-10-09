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

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        
        "--instruct",
        action='store_true',
        default=False,
        help="Load Emu-I",
    )
    parser.add_argument(
        
        "--avg_reg",
        action='store_true',
        default=False,
    
    )
    parser.add_argument(
        
        "--just_end",
        action='store_true',
        default=False,
    
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default='ckpts/',
        help="Emu ckpt path",
    )
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
        "--mlt_emu",
        action='store_true',
        default=False,
        help="Load Emu-I",
    )
    parser.add_argument(
        "--clip_norm",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--lr_base",
        type=float,
        default=1e-4,
    )
    
    parser.add_argument(
        "--lora",
        action='store_true',
        default=False,
    )
    
    parser.add_argument(
        "--gckpt",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--load_ckpt",
        action='store_true',
        default=False,
    )
    
    parser.add_argument(
        "--train",
        action='store_true',
        default=False,
    )
    
    parser.add_argument(
        "--bf16",
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

    emu_encoder = EmuGenerationPipeline.prepare_emu("Emu-14B", "results/finetune_exp/lrb{0.00001}-epo{100}-bs{8}-norm{wo}-ckpt{w}-loss{all}-lora{w}-lrd{w}-bf16{wo}-data{all}/ckpt/finetune_20_cls0.20_reg0.13.bin", args=args)
    

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
    
    if args.train:
        pipeline.unet.train()
    
    pipeline.wrap_fsdp(wrapper_kwargs)
    
    visual_decoding_example(pipeline, args)

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
    # init_start_event = torch.cuda.Event(enable_timing=True)
    # init_end_event = torch.cuda.Event(enable_timing=True)
    # init_start_event.record()
    # image_2 = Image.open("examples/arm.png").convert("RGB") 

    # image, safety = pipeline(
    #     [
    #         image_2,
    #         "Make the blue box on top of the yellow box in the image:",
    #     ],
    #     height=512,
    #     width=512,
    #     guidance_scale=10.,
    # )
    # init_end_event.record()

    # if rank == 0:
    #     print(f"Model inference time : {init_start_event.elapsed_time(init_end_event) / 1000}sec")

    # if safety is None or not safety:
    #     image.save("results/arm.jpg")
    # else:
    #     print("In-context Generated Image Has Safety Concern!!!")
    # if rank == 0:
    #     print(f'Finish torch.cuda.memory_reserved : {torch.cuda.memory_reserved(torch.cuda.current_device())/1024**3.:3f} GB')

def visual_decoding_example(pipeline, args):
    rank = torch.cuda.current_device()
    
    from src.pretrain_dataset import Visual_Decoding_Dataset
    dataset = Visual_Decoding_Dataset(_dataset_path=args.data_path)
    val_dataset = Visual_Decoding_Dataset(_dataset_path=args.val_data_path)
    
    if torch.cuda.current_device() == 0:
        logger.info(f"Dataset Len : {len(dataset)} || World Size : {torch.cuda.device_count()}")
    
    train_sampler = DistributedSampler(dataset, 
                                rank=rank, 
                                num_replicas=torch.cuda.device_count(), 
                                shuffle=True)
    train_loader = torch.utils.data.DataLoader(dataset,
                                            sampler=train_sampler,
                                            batch_size=args.batch_size ,
                                            num_workers=4)
    val_sampler = DistributedSampler(val_dataset, 
                                rank=rank, 
                                num_replicas=torch.cuda.device_count(), 
                                shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            sampler=val_sampler,
                                            batch_size=1 ,
                                            num_workers=1)
    
    
    # float16的有效动态范围： 5.960464477539063e-08 ~65504 故default的eps为 1e-8可能导致 计算中分母为0导致grad没有
    # nan但是optimizer step后出现nan
    optimizer = optim.AdamW(pipeline.parameters(), lr = args.lr_base, betas=(0.9,0.98), weight_decay=0.05, eps=1e-4)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        
        train_sampler.set_epoch(epoch)
        
        
        loss_freg = torch.nn.MSELoss()
        
        batch_loss = []
        import tqdm
        for bi, batch in enumerate(tqdm.tqdm(train_loader, desc=f'Epoch {epoch}')):

            
            global_step += 1
            
            batch_prompts = list(batch[0])
            gt_images = batch[1]
            batch_fps = []
            batch_gt_images = []
            for bii in range(len(args.batch_size)):
                image = Image.open(fps[bii]).convert("RGB")
                gt_images = (Image.open(gt_images[bii]).convert("RGB"))
            

                image, safety = pipeline(
                    [
                        batch_prompts[bii][0],
                        image,
                        batch_prompts[bii][1]
                        
                    ],
                    height=512,
                    width=512,
                    guidance_scale=10.,
                )
            
            
            
            batch_loss.append(loss_freg(image, gt_images))
        
        loss = sum(batch_loss)
        if (rank == 0):
            writer.add_scalar("train_loss_reg_rank0", loss.item(), global_step=global_step)
            
        optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(parameters=emu_model.visual.parameters(), max_norm=10, norm_type=2)
        # torch.nn.utils.clip_grad_norm_(parameters=emu_model.cformer.parameters(), max_norm=10, norm_type=2)
        # torch.nn.utils.clip_grad_norm_(parameters=emu_model.decoder.parameters(), max_norm=10, norm_type=2)
        if args.clip_norm:
            torch.nn.utils.clip_grad_norm_(parameters=emu_model.parameters(), max_norm=10, norm_type=2)
    
        
        # print("##################################################################")
        optimizer.step()

        torch.cuda.empty_cache()
        # if torch.cuda.current_device() == 0:
        #     print(f'Batch 1 takes torch.cuda.memory_reserved : {torch.cuda.memory_reserved(torch.cuda.current_device())/1024**3.:3f} GB')

if __name__ == "__main__":
    args = parse_args()

    # NOTE
    # Emu Decoder Pipeline only supports pretrain model
    # Using instruct tuning model as image encoder may cause unpredicted results
    assert args.instruct is False, "Image Generation currently do not support instruct tuning model"

    
     
    WORLD_SIZE = torch.cuda.device_count()
    logger.info(f"world size : {WORLD_SIZE}")
    
    emu_encoder = None
    if not args.mlt_emu:
        emu_encoder = EmuGenerationPipeline.prepare_emu("Emu-14B", "ckpts/multimodal_encoder/pytorch_model.bin", args=args)
    
    # WORLD_SIZE = 8
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args, emu_encoder),
        nprocs=WORLD_SIZE,
        join=True)