# -*- coding: utf-8 -*-
        
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
        "--emu_ckpt",
        type=str,
        default='ckpts/multimodal_encoder/pytorch_model.bin',

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
    os.environ['MASTER_PORT'] = '55567'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def fsdp_main(rank, world_size, args, emu_encoder):
    
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    if args.mlt_emu:
        logger.info(f"Emu ckpt: {args.emu_ckpt}")
        emu_encoder = EmuGenerationPipeline.prepare_emu("Emu-14B", args.emu_ckpt, args=args)
    

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
    pipeline = pipeline.to(torch.bfloat16)
    
    # if args.train:
    #     pipeline.eval()
    #     pipeline.unet.train()
    
    pipeline.wrap_fsdp(wrapper_kwargs)
    
    visual_decoding_example(pipeline, args, world_size)

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

def visual_decoding_example(pipeline, args, world_size):
    rank = torch.cuda.current_device()
    
    print("rank",rank)
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
                                            num_workers=1,
                                            drop_last=True)
    
    val_sampler = DistributedSampler(val_dataset, 
                                rank=rank, 
                                num_replicas=torch.cuda.device_count(), 
                                shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            sampler=val_sampler,
                                            batch_size=1 ,
                                            num_workers=1,
                                            drop_last=True)
    
    
    # float16的有效动态范围： 5.960464477539063e-08 ~65504 故default的eps为 1e-8可能导致 计算中分母为0导致grad没有
    # nan但是optimizer step后出现nan
    optimizer = optim.AdamW(pipeline.unet.parameters(), lr = args.lr_base, betas=(0.9,0.999), weight_decay=1e-2, eps=1e-8)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    global_step = 0
    
    # pipeline.train()
    # pipeline.eval() 
    
    pipeline.emu_encoder.eval()
    pipeline.emu_encoder.requires_grad_(False)
    pipeline.vae.eval()
    pipeline.vae.requires_grad_(False)
    pipeline.safety_checker.eval()
    pipeline.safety_checker.requires_grad_(False)
    pipeline.unet.train()
    pipeline.unet.requires_grad_(True)
    total_train_param = 0
    for n,p in pipeline.named_parameters():
        if p.requires_grad == True:
    #         # print(n)
            total_train_param += p.numel()
    print("trainable params: ", total_train_param)
    
    for epoch in range(args.epochs):
        
        train_sampler.set_epoch(epoch)
        if rank == 0:
            pbar = tqdm.tqdm(total=len(train_loader), desc=f'Epoch {epoch}')
        # for bi, batch in enumerate(tqdm.tqdm(train_loader, desc=f'Epoch {epoch}')):
        for bi ,batch in enumerate(train_loader):
            if rank == 0:
                pbar.update(1)
            global_step += len(batch[1]) * torch.cuda.device_count()
            
            batch_squences = []
            batch_gt_images = []

            for bii in range(len(batch[1])):
                gt_image = Image.open(batch[1][bii])
                gt_image = pipeline.gt_transform(gt_image).unsqueeze(0).to(torch.bfloat16).requires_grad_(True).to(torch.cuda.current_device())
                squence = [
                    batch[0][0][bii],
                    Image.open(batch[0][1][bii]),
                    batch[0][2][bii],
                ]
                batch_squences.append(squence)
                batch_gt_images.append(gt_image)
            
            batch_gt_images = torch.cat(batch_gt_images,dim=0)
            
            loss = pipeline.batch_visual_decoding(
                batch_squences,
                batch_gt_images,
                height=512,
                width=512,
                num_inference_steps=50,
                guidance_scale=args.classify_scale,
            )
            
            # import sys
            # sys.setrecursionlimit(100000)
            # from torchviz import make_dot
            # g = make_dot(batch_raw_image)
            # g.view()
            # g.render(filename='graph', view=False,format='pdf') 
        
            
            if (rank == 0):
                logger.info('Train Epoch: {}\t Step: {}\tBatch: {}\t Loss: {:.6f} \t  Lr: {}'.format(epoch, global_step, bi, loss, optimizer.state_dict()['param_groups'][0]['lr']))
                writer.add_scalar("train_loss_decoding_rank0", loss.item(), global_step=global_step)
            
            
            # for n,p in pipeline.named_parameters():
            #     if p.requires_grad == True:
            #         print(n)
            
            # with torch.autograd.detect_anomaly():
            # loss.backward(retain_graph=True)
            loss.backward()

            
            nono_zero_grad = 0
            for n,p in pipeline.unet.named_parameters():
                if not torch.sum(p.grad) == 0:
                    nono_zero_grad += 1
            assert nono_zero_grad > 0
            
            # if args.clip_norm:
            #     torch.nn.utils.clip_grad_norm_(parameters=pipeline.parameters(), max_norm=10, norm_type=2)
        
            optimizer.step()
            
            
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            # pipeline.zero_grad()
        
            if (bi % 20 == 0 ):
                batch_image, _ = pipeline.batch_forward(batch_squences,
                                                    height=512,
                                                    width=512,
                                                    num_inference_steps=50,
                                                    guidance_scale=args.classify_scale,)
                vis_batch_gt_images = batch_gt_images.cpu().permute(0, 2, 3, 1).float().detach().numpy()
                vis = np.concatenate((vis_batch_gt_images, batch_image), axis=2)
                vis = pipeline.numpy_to_pil(vis)
                if not os.path.exists(p:=(args.log_dir + f"/vis")):
                    os.makedirs(p)
                for iii, vi in enumerate(vis):
                    vi.save(args.log_dir + f"/vis/{epoch}_{bi}_{rank}_{iii}.png")
                    

        scheduler.step()
        
        if epoch % 5 == 0 :
            print("Stop At Saving Parameters")
            # use a barrier to make sure training is done on all ranks
            dist.barrier()
            # state_dict for FSDP model is only available on Nightlies for now
            states = pipeline.state_dict()
            
            if rank == 0:
                print("Save Unet Parameters")
                unet_state = {}
                for param_tensor in states:
                    if 'unet' in param_tensor:
                        unet_state.update({param_tensor:states[param_tensor]})
                if not os.path.exists(p:=args.log_dir+"/unet_ckpt"):
                    os.mkdir(p)
                torch.save(unet_state, args.log_dir+f"/unet_ckpt/finetune_{epoch}_{global_step}_cls{loss:.2f}")

if __name__ == "__main__":
    args = parse_args()

    # NOTE
    # Emu Decoder Pipeline only supports pretrain model
    # Using instruct tuning model as image encoder may cause unpredicted results
    assert args.instruct is False, "Image Generation currently do not support instruct tuning model"

    assert args.bf16 is True
     
    WORLD_SIZE = torch.cuda.device_count()
    logger.info(f"world size : {WORLD_SIZE}")
    
    emu_encoder = None
    if not args.mlt_emu:
        logger.info(f"Emu ckpt: {args.emu_ckpt}")
        emu_encoder = EmuGenerationPipeline.prepare_emu("Emu-14B", args.emu_ckpt, args=args)
    
    # WORLD_SIZE = 8
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args, emu_encoder),
        nprocs=WORLD_SIZE,
        join=True)
