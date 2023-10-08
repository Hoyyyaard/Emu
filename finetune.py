import argparse

import json
import time
import torch
from models.modeling_emu import Emu
from models.modeling_emu_pretrain import Emu as Emu_pretrain
from utils import process_img, process_video
import functools
image_placeholder = "[IMG]" + "<image>" * 32 + "[/IMG]"
image_system_msg = "You will be presented with an image: [IMG]ImageContent[/IMG]. You will be able to see the image after I provide it to you. Please answer my questions based on the given image."
video_system_msg = "You are a helpful assistant and you will be presented with a video consisting of multiple chronological images: [IMG]ImageContent[/IMG]. You will be able to see the video after I provide it to you. Please answer my questions based on the given video."
import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from transformers import Trainer as T_Trainer
from transformers import TrainingArguments
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

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
        default='ckpts/multimodal_encoder/pytorch_model.bin',
        help="Emu ckpt path",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    
    parser.add_argument(
        "--lora_finetune",
        action='store_true',
        default=False,
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

class MyTrainer(T_Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        prompt = inputs[0][0]
        # print(batch)
        images = torch.cat([process_img(img_path=fp[0], device=torch.cuda.current_device()).to(torch.) for fp in inputs[1]], dim=0)
        
        input_tokens = emu_model.decoder.tokenizer(
                                        prompt, 
                                        padding="longest", 
                                        return_tensors="pt",
                                        add_special_tokens=True,
                                        ).to(torch.cuda.current_device())
        
        loss = emu_model(images,
                        input_tokens.input_ids[0].unsqueeze(0),
                        input_tokens.attention_mask[0].unsqueeze(0)).llm_loss
    
        print("Loss : ", loss)
        
        if torch.cuda.current_device() == 0:
            print(f'Batch 1 infernece takes torch.cuda.memory_reserved : {torch.cuda.memory_reserved(torch.cuda.current_device())/1024**3.:3f} GB')
            
        return (loss, None) if return_outputs else loss


def finetune_example(emu_model, args):

    rank = torch.cuda.current_device()
    
    from src.pretrain_dataset import Pretrain_Dataset
    dataset = Pretrain_Dataset()
    if torch.cuda.current_device() == 0:
        print(f"Dataset Len : {len(dataset)}")
    
    train_sampler = DistributedSampler(dataset, 
                                rank=rank, 
                                num_replicas=torch.cuda.device_count(), 
                                shuffle=False)
    train_loader = torch.utils.data.DataLoader(dataset,
                                            sampler=train_sampler,
                                            batch_size = 1 ,
                                            num_workers = 1)
    
    # 将模型的参数分成几个组 每个组不同的学习率
    # param_groups = [
    #     {'params': emu_model.visual.parameters(), 'lr': 4e-5},
    #     {'params': emu_model.decoder.parameters(), 'lr': 3e-5},
    #     {'params': emu_model.cformer.parameters(), 'lr': 1e-4}
    # ]
    
        
    optimizer = optim.AdamW(emu_model.parameters(), lr = 1e-6, betas=(0.9,0.98), weight_decay=0.05)
    
    

    training_args = TrainingArguments(
        per_device_train_batch_size=1, gradient_checkpointing=True, 
    )
    tf_trainer = MyTrainer(model=emu_model, 
                           train_dataset=dataset,
                           optimizers=optimizer,
                           args=training_args)
    for epoch in range(1, args.epochs + 1):
        tf_trainer.train()

    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # for epoch in range(1, args.epochs + 1):
        
    #     train_sampler.set_epoch(epoch)
        
    #     ddp_loss = torch.zeros(2).to(rank)
        
    #     for batch in tqdm.tqdm(train_loader):
            
    #         # if epoch == 1:
    #         #     print(batch)
    #         prompt = batch[0][0]
    #         # print(batch)
    #         images = torch.cat([process_img(img_path=fp[0], device=torch.cuda.current_device()).to(torch.float16) for fp in batch[1]], dim=0)
            
    #         input_tokens = emu_model.decoder.tokenizer(
    #                                         prompt, 
    #                                         padding="longest", 
    #                                         return_tensors="pt",
    #                                         add_special_tokens=True,
    #                                         ).to(torch.cuda.current_device())
            
    #         loss = emu_model(images,
    #                         input_tokens.input_ids[0].unsqueeze(0),
    #                         input_tokens.attention_mask[0].unsqueeze(0)).llm_loss
    #         if torch.cuda.current_device() == 0:
    #             print(f'Batch 1 infernece takes torch.cuda.memory_reserved : {torch.cuda.memory_reserved(torch.cuda.current_device())/1024**3.:3f} GB')
                
    #         loss.backward()
    #         optimizer.step()
            
    #         ddp_loss[0] += loss.item()
    #         ddp_loss[1] += 1
    #         print(f"{torch.cuda.current_device()} : ",ddp_loss)
            
    #         torch.cuda.empty_cache()
    #         # if torch.cuda.current_device() == 0:
    #         #     print(f'Batch 1 takes torch.cuda.memory_reserved : {torch.cuda.memory_reserved(torch.cuda.current_device())/1024**3.:3f} GB')
        
    #     dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)   
    #     if rank == 0:
    #         print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))

    #     # 在每个周期结束后，更新学习率
    #     scheduler.step()
        
        
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def fsdp_main(rank, world_size, model, args):
    
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    model.wrap_fsdp()
    if args.gckpt:
        model.decoder.lm.gradient_checkpointing_enable()

    # emu_model = FSDP(emu_model, 
    #                 auto_wrap_policy=size_based_auto_wrap_policy,
    #                 device_id=torch.cuda.current_device(),
    #                 cpu_offload=CPUOffload(offload_params=True),
    #                 sync_module_states=True)
    # print(model)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Rank {rank} 模型的总参数数量: {total_params}")
    # time.sleep(2000)
    # print(f"[LOG] : Rank {rank} Load ALL Model Done")
    # instruct_example(model)
    
    # torch.distributed.barrier() 
    
    # if args.instruct:
    #     instruct_example(model)
    # else:
    #     pretrain_example(model)
    
    finetune_example(emu_model=model, args=args)
    
    
def prepare_model(model_name, args):
    with open(f'models/{model_name}.json', "r", encoding="utf8") as f:
        model_cfg = json.load(f)
    print(f"=====> model_cfg: {model_cfg}")

    
    model = Emu(**model_cfg, args=args) if args.instruct or args.lora_finetune else Emu_pretrain(**model_cfg, args=args)

    if args.train:
        model.train()

    if args.instruct :
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


    print(f"=====> loading from ckpt_path {args.ckpt_path}")
    
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    msg = model.load_state_dict(ckpt, strict=False)
    # model.eval()
    print(f"=====> get model.load_state_dict msg: {msg}")
    
    if args.lora_finetune:
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
        model.decoder.lm.print_trainable_parameters()
    


    return model


if __name__ == '__main__':
    
    args = parse_args()

    # initialize and load model
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    emu_model = prepare_model('Emu-14B', args).to(torch.float16)

 
    

    total_params = sum(p.numel() for p in emu_model.parameters())
    print(f"total params: {total_params*2/(1024**3):.3f} GB")
    print(f"Decoder params : {(sum(p.numel() for p in emu_model.decoder.parameters()))*2/(1024**3):.3f} GB")
    print(f"Cformer params : {(sum(p.numel() for p in emu_model.cformer.parameters()))*2/(1024**3):.3f} GB")
    print(f"visual params : {(sum(p.numel() for p in emu_model.visual.parameters()))*2/(1024**3):.3f} GB")
    


    total_train_param = 0
    for _,p in emu_model.named_parameters():
        if p.requires_grad == True:
            # print(_)
            total_train_param += p.numel()
    print(f'emu_model trainable parameters : {total_train_param * 2/1024**3.:3f} GB')
    
    WORLD_SIZE = torch.cuda.device_count()

    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, emu_model, args),
        nprocs=WORLD_SIZE,
        join=True)