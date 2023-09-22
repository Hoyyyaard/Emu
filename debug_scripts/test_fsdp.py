import argparse

import json
import time
import torch
from models.modeling_emu import Emu
from utils import process_img, process_video
import functools
image_placeholder = "[IMG]" + "<image>" * 32 + "[/IMG]"
image_system_msg = "You will be presented with an image: [IMG]ImageContent[/IMG]. You will be able to see the image after I provide it to you. Please answer my questions based on the given image."
video_system_msg = "You are a helpful assistant and you will be presented with a video consisting of multiple chronological images: [IMG]ImageContent[/IMG]. You will be able to see the video after I provide it to you. Please answer my questions based on the given video."

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
    _recursive_wrap
)

from torch.distributed.fsdp import (
CPUOffload,
MixedPrecision,
ShardingStrategy,
BackwardPrefetch,
)

import os



from models.model import _build_vision_tower, CLIPVisionCfg
import copy

WORLD_SIZE = 2



def wrap_model(model):
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1
    )
    wrapper_kwargs = dict(
        process_group=None,
        cpu_offload=CPUOffload(offload_params=True),
        device_id=torch.cuda.current_device(),
        auto_wrap_policy=my_auto_wrap_policy,
        sync_module_states=True,  # broadcast loaded ckpt from rank 0 -> all ranks
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=None,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
    )
        # for name, param in model.named_parameters():
        #     if '.' not in name:
        #         param.data = param.data.to(torch.cuda.current_device())
        # model.cls_token.to(torch.cuda.current_device())
                # print(name)
                # setattr(model, name, param.to(torch.cuda.current_device()))
    def apply_with_stopping_condition(module, apply_fn, apply_condition=None, stopping_condition=None, **other_args):
        # if stopping_condition(module):
        #     return
        if apply_condition(module):
            apply_fn(module, **other_args)
        for child in module.children():
            apply_with_stopping_condition(
                child,
                apply_fn,
                apply_condition=apply_condition,
                stopping_condition=stopping_condition,
                **other_args
            )
            
    def apply_fn(no_child_module):
        for name, param in no_child_module.named_parameters():
            print(name)
            if '.' not in name:
                print(f"Move parameter {name} from module to device")
                param.data = param.data.to(torch.cuda.current_device())
    
    # apply_with_stopping_condition(
    #     module=model,
    #     # apply_fn=lambda m: m.to(torch.cuda.current_device()),
    #     apply_fn=apply_fn,
    #     apply_condition=lambda m: len(list(m.children())) == 0,
    #     stopping_condition=lambda m: isinstance(m, FSDP),
    # )
    
    # print("Move nn.Param to device done")
    
    with enable_wrap(wrapper_cls=FSDP, **wrapper_kwargs):
        for name, module in model.named_children():
            # print(name)
            # if 'blocks' in name:
            #     model.blocks = torch.nn.ModuleList(
            #         wrap(m) 
            #         for m in module.children()
            #     )
                # for n, m in module.named_children():
                #     # print(n)
                #     setattr(module, n, wrap(m))
                    
            # else:
            if isinstance(module, torch.nn.ModuleList):
                tmp_module = torch.nn.ModuleList(
                FSDP(block, **wrapper_kwargs) for block in module
                )
                setattr(model, name, tmp_module)
                print("yes")
            else:
                setattr(model, name, FSDP(module, **wrapper_kwargs))
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1
    )
    # model = FSDP(model,
    #             sharding_strategy=ShardingStrategy.FULL_SHARD,
    #             auto_wrap_policy=my_auto_wrap_policy,
    #             device_id=torch.cuda.current_device(),
    #             use_orig_params=False,
    #             )
    print(model)
    
    model.cls_token.data = model.cls_token.data.to(torch.cuda.current_device())
    model.pos_embed.data = model.pos_embed.data.to(torch.cuda.current_device())
    # for param_name in model.get_non_module_params():
    #     getattr(model, param_name).to(torch.cuda.current_device())
    # print("--------------[INFO] Load non module parameters done------------------")
    # model = FSDP(model, **wrapper_kwargs)
    
    return model


def setup(rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=WORLD_SIZE)

def main(rank, model):
    
    # print(model)
    
    setup(rank)
    torch.cuda.set_device(rank)
    
    model = wrap_model(model)
    print("----------------------[INFO] Wrap model done--------------")
    image = process_img(img_path='examples/iron_man.jpg', device=torch.cuda.current_device())    
    image_fts = model.forward_features(image)
    print("----------------------[INFO] Forward feature done--------------")


if __name__ == '__main__':
    with open(f'models/Emu-14B.json', "r", encoding="utf8") as f:
        model_cfg = json.load(f)
    print(f"=====> model_cfg: {model_cfg}")
    
    vision_cfg = CLIPVisionCfg(**model_cfg['vision_cfg']) if isinstance(model_cfg['vision_cfg'], dict) else model_cfg['vision_cfg']
    visual = _build_vision_tower(
                embed_dim=model_cfg['embed_dim'],
                cast_dtype=torch.float16,
                vision_cfg=vision_cfg
            )
    visual = visual.eval()
    
    # addition_devicc_id = torch.cuda.device_count()-1
    # visual_single_device = copy.deepcopy(visual)
    # visual_single_device = visual_single_device.to(addition_devicc_id)
    # print('[INFO] Load {visual_single_device} Done')
    
    mp.spawn(main,
        args=(visual,),
        nprocs=WORLD_SIZE,
        join=True)

