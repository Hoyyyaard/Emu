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
        default='ckpts/Emu-instruct.pt',
        help="Emu ckpt path",
    )
    args = parser.parse_args()

    return args


def prepare_model(model_name, args):
    with open(f'models/{model_name}.json', "r", encoding="utf8") as f:
        model_cfg = json.load(f)
    print(f"=====> model_cfg: {model_cfg}")

    model = Emu(**model_cfg, args=args)

    if args.instruct:
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
    model.eval()
    print(f"=====> get model.load_state_dict msg: {msg}")

    return model


def Emu_inference(emu_model, image_list, text_sequence, system='', instruct=True, max_new_tokens=128, beam_size=5, length_penalty=0.0):
    if instruct:
        prompt = f"{system} [USER]: {text_sequence} [ASSISTANT]:".strip()
    else:
        prompt = text_sequence

    print(f"===> prompt: {prompt}")

    samples = {"image": torch.cat(image_list, dim=0), "prompt": prompt}

    output_text = emu_model.generate(
        samples,
        max_new_tokens=max_new_tokens,
        num_beams=beam_size,
        length_penalty=length_penalty,
        repetition_penalty=1.0,
    )[0].strip()

    print(f"===> output: {output_text}\n")


def Emu_instruct_caption(img, emu_model):
    system = image_system_msg

    prompt = f"{system} [USER]: {image_placeholder}Please provide an accurate and concise description of the given image. [ASSISTANT]: The image depicts a photo of".strip()

    print(f"===> caption prompt: {prompt}")

    samples = {"image": img, "prompt": prompt}

    print('[LOG] Start General Response')
    output_text = emu_model.generate(
        samples,
        max_new_tokens=512,
        num_beams=5,
        length_penalty=0.0,
        repetition_penalty=1.0,
    )[0].strip()

    print(f"===> caption output: {output_text}\n")


def pretrain_example():
    # prepare in-context learning example
    image_text_sequence = [
        process_img(img_path='examples/dog.png', device=args.device),
        'There are two dogs.',
        process_img(img_path='examples/panda.png', device=args.device),
        'There are three pandas.',
        process_img(img_path='examples/sunflower.png', device=args.device),
    ]
    interleaved_sequence_1 = ''
    image_list_1 = []
    for item in image_text_sequence:
        if isinstance(item, str):  # text
            interleaved_sequence_1 += item
        else:  # image
            image_list_1.append(item)
            interleaved_sequence_1 += image_placeholder

    # Pretrained Model Inference
    # -- in-context learning
    Emu_inference(image_list_1, interleaved_sequence_1, instruct=False)


def instruct_example(emu_model):
    # prepare image captioning and vqa examples
    image = process_img(img_path='assets/iron_man.jpg', device=torch.cuda.current_device())
    question = 'what is the man doing?'

    # prepare interleaved image-text input example
    image_text_sequence = [
        process_img(img_path='examples/book1.jpeg', device=torch.cuda.current_device()),
        'This is the first image.',
        process_img(img_path='examples/book2.jpeg', device=torch.cuda.current_device()),
        'This is the second image.',
        process_img(img_path='examples/book3.jpeg', device=torch.cuda.current_device()),
        'This is the third image.',
        process_img(img_path='examples/book4.jpeg', device=torch.cuda.current_device()),
        'This is the fourth image.',
        'Describe all images.'
    ]
    interleaved_sequence_1 = ''
    image_list_1 = []
    for item in image_text_sequence:
        if isinstance(item, str):  # text
            interleaved_sequence_1 += item
        else:  # image
            image_list_1.append(item)
            interleaved_sequence_1 += image_placeholder

    # prepare video example
    image_list_2, interleaved_sequence_2 = process_video('examples/AppleVR.mp4')
    interleaved_sequence_2 += "What's the woman doing in the video?"

    # # Instruct Model Inference
    # # -- image captioning
    Emu_instruct_caption(image, emu_model)
    # -- visual question answering
    Emu_inference(emu_model, [image], image_placeholder + question, system=image_system_msg)
    # -- image-text interleaved input, text output
    Emu_inference(emu_model, image_list_1, interleaved_sequence_1, system='')
    # -- video understanding
    Emu_inference(emu_model, image_list_2, interleaved_sequence_2, system=video_system_msg, length_penalty=1.0)

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

    
    # init FSDP
    from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    )
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    wrapper_kwargs = dict(
        process_group=None,
        cpu_offload=CPUOffload(offload_params=False),
        device_id=torch.cuda.current_device(),
        auto_wrap_policy=size_based_auto_wrap_policy,

    )
    model.wrap_fsdp(wrapper_kwargs)
    

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
    instruct_example(model)
    
    # torch.distributed.barrier() 
    # if args.instruct:
    #     instruct_example()
    # else:
    #     pretrain_example()

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

    # for n,_ in emu_model.named_parameters():
    #     if "input_layernorm" in n or "post_attention_layernorm" in n:
    #         print(n)
    
    WORLD_SIZE = torch.cuda.device_count()
    # WORLD_SIZE = 8
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, emu_model, args),
        nprocs=WORLD_SIZE,
        join=True)