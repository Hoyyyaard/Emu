from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
import logging
import os
import argparse
from PIL import Image
import torch
from models.modeling_emu import Emu
import json
from peft import LoraConfig, get_peft_model
import tqdm
from utils import process_img, process_video
import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import logging

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default='val_data/',

    )
    parser.add_argument(
        "--lr_base",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--bf16",
        action='store_true',
        default=True,

    )
    parser.add_argument(
        "--lora",
        action='store_true',
        default=False,

    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default='val_data/',

    )
    parser.add_argument(
        "--instruct",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--instruction_tuning",
        action='store_true',
        default=False,
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
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default='ckpts/Emu-instruct.pt',
    )
    
    args = parser.parse_args()
    return args

args = parse_args()
args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

class BenchMark(Dataset):
    
    def __init__(self, dataset_path='val_data/'):
        self._dataset_path = dataset_path
        self._episodes = []
        self._parse_dataset()
        
    def _parse_dataset(self):
        ebar = tqdm.tqdm()
        for task in os.listdir(self._dataset_path):
            p1 = os.path.join(self._dataset_path, task)
            for epi in os.listdir(p1):
                gt_subtask_states = []
                gt_subtask = ''
                episode_dict = {}
                
                p2 = os.path.join(p1, epi)
                with open(f'{p2}/info.json', 'r') as f:
                    task_info = json.load(f)
                episode_dict['task'] = list(task_info.values())[0]
                episode_dict['init_state_p'] =  f'{p2}/origin_rgb.png'
                
                for si,subtask in enumerate(list(task_info.values())[1:]):
                    gt_subtask+=(f'{si+1}.{subtask}\n')
                    gt_subtask_states.append(f'{p2}/subtask{si+1}_rgb.png')
                episode_dict['gt_subtask'] = gt_subtask
                # episode_dict['gt_subtask_states_p'] = gt_subtask_states
                ebar.update(1)
                self._episodes.append(episode_dict)
    
    def __getitem__(self, index):
        return self._episodes[index]
    
    def __len__(self):
        return len(self._episodes)
    

class Chat():
    
    image_placeholder = "[IMG]" + "<image>" * 32 + "[/IMG]"
    
    @classmethod
    def system_prompt_inf(cls):
        return f'You are a helpful robot assistant to manipulation tasks,\
I will give you a task and corresponding initial scene state in [IMG]ImageContent[/IMG],\
please help me generate a series of sub-tasks to complete this task step by step'

    @classmethod
    def user_prompt_inf(cls, image_list, task):
        text_sequence = f'Manipulation task: {task}.\n Initial state: {cls.image_placeholder}.\n\
Generate all subtasks to finsish this manipulation task step by step.'
        
        return {'text_sequence': text_sequence,
                'image_list': image_list}
        
    @classmethod
    def user_prompt_instruction_tuning(cls, task, subtasks=None):
        text_sequence_prefix = f'{cls.system_prompt_inf()} [USER]: Manipulation task: {task}\n Initial state: {cls.image_placeholder}.\n\
Subtasks to finsish this manipulation task step by step.\nSubtasks:\n [ASSISTANT]:'
        if not subtasks is None:
            return text_sequence_prefix+subtasks
        else:
            return text_sequence_prefix
        


class Emu_Robot_Pipeline():
    
    def __init__(self, cast_type=torch.bfloat16) -> None:
        self.model = self._prepare_model().to(cast_type).cuda()
    
    def ddp_model(self):
        self.model = DDP(self.model,find_unused_parameters=True)
    
    def describe_scene_example(self):
        image_text_sequence = [
        'Manipulation task: packing all the objects into the box.\n Initial state:',
        process_img(img_path='val_data/task1/epi_0/origin_rgb.png', device=torch.cuda.current_device()),
        'Subtasks:\n'
        ]
        interleaved_sequence_1 = ''
        image_list_1 = []
        for item in image_text_sequence:
            if isinstance(item, str):  # text
                interleaved_sequence_1 += item
            else:  # image
                image_list_1.append(item)
                interleaved_sequence_1 += Chat.image_placeholder

        # Pretrained Model Inference
        # -- in-context learning
        answer = self.inference(image_list_1, interleaved_sequence_1, instruct=True)
        print(answer)
        
    def example(self):
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
                interleaved_sequence_1 += Chat.image_placeholder

        # Pretrained Model Inference
        # -- in-context learning
        answer = self.inference(image_list_1, interleaved_sequence_1, instruct=True)
        print(answer)
    
    def _prepare_model(self, model_name='Emu-14B'):
        with open(f'models/{model_name}.json', "r", encoding="utf8") as f:
            model_cfg = json.load(f)
        logger.info(f"=====> model_cfg: {model_cfg}")
        
        model = Emu(**model_cfg, args=args) 

        if not args.instruct:
            logger.info(f"=====> loading from ckpt_path {args.ckpt_path}")
            ckpt = torch.load(args.ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt, strict=False)
            # model.eval()
            logger.info(f"=====> get model.load_state_dict msg: {msg}")
        
        if args.lora or args.instruct:
            print('Patching LoRA...')
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model.decoder.lm = get_peft_model(model.decoder.lm, lora_config)

            if args.instruct:
                logger.info(f"=====> loading from ckpt_path {args.ckpt_path}")
                ckpt = torch.load(args.ckpt_path, map_location="cpu")
                msg = model.load_state_dict(ckpt, strict=False)
                # model.eval()
                logger.info(f"=====> get model.load_state_dict msg: {msg}")
        
        return model

    def inference(self, image_list, text_sequence, system='', instruct=False, max_new_tokens=128, beam_size=5, length_penalty=0.0):
        if instruct:
            prompt = f"{system} [USER]: {text_sequence} [ASSISTANT]:".strip()
        else:
            prompt = text_sequence

        print(f"===> prompt: {prompt}")

        samples = {"image": torch.cat(image_list, dim=0), "prompt": prompt}

        output_text = self.model.generate(
            samples,
            max_new_tokens=max_new_tokens,
            num_beams=beam_size,
            length_penalty=length_penalty,
            repetition_penalty=1.0,
        )[0].strip()
        
        return output_text
        
        
def train(rank, world_size, args, pipeline):
    
    dataset = BenchMark(dataset_path=args.data_path)
    val_dataset = BenchMark(dataset_path=args.val_data_path)
    train_loader = torch.utils.data.DataLoader(dataset,
                            batch_size=args.batch_size ,
                            num_workers=4,
                            drop_last=True)
    train_sampler = DistributedSampler(dataset, 
                            rank=rank, 
                            num_replicas=torch.cuda.device_count(), 
                            shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                            batch_size=args.batch_size ,
                            num_workers=1,
                            drop_last=True)
    val_sampler = DistributedSampler(val_dataset, 
                            rank=rank, 
                            num_replicas=torch.cuda.device_count(), 
                            shuffle=True)
    
    # 将模型的参数分成几个组 每个组不同的学习率
    param_groups = [
        {'params': pipeline.model.module.visual.parameters(), 'lr': 4 * args.lr_base},
        {'params': pipeline.model.module.decoder.parameters(), 'lr': 3 *  args.lr_base},
        {'params': pipeline.model.module.cformer.parameters(), 'lr': 10 *  args.lr_base}
    ]
    
    # float16的有效动态范围： 5.960464477539063e-08 ~65504 故default的eps为 1e-8可能导致 计算中分母为0导致grad没有
    # nan但是optimizer step后出现nan
    optimizer = optim.AdamW(param_groups, lr = args.lr_base, betas=(0.9,0.98), weight_decay=0.05, eps=1e-6)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    global_step = 0
    pipeline.model.module.visual.requires_grad_(True)
    pipeline.model.module.cformer.requires_grad_(True)
    pipeline.model.module.decoder.lm.base_model.model.stu_regress_head.requires_grad_(True)
    pipeline.model.module.decoder.lm.base_model.model.lm_head.requires_grad_(True)
    
    for epoch in range(1, args.epochs + 1):
        
        train_sampler.set_epoch(epoch)
        
        for bi, batch in enumerate(tqdm.tqdm(train_loader, desc=f'Epoch {epoch}')):
            
            global_step += 1 * torch.cuda.device_count()
            
            batch_images = torch.cat([process_img(img_path=fp, 
                                            device=torch.cuda.current_device()).to(torch.bfloat16) 
                                            for fp in batch['init_state_p'] 
                                            ],
                               dim=0)
            
            batch_prompts = []
            batch_prefix_index = []
            for task, subtasks in zip(batch['task'], batch['gt_subtask']):
                seq = Chat.user_prompt_instruction_tuning(task, subtasks)
                batch_prompts.append(seq)

            
            # [B, max_seq_len]
            batch_input_tokens = pipeline.model.module.decoder.tokenizer(
                                            batch_prompts, 
                                            padding="longest", 
                                            return_tensors="pt",
                                            add_special_tokens=True,
                                            ).to(torch.cuda.current_device())
            ASSISTANT_TOKEN_ID = pipeline.model.module.decoder.tokenizer.convert_tokens_to_ids(['[ASSISTANT]'])[0]
            # for it in batch_input_tokens.input_ids:
            #     prefix_index = torch.where(it==ASSISTANT_TOKEN_ID)
            #     assert len(prefix_index[0]) == 1
            #     prefix_index = prefix_index[1][0]
            #     batch_prefix_index.append(prefix_index)
            batch_prefix_index = torch.where(batch_input_tokens.input_ids==ASSISTANT_TOKEN_ID)[1]
            if args.instruction_tuning:
                assert len(batch_prefix_index) == args.batch_size
                
            args.batch_prefix_index = batch_prefix_index
            loss_cls, loss_reg = pipeline.model(batch_images,
                            batch_input_tokens.input_ids, 
                            batch_input_tokens.attention_mask, args).llm_loss
    
            if (rank == 0):
                if bi % 50 == 0:
                    logger.info(f"Step: {global_step}, CLS loss: {loss_cls}, REG_loss: {loss_reg}")
                    writer.add_scalar("train_loss_reg_rank0", loss_reg.item(), global_step=global_step)
                    writer.add_scalar("train_loss_cls_rank0", loss_cls.item(), global_step=global_step)


            loss = loss_cls + loss_reg if not args.instruction_tuning else loss_cls
            # if torch.cuda.current_device() == 0:
            #     print(f'Batch 1 infernece takes torch.cuda.memory_reserved : {torch.cuda.memory_reserved(torch.cuda.current_device())/1024**3.:3f} GB')
            
            optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
            loss.backward()
            scheduler.step()
            
            nono_zero_grad = 0
            for n,p in pipeline.model.module.named_parameters():
                if not p.grad is None:
                    if not torch.sum(p.grad) == 0:
                        nono_zero_grad += 1
                        # print(n)
            assert nono_zero_grad > 0
            
            optimizer.step()
            torch.cuda.empty_cache()

            if bi % 50 == 0:
                with torch.no_grad():
                    val_sampler.set_epoch(epoch)
        
                    for bi, batch in enumerate(tqdm.tqdm(val_loader)):
                        
                        batch_images = torch.cat([process_img(img_path=fp, 
                                                        device=torch.cuda.current_device()).to(torch.bfloat16) 
                                                        for fp in batch['init_state_p'] 
                                                        ],
                                        dim=0)
                        
                        batch_prompts = []
                        batch_prefix_index = []
                        for task, subtasks in zip(batch['task'], batch['gt_subtask']):
                            seq = Chat.user_prompt_instruction_tuning(task, subtasks)
                            batch_prompts.append(seq)
                        
                        # [B, max_seq_len]
                        batch_input_tokens = pipeline.model.module.decoder.tokenizer(
                                                        batch_prompts, 
                                                        padding="longest", 
                                                        return_tensors="pt",
                                                        add_special_tokens=True,
                                                        ).to(torch.cuda.current_device())
                        
                        ASSISTANT_TOKEN_ID = pipeline.model.module.decoder.tokenizer.convert_tokens_to_ids(['[ASSISTANT]'])[0]
                        batch_prefix_index = torch.where(batch_input_tokens.input_ids==ASSISTANT_TOKEN_ID)[1]
                        if args.instruction_tuning:
                            assert len(batch_prefix_index) == args.batch_size
                            
                        args.batch_prefix_index = batch_prefix_index

                        loss_cls, loss_reg = pipeline.model(batch_images,
                                        batch_input_tokens.input_ids, 
                                        batch_input_tokens.attention_mask, args).llm_loss
                
                        if (rank == 0):
                            logger.info(f"Step: {global_step}, VAL CLS loss: {loss_cls}, VAL REG_loss: {loss_reg}")
                            writer.add_scalar("val_loss_reg_rank0", loss_reg.item(), global_step=global_step)
                            writer.add_scalar("val_loss_cls_rank0", loss_cls.item(), global_step=global_step)

                    torch.cuda.empty_cache()
            
            if global_step % 400 == 0:
                print("Stop At Saving Parameters")
                # use a barrier to make sure training is done on all ranks
                dist.barrier()
                # state_dict for FSDP model is only available on Nightlies for now
                states = pipeline.model.module.state_dict()
                
                if rank == 0:
                    if not os.path.exists(p:=args.log_dir+"/robot_ckpt"):
                        os.mkdir(p)
                    torch.save(states, args.log_dir+f"/robot_ckpt/finetune_{epoch}_{global_step}")

    
    
    # with torch.no_grad():
    #     for batch in tqdm.tqdm(train_loader):
            
    #         task = batch['task'][0]
    #         init_state = process_img(img_path=batch['init_state_p'][0])
    #         user_prompt = Chat.user_prompt([init_state], task)
    #         answer = pipeline.inference(**user_prompt, instruct=args.instruct, system=Chat.system_prompt())
    #         print("===> answer:", answer)
    #         # pipeline.describe_scene_example()
    #         break

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '55567'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(rank, world_size, args):
    
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    pipeline = Emu_Robot_Pipeline()
    pipeline.ddp_model()
    
    train(rank, world_size, args, pipeline)
    
if __name__ == '__main__':
    WORLD_SIZE = torch.cuda.device_count()
    logger.info(f"world size : {WORLD_SIZE}")

    
    mp.spawn(main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)
    
    
    
    