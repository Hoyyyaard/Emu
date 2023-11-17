from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
import logging
import os
import argparse
from PIL import Image
import torch
from models.modeling_emu import Emu
from collections import Counter
import json
from peft import LoraConfig, get_peft_model
import tqdm
from utils import process_img, process_video
import tqdm
import random
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import logging

task2details_map = {
    'done assembling kit.' : 'Place all the objects on the table in the position corresponding to the shape on the assembly kit.',
    'done packing shapes.' : 'Move one object on the table into the box.',
    'done packing blocks.' : 'Fill the box with objects placed on the table.',
    'done packing objects.' : 'Move one object on the table into the box.',
    'done placing blocks in bowls.' : 'Fill all bowls with the blocks on the table, asking that the bowl be a different color than the block you put in.',
    'done stacking block pyramid.' : 'Use the blocks on the table to build the pyramid on the brown board.',
    'done separating pile.' : 'Move all the small pieces into the box.' 
}


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
        "--detail_task",
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
        "--save_ckpt_epoch",
        type=int,
        default=10,

    )
    parser.add_argument(
        "--instruct",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--pretrain_predict_task",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--instruction_tuning",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--co_finetune",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--fsdp",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--toy",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--pretrain",
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
    parser.add_argument(
        "--pretrain_lora_ckpt",
        type=str,
        default='None',
    )
    args = parser.parse_args()
    return args

args = parse_args()
assert not(args.pretrain and args.instruction_tuning)
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
        # episode [domain:[robot,internet],sequence,image_p]
        self._episodes = []
        self.image_token = '<image>'
        self.end_token = '</s>'

        if args.pretrain_predict_task:
            logger.info("Load ROBOT Pretrain Tasks")
            robot_episodes = self._parse_robot_dataset_pretrain()
        elif args.instruction_tuning:
            logger.info("Load Predict All Subtasks Tasks")
            robot_episodes = self._parse_robot_dataset()
        else:
            logger.info("Load Predict All Subtasks Tasks")
            robot_episodes = self._parse_robot_dataset()
            
        llava_episodes = []
        if args.co_finetune:
            llava_episodes = self._parse_llava_dataset()
        if args.toy:
            random.shuffle(llava_episodes)
            random.shuffle(robot_episodes)
            llava_episodes = llava_episodes[:int(len(llava_episodes)/10)]
            robot_episodes = robot_episodes[:int(len(robot_episodes)/10)]
        self._episodes.extend(llava_episodes)
        self._episodes.extend(robot_episodes)
        
        max_len = 0
        for epi in self._episodes:
            if max_len < len(epi['image_p']):
                max_len = len(epi['image_p'])
        for i in range(len(self._episodes)):
            self._episodes[i]['image_p'].extend(['None'] * (max_len-len(self._episodes[i]['image_p'])))
            self._episodes[i]['sequence'] += self.end_token
    
    def _parse_llava_dataset(self):
        llava_episodes = []
        instruct_path_list = ['/root/Projects/dataset/llava_instruct_150k/llava_instruct_80k.json']
        image_path_prefix = '/root/Projects/dataset/llava_instruct_150k/images/train2017'
        pbar = tqdm.tqdm(desc='load internet data')
        for instruct_path in instruct_path_list:
            with open(instruct_path,"r") as f:
                instruction_data = json.load(f)
            for id in instruction_data:
                episode_dict = {}
                imgid = id['image']
                episode_dict['image_p'] = [f'{image_path_prefix}/{imgid}']
                episode_dict['sequence'] = '[USER]: '+ \
                                            id['conversations'][0]['value'].replace(self.image_token, Chat.image_placeholder) + \
                                            '\n[ASSISTANT]:' + \
                                            id['conversations'][1]['value']
                episode_dict['domain'] = 'internet'
                pbar.update(1)
                llava_episodes.append(episode_dict)
        return llava_episodes
    
    def _parse_robot_dataset(self):
        robot_episodes = []
        ebar = tqdm.tqdm(desc='load robot domain data')
        for task in os.listdir(self._dataset_path):
            p1 = os.path.join(self._dataset_path, task)
            # for epi in (os.listdir(p1)[:int(len(os.listdir(p1))/2)]):
            for epi in os.listdir(p1):
                gt_subtask_states = []
                gt_subtask = ''
                episode_dict = {}
                
                p2 = os.path.join(p1, epi)
                with open(f'{p2}/info.json', 'r') as f:
                    task_info = json.load(f)
                episode_dict['domain'] = 'robot'
                task = list(task_info.values())[0]
                if args.detail_task:
                    task = task2details_map[task]
                episode_dict['image_p'] =  [f'{p2}/origin_rgb.png']
                
                for si,subtask in enumerate(list(task_info.values())[1:]):
                    gt_subtask+=(f'{si+1}.{subtask}\n')
                # gt_subtask+='<End>'
                    # gt_subtask_states.append(f'{p2}/subtask{si+1}_rgb.png')
                episode_dict['sequence'] = Chat.user_prompt_instruction_tuning_1(task, gt_subtask)
                # episode_dict['gt_subtask_states_p'] = gt_subtask_states
                ebar.update(1)
                robot_episodes.append(episode_dict)
        return robot_episodes   

    def _predict_subtask_from_2_frames(self, task, subtasks, states_p):
        episode_dict = {}
        frame_id = random.randint(0, len(states_p)-2)
        episode_dict['domain'] = 'robot'
        seq = f'Manipulation task: {task}.\n State1: {Chat.image_placeholder}\n\
                                    State2: {Chat.image_placeholder}\n \
                                    Subtask executed from State1 to State2: {subtasks[frame_id]}'
        instr_seq = f'{Chat.system_prompt_inf()} [USER]: Manipulation task: {task}.\n State1: {Chat.image_placeholder}\n\
                                    State2: {Chat.image_placeholder}\n \
                                    Subtask executed from State1 to State2: [ASSISTANT]: {subtasks[frame_id]}'
        episode_dict['sequence'] = seq if not args.instruct else instr_seq
        
                                    
        episode_dict['image_p'] = [states_p[frame_id], states_p[frame_id+1]]
        return episode_dict
    
    def _predict_task_from_all_frames(self, task, subtasks, states_p):
        episode_dict = {}
        episode_dict['domain'] = 'robot'
        seq = 'Given a sequence of manipulation task states: '
        instr_seq = f'{Chat.system_prompt_inf()} [USER]: Given a sequence of manipulation task states: '
        for st in range(len(states_p)):
            seq += f'State{st}: {Chat.image_placeholder}'
            instr_seq += f'State{st}: {Chat.image_placeholder}'
        seq += f"The task the sequence finished : {task}."
        instr_seq += f"The task the sequence finished: [ASSISTANT]: {task}."
        episode_dict['sequence'] = seq if not args.instruct else instr_seq
        episode_dict['image_p'] = states_p
        return episode_dict
    
    def _predict_success_from_all_frames_and_task(self, task, subtasks, states_p):
        episode_dict = {}
        success = random.randint(0, 1)
        episode_dict['domain'] = 'robot'
        seq = f'Given task description : {task}. And a sequence of manipulation task states: '
        instr_seq = f'{Chat.system_prompt_inf()} [USER]: Given task description : {task}. And a sequence of manipulation task states: '
        frame_num = len(subtasks) if success else  random.randint(1, len(subtasks))
        for st in range(frame_num):
            seq += f'State{st}: {Chat.image_placeholder}'
            instr_seq += f'State{st}: {Chat.image_placeholder}'
        success = 'yes' if success else 'no'
        seq += f"If the task Success: {success}."
        instr_seq += f"If the task Success: [ASSISTANT]: {success}."
        episode_dict['sequence'] = seq if not args.instruct else instr_seq
        episode_dict['image_p'] = states_p[:frame_num]
        return episode_dict

    def _predict_all_subtasks_from_task_and_init_state(self, task, subtasks, states_p):
        episode_dict = {}
        episode_dict['domain'] = 'robot'
        seq = f'Given manipulation task description: {task}. Given the initial state of the task: {Chat.image_placeholder} Subtasks to finish the task step by step:'
        instr_seq = f'{Chat.system_prompt_inf()} [USER]: Given manipulation task description: {task}. Given the initial state of the task: {Chat.image_placeholder} Subtasks to finish the task step by step: [ASSISTANT]: '
        for si,st in enumerate(subtasks):
            seq += f'{si}:{st}\n'
            instr_seq += f'{si}:{st}\n'
        episode_dict['sequence'] = seq if not args.instruct else instr_seq
        episode_dict['image_p'] = [states_p[0]]
        return episode_dict
    
    def _predict_next_subtask_from_state_and_task(self, task, subtasks, states_p):
        episode_dict = {}
        frame_id = random.randint(0, len(states_p)-2)
        episode_dict['domain'] = 'robot'
        seq = f'Given manipulation task description: {task}. Given the current state of the task: {Chat.image_placeholder} Next one subtask to finish the task step by step: {subtasks[frame_id]}'
        instr_seq = f'{Chat.system_prompt_inf()} [USER]: Given manipulation task description: {task}. Given the current state of the task: {Chat.image_placeholder} Next one subtask to finish the task step by step: [ASSISTANT]: {subtasks[frame_id]}'
        episode_dict['sequence'] = seq if not args.instruct else instr_seq
        episode_dict['image_p'] = [states_p[frame_id]]
        return episode_dict
    
    def _parse_robot_dataset_pretrain(self):
        pretrain_task = ['predict_subtask_from_2_frames',
                         'predict_task_from_all_frames',
                         'predict_success_from_all_frames_and_task',
                         'predict_all_subtasks_from_task_and_init_state',
                         'predict_next_subtask_from_state_and_task']
        counter = Counter()
        random.seed(0)
        
        robot_episodes = []
        ebar = tqdm.tqdm(desc='load robot domain data')
        for task in os.listdir(self._dataset_path):
            p1 = os.path.join(self._dataset_path, task)
            # for epi in (os.listdir(p1)[:int(len(os.listdir(p1))/2)]):
            for epi in os.listdir(p1):
                gt_subtask_states = []
                gt_subtask = []
                episode_dict = {}
                
                p2 = os.path.join(p1, epi)
                with open(f'{p2}/info.json', 'r') as f:
                    task_info = json.load(f)
                
                task = list(task_info.values())[0]
                if args.detail_task:
                    task = task2details_map[task]
                gt_subtask_states.append(f'{p2}/origin_rgb.png')
                for si,subtask in enumerate(list(task_info.values())[1:]):
                    gt_subtask.append(f'{subtask}')
                    gt_subtask_states.append(f'{p2}/subtask{si+1}_rgb.png')
                
                predict_task = random.choice(pretrain_task)
                dataset_method = getattr(self, f'_{predict_task}')
                episode_dict = dataset_method(task, gt_subtask, gt_subtask_states)
                counter[predict_task] += 1
                
                # episode_dict['gt_subtask_states_p'] = gt_subtask_states
                ebar.update(1)
                robot_episodes.append(episode_dict)
                
        print(counter)
        return robot_episodes   
        
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
        
    @classmethod
    def user_prompt_instruction_tuning_1(cls, task, subtasks=None):
        text_sequence_prefix = f'{cls.system_prompt_inf()} [USER]: Manipulation task: {task}\n Initial state: {cls.image_placeholder}.\n\
Subtasks to finsish this manipulation task step by step. Remember do not generate sub-tasks that are not related to the initial state. If you think the subtasks can finish the whole task you should stop output. \nSubtasks:\n [ASSISTANT]:'
        if not subtasks is None:
            return text_sequence_prefix+subtasks
        else:
            return text_sequence_prefix


class Emu_Robot_Pipeline():
    
    def __init__(self, cast_type=torch.bfloat16) -> None:
        self.model = self._prepare_model().to(cast_type)
    
    def ddp_model(self):
        self.model.cuda()
        self.model = DDP(self.model, find_unused_parameters=True)
        
    
    def fsdp_model(self):
        self.model.wrap_fsdp()
    
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
            try :
                msg = model.load_state_dict(ckpt, strict=False)
            except:
                del ckpt['decoder.lm.lm_head.weight']
                del ckpt['decoder.lm.model.embed_tokens.weight']
                msg = model.load_state_dict(ckpt, strict=False)
                
            # model.eval()
            logger.info(f"=====> pretrain model get model.load_state_dict msg: {msg}")
        
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

            if not args.pretrain_lora_ckpt == 'None':
                ckpt = torch.load(args.pretrain_lora_ckpt, map_location="cpu")
                msg = model.load_state_dict(ckpt, strict=False)
                print(f"=====> Load pretrain_lora_ckpt: {msg}")
                model.decoder.lm = model.decoder.lm.merge_and_unload()
                model.decoder.lm = get_peft_model(model.decoder.lm, lora_config)
            
            if args.instruct:
                logger.info(f"=====> loading from ckpt_path {args.ckpt_path}")
                ckpt = torch.load(args.ckpt_path, map_location="cpu")
                msg = model.load_state_dict(ckpt, strict=False)
                # try:
                #     msg = model.load_state_dict(ckpt, strict=False)
                # except:
                #     del ckpt['decoder.lm.base_model.model.lm_head.weight']
                #     del ckpt['decoder.lm.base_model.model.model.embed_tokens.weight']
                #     msg = model.load_state_dict(ckpt, strict=False)
                    
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
        
        
def train(rank, world_size, args, pipeline, dataset):
    
    
    # val_dataset = BenchMark(dataset_path=args.val_data_path)
    train_sampler = DistributedSampler(dataset, 
                        rank=rank, 
                        num_replicas=torch.cuda.device_count(), 
                        shuffle=True)
    train_loader = torch.utils.data.DataLoader(dataset,
                            batch_size=args.batch_size ,
                            num_workers=16,
                            sampler=train_sampler,
                            drop_last=True)

    # val_loader = torch.utils.data.DataLoader(val_dataset,
    #                         batch_size=args.batch_size ,
    #                         num_workers=1,
    #                         drop_last=True)
    # val_sampler = DistributedSampler(val_dataset, 
    #                         rank=rank, 
    #                         num_replicas=torch.cuda.device_count(), 
    #                         shuffle=True)
    
    # 将模型的参数分成几个组 每个组不同的学习率
    param_groups = [
                {'params': pipeline.model.module.visual.parameters(), 'lr': 4 * args.lr_base},
                {'params': pipeline.model.module.decoder.parameters(), 'lr': 3 *  args.lr_base},
                {'params': pipeline.model.module.cformer.parameters(), 'lr': 10 *  args.lr_base}
            ]
    # if args.fsdp:
    #     if args.instruction_tuning:
    #         param_groups = [
    #         {'params': pipeline.model.decoder.parameters(), 'lr': 3 *  args.lr_base},
    #         ]
    #     else:
    #         param_groups = [
    #         {'params': pipeline.model.visual.parameters(), 'lr': 4 * args.lr_base},
    #         {'params': pipeline.model.decoder.parameters(), 'lr': 3 *  args.lr_base},
    #         {'params': pipeline.model.cformer.parameters(), 'lr': 10 *  args.lr_base}
    #         ]
    # else:
    #     if args.instruction_tuning:
    #         param_groups = [
    #         {'params': pipeline.model.module.decoder.parameters(), 'lr': 3 *  args.lr_base},
    #         ]
    #     else:
    #         param_groups = [
    #             {'params': pipeline.model.module.visual.parameters(), 'lr': 4 * args.lr_base},
    #             {'params': pipeline.model.module.decoder.parameters(), 'lr': 3 *  args.lr_base},
    #             {'params': pipeline.model.module.cformer.parameters(), 'lr': 10 *  args.lr_base}
    #         ]
    
    # float16的有效动态范围： 5.960464477539063e-08 ~65504 故default的eps为 1e-8可能导致 计算中分母为0导致grad没有
    # nan但是optimizer step后出现nan
    optimizer = optim.AdamW(param_groups, lr = args.lr_base, betas=(0.9,0.98), weight_decay=0.05, eps=1e-6)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    global_step = 0
    
    pipeline.model.module.visual.requires_grad_(True)
    pipeline.model.module.cformer.requires_grad_(True)
    pipeline.model.module.decoder.lm.base_model.model.stu_regress_head.requires_grad_(True)
    pipeline.model.module.decoder.lm.base_model.model.lm_head.requires_grad_(True)
    
    # if args.fsdp:
    #     if args.instruction_tuning:
    #         pipeline.model.visual.requires_grad_(False)
    #         pipeline.model.cformer.requires_grad_(False)
    #         pipeline.model.decoder.lm.base_model.model.lm_head.requires_grad_(True)
    #     # elif args.pretrain:
    #     #     pipeline.model.visual.requires_grad_(True)
    #     #     pipeline.model.cformer.requires_grad_(True)
    #     #     pipeline.model.decoder.lm.base_model.model.lm_head.requires_grad_(True)
    #     else: 
    #         pipeline.model.visual.requires_grad_(True)
    #         pipeline.model.cformer.requires_grad_(True)
    #         pipeline.model.decoder.lm.base_model.model.stu_regress_head.requires_grad_(True)
    #         pipeline.model.decoder.lm.base_model.model.lm_head.requires_grad_(True)
    # else:
    #     if args.instruction_tuning:
    #         pipeline.model.module.visual.requires_grad_(False)
    #         pipeline.model.module.cformer.requires_grad_(False)
    #         pipeline.model.module.decoder.lm.base_model.model.lm_head.requires_grad_(True)
    #     # elif args.pretrain:
    #     #     pipeline.model.module.visual.requires_grad_(True)
    #     #     pipeline.model.module.cformer.requires_grad_(True)
    #     #     pipeline.model.module.decoder.requires_grad_(True)
    #     #     pipeline.model.module.decoder.set_grad_checkpointing()
    #     else:
    #         pipeline.model.module.visual.requires_grad_(True)
    #         pipeline.model.module.cformer.requires_grad_(True)
    #         pipeline.model.module.decoder.lm.base_model.model.stu_regress_head.requires_grad_(True)
    #         pipeline.model.module.decoder.lm.base_model.model.lm_head.requires_grad_(True)
    
    for epoch in range(1, args.epochs + 1):
        
        train_sampler.set_epoch(epoch)
        
        for bi, batch in enumerate(tqdm.tqdm(train_loader, desc=f'Epoch {epoch}')):
            
            global_step += 1 * torch.cuda.device_count()
            batch_images_p = []
            for b in range(args.batch_size):
                for fp in range(len(batch['image_p'])) :
                    if not batch['image_p'][fp][b] == 'None':
                        batch_images_p.append(batch['image_p'][fp][b])

            batch_images = torch.cat([process_img(img_path=fp, 
                                            device=torch.cuda.current_device()).to(torch.bfloat16) 
                                            for fp in batch_images_p 
                                            ],
                               dim=0)
            
            # batch_prompts = []
            # batch_prefix_index = []
            # for task, subtasks in zip(batch['task'], batch['gt_subtask']):
            #     seq = Chat.user_prompt_instruction_tuning(task, subtasks)
            #     batch_prompts.append(seq)

            batch_prompts = batch['sequence']
            
            # [B, max_seq_len]
            if args.fsdp :
                batch_input_tokens = pipeline.model.decoder.tokenizer(
                                            batch_prompts, 
                                            padding="longest", 
                                            return_tensors="pt",
                                            add_special_tokens=True,
                                            ).to(torch.cuda.current_device())
                ASSISTANT_TOKEN_ID = pipeline.model.decoder.tokenizer.convert_tokens_to_ids(['[ASSISTANT]'])[0]
            else:
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
                if bi % 10 == 0:
                    logger.info(f"Step: {global_step}, CLS loss: {loss_cls}, REG_loss: {loss_reg}")
                    writer.add_scalar("train_loss_reg_rank0", loss_reg.item(), global_step=global_step)
                    writer.add_scalar("train_loss_cls_rank0", loss_cls.item(), global_step=global_step)


            # loss = loss_cls + loss_reg if not args.instruction_tuning else loss_cls
            
            loss = loss_cls + loss_reg
            # if torch.cuda.current_device() == 0:
            #     print(f'Batch 1 infernece takes torch.cuda.memory_reserved : {torch.cuda.memory_reserved(torch.cuda.current_device())/1024**3.:3f} GB')
            
            optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
            loss.backward()
            scheduler.step()
            
            nono_zero_grad = 0
            if args.fsdp:
                for n,p in pipeline.model.named_parameters():
                    if not p.grad is None:
                        if not torch.sum(p.grad) == 0:
                            nono_zero_grad += 1
                            # print(n)
            else:
                for n,p in pipeline.model.module.named_parameters():
                    if not p.grad is None:
                        if not torch.sum(p.grad) == 0:
                            nono_zero_grad += 1
                            # print(n)
            assert nono_zero_grad > 0
            
            optimizer.step()
            torch.cuda.empty_cache()

            # if bi % 50 == 0:
            #     with torch.no_grad():
            #         val_sampler.set_epoch(epoch)
        
            #         for bi, batch in enumerate(tqdm.tqdm(val_loader)):
                        
            #             batch_images = torch.cat([process_img(img_path=fp, 
            #                                             device=torch.cuda.current_device()).to(torch.bfloat16) 
            #                                             for fp in batch['init_state_p'] 
            #                                             ],
            #                             dim=0)
                        
            #             batch_prompts = []
            #             batch_prefix_index = []
            #             for task, subtasks in zip(batch['task'], batch['gt_subtask']):
            #                 seq = Chat.user_prompt_instruction_tuning(task, subtasks)
            #                 batch_prompts.append(seq)
                        
            #             # [B, max_seq_len]
            #             batch_input_tokens = pipeline.model.module.decoder.tokenizer(
            #                                             batch_prompts, 
            #                                             padding="longest", 
            #                                             return_tensors="pt",
            #                                             add_special_tokens=True,
            #                                             ).to(torch.cuda.current_device())
                        
            #             ASSISTANT_TOKEN_ID = pipeline.model.module.decoder.tokenizer.convert_tokens_to_ids(['[ASSISTANT]'])[0]
            #             batch_prefix_index = torch.where(batch_input_tokens.input_ids==ASSISTANT_TOKEN_ID)[1]
            #             if args.instruction_tuning:
            #                 assert len(batch_prefix_index) == args.batch_size
                            
            #             args.batch_prefix_index = batch_prefix_index

            #             loss_cls, loss_reg = pipeline.model(batch_images,
            #                             batch_input_tokens.input_ids, 
            #                             batch_input_tokens.attention_mask, args).llm_loss
                
            #             if (rank == 0):
            #                 logger.info(f"Step: {global_step}, VAL CLS loss: {loss_cls}, VAL REG_loss: {loss_reg}")
            #                 writer.add_scalar("val_loss_reg_rank0", loss_reg.item(), global_step=global_step)
            #                 writer.add_scalar("val_loss_cls_rank0", loss_cls.item(), global_step=global_step)

            #         torch.cuda.empty_cache()
            
            # if global_step % 400 == 0:
        if epoch % args.save_ckpt_epoch == 0:
            print("Stop At Saving Parameters")
            # use a barrier to make sure training is done on all ranks
            dist.barrier()
            # state_dict for FSDP model is only available on Nightlies for now
            if args.fsdp:
                states = pipeline.model.state_dict()
            else:
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
    os.environ['MASTER_PORT'] = '55380'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(rank, world_size, args, dataset):
    
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    pipeline = Emu_Robot_Pipeline()
    if args.fsdp:
        pipeline.fsdp_model()
    else: 
        pipeline.ddp_model()
    
    train(rank, world_size, args, pipeline, dataset)
    
if __name__ == '__main__':
    WORLD_SIZE = torch.cuda.device_count()
    logger.info(f"world size : {WORLD_SIZE}")
    dataset = BenchMark(dataset_path=args.data_path)
    logger.info(args)
    
    mp.spawn(main,
        args=(WORLD_SIZE, args, dataset),
        nprocs=WORLD_SIZE,
        join=True)
    
    
    
    