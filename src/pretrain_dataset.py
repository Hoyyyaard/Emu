from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
import os
import json
import numpy as np
import pickle
from PIL import Image
class Pretrain_Dataset(Dataset):
    
    def __init__(self, n_causal=32, _dataset_path='test_data/', just_end=False):
        self._dataset_path = _dataset_path
        self.just_end = just_end 
        self._episodes = []
        
        self._image_holder_tokens = ''
        for _ in range(n_causal):
            self._image_holder_tokens += '<image>'
        self._image_holder_tokens = '[IMG]' + self._image_holder_tokens + '[/IMG]'
        
        self._parse_dataset()
        
    def _parse_dataset(self):
        max_fp_len = 0
        for task in os.listdir(self._dataset_path):
            p1 = os.path.join(self._dataset_path, task)
            for epi in os.listdir(p1):
                p2 = os.path.join(p1, epi)
                sequence = ''
                state_files = []
                with open(f'{p2}/info.json', 'r') as f:
                    task_info = json.load(f)
                if not self.just_end:
                    for i,v in enumerate(list(task_info.values())):
                        if i == 0:
                            sequence += self._system_prompt(v)
                        else:
                            sequence += self._subtask_prompt(i, v)
                    state_files.append(f'{p2}/origin_rgb.png')
                    state_files.extend([f'{p2}/subtask{j+1}_rgb.png' for j in range(len(list(task_info.values()))-1)])
                else:
                    for i,v in enumerate(list(task_info.values())):
                        if i == 0:
                            sequence += self._system_prompt(v)
                        elif i == len(list(task_info.values())) - 1:
                            sequence += self._subtask_prompt(i, v)

                    state_files.append(f'{p2}/origin_rgb.png')
                    state_files.extend([f'{p2}/subtask{len(list(task_info.values()))-1}_rgb.png'])
                
                
                self._episodes.append([sequence, state_files])
                if len(state_files) > max_fp_len:
                    max_fp_len = len(state_files)
                    
        for ei,epi in enumerate(self._episodes):
            pad_num = max_fp_len - len(epi[1])
            self._episodes[ei][1].extend(['None']  * pad_num)
    
    def _system_prompt(self, task):
        return f'Task : {task}; Initial State : {self._image_holder_tokens}.\n Details:\n'
    
    def _subtask_prompt(self, index, subtask):
        return f'Subtask_{index} : {subtask}; State : {self._image_holder_tokens}.\n'
    
    def __getitem__(self, index):
        # return [prompt, [initial state file, subtask1 state file, .....]]
        return self._episodes[index]
    
    def __len__(self):
        return len(self._episodes)
    
    
    
class Visual_Decoding_Dataset(Dataset):
    
    def __init__(self, n_causal=32, _dataset_path='test_data/', just_end=False):
        self._dataset_path = _dataset_path
        self._episodes = []
        
        self._image_holder_tokens = ''
        for _ in range(n_causal):
            self._image_holder_tokens += '<image>'
        self._image_holder_tokens = '[IMG]' + self._image_holder_tokens + '[/IMG]'
        
        self._parse_dataset()
        
    def _parse_dataset(self):
        for task in os.listdir(self._dataset_path):
            p1 = os.path.join(self._dataset_path, task)
            for epi in os.listdir(p1):
                p2 = os.path.join(p1, epi)

                with open(f'{p2}/info.json', 'r') as f:
                    task_info = json.load(f)
                sequence = [
                    f'Task: {list(task_info.values())[0]}. Initial State:',
                    f'{p2}/origin_rgb.png',
                    f'\n Details:\nSubtask_1: {list(task_info.values())[1]}. State: ',
                ]
                gt_image = f'{p2}/subtask1_rgb.png'
            
                self._episodes.append([sequence, gt_image])

    
    def _system_prompt(self, task):
        return f'Task : {task}; Initial State : {self._image_holder_tokens}.\n Details:\n'
    
    def _subtask_prompt(self, index, subtask):
        return f'Subtask_{index} : {subtask}; State : '
    
    def __getitem__(self, index):
        return self._episodes[index]
    
    def __len__(self):
        return len(self._episodes)