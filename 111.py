# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


from torch.optim.lr_scheduler import StepLR

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
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from torch.distributed.fsdp import (
CPUOffload,
MixedPrecision,
ShardingStrategy,
BackwardPrefetch,
)

import os


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '32131'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(1, 13, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(1,1,100))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.test1 = nn.Parameter(torch.zeros(1, 1, 100))
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.attn = Attention()

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    def parge_param_info(self):
        wrap_list = []
        non_wrap_list = []
        for n, _ in self.named_parameters():
            if not '.' in n:
                non_wrap_list.append(n)
            else:
                wrap_list.append(n)           
        return  wrap_list, non_wrap_list  
    
def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))
        
        
def test(model, rank, world_size, test_loader):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            ddp_loss[0] += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
            100. * ddp_loss[1] / ddp_loss[2]))
        
def fsdp_main(rank, world_size, args):
    setup(rank, world_size)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform)

    sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)

    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    torch.cuda.set_device(rank)


    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    
    model = Net()
    
    def find_leaf_modules_with_parents(model, parent_name='', result_dict=None):
        if result_dict is None:
            result_dict = {}

        for name, module in model.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name  # 构建当前模块的全名

            if len(list(module.children())) == 0:
                result_dict[full_name] = module  # 保存叶子模块及其完整名称
            else:
                find_leaf_modules_with_parents(module, full_name, result_dict)

        return result_dict
    result_dict = find_leaf_modules_with_parents(model)
    print(result_dict)
    
    def apply_with_stopping_condition(module, apply_fn, apply_condition=None, stopping_condition=None, **other_args):
        # if stopping_condition(module):
        #     return
        if apply_condition(module):
            print(module)
            apply_fn(module, **other_args)
            return
            
        for name, child in module.named_children():
            apply_with_stopping_condition(
                child,
                apply_fn,
                apply_condition=apply_condition,
                stopping_condition=stopping_condition,
                **other_args
            )

    # print(model.parameters)
    # wrap_list, non_wrap_list = model.parge_param_info()
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1
    )
    def only_leaf_auto_wrap_policy(
        module: nn.Module,
        recurse: bool,
        nonwrapped_numel: int,
        # Additional custom arguments
        min_num_params: int = int(1e8),
    ) -> bool:
        # 如果返回true就停止wrap 返回false就recurse继续
        print(module)
        print(len(list(module.children())))
        return ~(len(list(module.children())) == 0)
    
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
    with enable_wrap(wrapper_cls=FSDP, **wrapper_kwargs):
        pass
        
        # def _recursive_getattr(model, attrs):
        #     if len(attrs) == 1:
        #         return 
        #     return _recursive_getattr(getattr(model, attrs[-1]), attrs[:-1])
        
        # for attr_str,v in result_dict.items():
        #     attr = eval("model."+attr_str) 
        #     print(attr)
        #     attr = wrap(v)
            
            # attrs = attr_str.split('.')
            # model_attr = _recursive_getattr(model, attrs)
            # print(model_attr)
            # model_attr = wrap(v)
        
        
            
        
        # for k,v in result_dict.items():
        #     setattr(model, k, wrap(v))
        
        
        # for name, module in model.named_children():
        #     print(name)
            # setattr(model, name, wrap(module))
        
        #     print(name)
        #     # setattr(model, name, wrap(module))1
        # model.conv1 = wrap(model.conv1)
        # model.conv2 = wrap(model.conv2)
        # model.dropout1 = wrap(model.dropout1)
        # model.dropout2 = wrap(model.dropout2)
        # model.fc1 = wrap(model.fc1)
        # model.fc2 = wrap(model.fc2)
        # model.attn.qkv = wrap(model.attn.qkv)
        # model.conv1._fsdp_wrap=True
        # model.conv2._fsdp_wrap=True
        # model.dropout1._fsdp_wrap=True
        # model.dropout2._fsdp_wrap=True
        # model.fc1._fsdp_wrap=True
        # model.fc2._fsdp_wrap=True
        # model.attn.qkv._fsdp_wrap=True
        
        # def apply_fn(m):
        #     wrap(m)
        
        # apply_with_stopping_condition(
        #     module=model,
        #     apply_fn=lambda m: wrap(m),
        #     apply_condition=lambda m: len(list(m.children())) == 0,
        #     stopping_condition=lambda m: isinstance(m, FSDP),
        # )
    
    model = FSDP(model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=my_auto_wrap_policy,
                device_id=torch.cuda.current_device(),
                use_orig_params=False,
                )
    print(model)
    # assert False
    # print(model.parameters)
    
    model.test1.data.to(torch.cuda.current_device())
    model.attn.q_bias.data.to(torch.cuda.current_device())
        
    # model = FSDP(model,
    #             auto_wrap_policy=my_auto_wrap_policy,
    #             device_id=torch.cuda.current_device(),)
    # model = FSDP(model, auto_wrap_policy=size_based_auto_wrap_policy)
    # print(model)

    # model.attn.q_bias.data = model.attn.q_bias.data.to(torch.cuda.current_device())
    
    a = model.test1.expand(1,-1,-1)
    b = model.attn.q_bias.expand(1,-1,-1)
    # print(model.test1)
    # print(model.attn.q_bias)
    # print(a)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    init_start_event.record()
    for epoch in range(1, args.epochs + 1):
        train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        test(model, rank, world_size, test_loader)
        scheduler.step()

    init_end_event.record()

    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")

    if args.save_model:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        # state_dict for FSDP model is only available on Nightlies for now
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "mnist_cnn.pt")

    cleanup()
    
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=640, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # WORLD_SIZE = torch.cuda.device_count()
    WORLD_SIZE=1
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)