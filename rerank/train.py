import os
import gc
import pdb
import random
from random import shuffle
import shutil
import time
import warnings
import sys

from torch.optim import optimizer
sys.path.append('../')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from transformers import get_linear_schedule_with_warmup
from backbone.resnet import *
from model import *
from dataset import *
from arguments import args 


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def log_loss(step_record, loss_record, save_name):
    if not os.path.exists("img"):
        os.mkdir("img")
    plt.plot(step_record, loss_record, lw=2)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('loss curve')
    plt.grid(True)
    plt.savefig(os.path.join("img", save_name))
    plt.close()
    
def run(args):
    if args.local_rank == -1:
        device = torch.device("cuda")
        args.distributed = False
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.tcp, rank=args.local_rank, world_size=args.n_gpu)
        args.distributed = True
    args.device = device

    # model 
    model = MatchingModel(args).to(args.device)
    cudnn.benchmark = True

    # data loader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = TextDataset(
        args,
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.local_rank != -1 else torch.utils.data.RandomSampler(dataset)
    params = {"batch_size": args.batch_size_per_gpu, "sampler": sampler, "num_workers": 2, "pin_memory": True}
    dataloader = torch.utils.data.DataLoader(dataset, **params)
    
    step_tot = args.num_train_epochs * len(dataset) // args.batch_size_per_gpu // args.n_gpu
    if args.optim == 'hybrid':
        # optimizer for test encoder
        optimizer_tm =  torch.optim.AdamW(
            [
                {"params": model.module.text_encoder.parameters() if hasattr(model, 'module') else model.text_encoder.parameters()},
                {"params": model.module.classifier.parameters() if hasattr(model, 'module') else model.classifier.parameters()},
            ],
            lr=3e-5,
        )
        scheduler = get_linear_schedule_with_warmup(optimizer_tm, num_warmup_steps=0.1*step_tot, num_training_steps=step_tot)
        
        # optimizer for image encoder
        optimizer_im = torch.optim.SGD(
            [
                {"params": model.module.image_encoder.parameters() if hasattr(model, 'module') else model.image_encoder.parameters()},
            ],
            lr=1e-3,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        optimizer = [optimizer_tm, optimizer_im]
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*step_tot, num_training_steps=step_tot)
        scheduler = None 
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        scheduler = None
    else:
        raise Exception("No such optimizer")

    # loop for trainng
    global_step = 0
    tot_loss = 0
    loss_record = [[],[]]
    for epoch in range(args.num_train_epochs):
        if args.distributed:
            sampler.set_epoch(epoch)
        if args.local_rank == 0:
            print("-"*20, "Epoch: %d" % epoch, "-"*20)
        
        # switch to train mode
        model.train()
        for i, batch in tqdm(enumerate(dataloader)):
            # params
            params = {
                # 'image': batch[0].cuda(),
                'gloss': batch[0].cuda(),
                'gloss_attention_mask': batch[1].cuda(),
                'labels': batch[2].cuda(),
            }
            # compute output
            loss, scores = model(**params)
            # compute gradient and do optimizer step
            optimizer.zero_grad() if not isinstance(optimizer, list) else [opt.zero_grad() for opt in optimizer]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step() if not isinstance(optimizer, list) else [opt.step() for opt in optimizer]
            if scheduler is not None:
                scheduler.step()
            
            global_step += 1
            tot_loss += loss.item() / args.record_step       
            if args.local_rank == 0 and global_step % args.record_step == 0:
                loss_record[0].append(global_step)
                loss_record[1].append(tot_loss)
                tot_loss = 0 
            if args.local_rank == 0 and global_step % (args.record_step * 10) == 0:
                print(f"Total loss record: {loss_record[0][-1]}, {loss_record[1][-1]}")
                log_loss(loss_record[0], loss_record[1], f"{args.model}.png")
        
        # save checkpoint
        if args.local_rank == 0 and (epoch+1) % args.save_epoch == 0:
            if not os.path.exists('ckpt'):
                os.mkdir('ckpt')
            if not os.path.exists(args.ckpt_dir):
                os.mkdir(args.ckpt_dir)
            torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), os.path.join(args.ckpt_dir, f"ckpt_{epoch}.pth"))


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = max(0.001, args.lr * (0.1**(epoch // 30)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('-'*5, lr, '-'*5)


if __name__ == '__main__':
    set_seed(42)
    run(args)
