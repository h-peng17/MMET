import os 
import gc
import pdb 
import json
from pdb import set_trace 
import random
from collections import defaultdict
from random import sample, shuffle
from tqdm import tqdm

import torch 
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from transformers import BertTokenizer

from model import MatchingModel
from arguments import args
from utils import compute_metric
from eval_dataset import CPDataset

def run(args):
    if args.local_rank == -1:
        device = torch.device("cuda")
        args.distributed = False
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.distributed = True
    args.device = device

    # model 
    model = MatchingModel(args)
    if args.ckpt != 'None':
        print(f"Load from {args.ckpt}")
        model.load_state_dict(torch.load(args.ckpt), strict=False)
    model.to(args.device)
    model.eval()
    cudnn.benchmark = True

    # data loader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = CPDataset(
        args,
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
    )
    params = {"batch_size": 1, "num_workers": 2, "pin_memory": True, "shuffle": False}
    dataloader = torch.utils.data.DataLoader(dataset, **params)
    
    scores = np.zeros((len(dataset), 200), dtype=np.float)
    preds = np.zeros((len(dataset), 200), dtype=np.int32)
    labels = np.zeros((len(dataset)), dtype=np.int32)
    for i, batch in enumerate(tqdm(dataloader)):
        recall_size = batch[0].shape[1] - 1
        in_bsz = 16
        count_in_batch = int(recall_size // in_bsz) if  recall_size % in_bsz == 0 else int(recall_size // in_bsz) + 1
        score = []
        # data in one batch 
        images = batch[0].squeeze(0).cuda()
        glosses = batch[1].squeeze(0).cuda()
        glosses_att_mask = batch[2].squeeze(0).cuda()
        # anchor 
        anchor_image = images[0:1]
        anchor_gloss = glosses[0:1]
        anchor_gloss_att_mask = glosses_att_mask[0:1]
        # data
        images = images[1:]
        glosses = glosses[1:]
        glosses_att_mask = glosses_att_mask[1:]
        # expert mask 
        expert_masks = batch[-2].squeeze(0).cuda()
        image_expert_masks = batch[-1].squeeze(0).cuda()
        for j in range(count_in_batch):
            image = torch.cat((anchor_image, images[j*in_bsz*args.image_expert_count:(j+1)*in_bsz*args.image_expert_count]), dim=0)
            gloss = torch.cat((anchor_gloss, glosses[j*in_bsz*args.expert_count:(j+1)*in_bsz*args.expert_count]), dim=0)
            gloss_att_mask = torch.cat((anchor_gloss_att_mask, glosses_att_mask[j*in_bsz*args.expert_count:(j+1)*in_bsz*args.expert_count]), dim=0)
            expert_mask = expert_masks[j*in_bsz:(j+1)*in_bsz]
            image_expert_mask = image_expert_masks[j*in_bsz:(j+1)*in_bsz]
            params = {
                'image': image,
                'gloss': gloss,
                'gloss_attention_mask': gloss_att_mask,
                'to_squeeze': False
            }
            if args.model == 'clip':
                text_score, image_score = model(**params)
                text_score = text_score.reshape(-1, args.expert_count)
                image_score = image_score.reshape(-1, args.image_expert_count)
                if text_score.shape[0] == 0:
                    continue
                text_score = (text_score * expert_mask).sum(-1) / expert_mask.sum(-1)
                image_score = (image_score * image_expert_mask).sum(-1) / image_expert_mask.sum(-1)
                in_score = text_score + image_score
            else:
                in_score = model(**params).reshape(-1, args.expert_count)
                if in_score.shape[0] == 0:
                    continue
                # in_score = (in_score * expert_mask).sum(-1) / expert_mask.sum(-1)
                in_score, _ = in_score.max(-1)
            score.append(in_score.detach().cpu())
            del in_score
        score = torch.cat(score, dim=0)
        label = batch[3].squeeze()
        pred = batch[4].squeeze(0)
        scores[i][:score.shape[0]] = score.numpy()
        preds[i][:pred.shape[0]] = pred.numpy()
        labels[i] = label
    if not os.path.exists("result"):
        os.mkdir("result")

    if args.expert_count > 1 or args.image_expert_count > 1:
        args.model = 'ensemble_' + args.model
        print('Ensemble')
    else:
        print("No Ensemble")

    np.save(f"result/{args.model}_scores.npy", scores)
    np.save(f"result/{args.model}_preds.npy", preds)
    np.save(f"result/{args.model}_labels.npy", labels)
    compute_metric(scores, preds, labels, args.ckpt)

    
if __name__ == '__main__':
    run(args)
