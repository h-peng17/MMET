import os 
import pdb
from tqdm import tqdm
from collections import defaultdict, Counter

import numpy 
import numpy as np 


def load_once(args):
    data_dir = "image_retrieval_result"
    image_vector = np.loadtxt(os.path.join(data_dir, args.img_save_name+'.txt'))
    np.save(os.path.join(data_dir, args.img_save_name+'.npy'), image_vector)

def load_image_vector(args):
    data_dir = "image_retrieval_result"
    image_vector = np.load(os.path.join(data_dir, args.img_save_name+'.npy'))
    return image_vector[:, 2:], np.array(image_vector[:, 0], dtype=np.int32), np.array(image_vector[:, 1], dtype=np.int32)

def compute_topk(preds, labels, k):
    tot = len(labels)
    cor = 0
    for i in range(len(labels)):
        pred = preds[i][:k]
        label = labels[i]
        if label in pred:
            cor += 1 
    return cor / tot

def vote_topk(preds, labels, k):
    tot = len(labels)
    cor = 0
    for i in range(len(labels)):
        node_count = Counter()
        for p in preds[i]:
            node_count[p] += 1
        sorted_pred = [k for k, v in sorted(node_count.items(), key=lambda item:item[1], reverse=True)]
        label = labels[i]
        if label in sorted_pred[:k]:
            cor += 1
    return cor / tot


