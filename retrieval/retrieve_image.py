import os 
import pdb 
import sys 
sys.path.append('../')
import json 
from tqdm import tqdm
import random
from collections import defaultdict

import torch 
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import numpy
from transformers import BertTokenizer
import hnswlib

from backbone.resnet import *

class ImageRecallModel:
    def __init__(self, args):
        self.args = args 
        self.dim = 2048
        self.img_library_path = "image_retrieval_result/image_library.bin"
    
    def construct_library(self, data):
        p = hnswlib.Index(space='cosine', dim=self.dim)  # possible options are l2, cosine or ip

        p.init_index(max_elements=data.shape[0], ef_construction=500, M=60)
        p.set_ef(1000)
        p.set_num_threads(64)

        print(f"Adding {data.shape[0]} elements")
        p.add_items(data)
        
        print(f"Saving index to `{self.img_library_path}`")
        p.save_index(self.img_library_path)

    def query_library(self, data):
        # Declaring index
        p = hnswlib.Index(space='cosine', dim=self.dim)  # possible options are l2, cosine or ip
        p.load_index(self.img_library_path)

        # Query the elements for themselves and measure recall:
        labels, distances = p.knn_query(data, k=100)
        return labels, distances
    
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, args, transform):
        self.args = args
        self.transform = transform
        self.data = json.load(open(args.data_path))
        self.node2id = json.load(open("../data/node2id.json"))

    def __len__(self):
        return len(self.data)
    
    def __get__(self, img):
        img_path = "../data/images/" + img[:2] + "/" + img + ".jpg"
        img_tensor= self.transform(Image.open(img_path).convert('RGB'))
        return img_tensor
    
    def __getitem__(self, index):
        item = self.data[index]
        img = item['image']
        nodeid = self.node2id[item['node']]
        img_tensor = self.__get__(img)

        return img_tensor, torch.tensor([index], dtype=torch.int32), torch.tensor([nodeid], dtype=torch.int32)

class ImageEncoderModel(nn.Module):
    def __init__(self, args):
        super(ImageEncoderModel, self).__init__()
        self.args = args 
        self.encoder = resnet152(pretrained=True)
    
    def forward(self, img_input):
        return self.encoder(img_input)

def compute_multi_image_rep(args):
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
    model = ImageEncoderModel(args).to(args.device)
    cudnn.benchmark = True

    # data loader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = ImageDataset(
        args,
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
    )
    params = {"batch_size": 128, "shuffle": False, "sampler": None}
    dataloader = torch.utils.data.DataLoader(dataset, **params)

    # loop for compute image representation
    model.eval()
    for i, batch in enumerate(tqdm(dataloader)):
        params = {
            'img_input': batch[0].cuda(args.device, non_blocking=True),
        }
        index = batch[1].cuda(args.device, non_blocking=True)
        nodeid = batch[2].cuda(args.device, non_blocking=True)

        # compute output
        output = model(**params)
        output = torch.cat((index, nodeid, output), dim=-1)

        if args.distributed:
            output_gather_list = [torch.zeros_like(output) for _ in range(torch.cuda.device_count())]
            torch.distributed.all_gather(output_gather_list, output)
            output = torch.cat(output_gather_list, dim=0)
        
        if args.local_rank in [-1, 0]:  
            if not os.path.exists("image_retrieval_result"):
                os.mkdir("image_retrieval_result")
            output = output.detach().cpu().numpy()
            f = open(os.path.join("image_retrieval_result", args.img_save_name+'.txt'), 'a')
            np.savetxt(f, output)
            f.close()






