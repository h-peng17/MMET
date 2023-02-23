import os 
import pdb 
import json 
import random
from collections import defaultdict, deque
from tqdm import tqdm, trange

import torch 
import torch.nn as nn
from PIL import Image
import numpy as np
import numpy
from transformers import BertTokenizer

from retrieve_image import ImageRecallModel, ImageEncoderModel, compute_multi_image_rep
from retrieve_text import TextRecallModel
from retrieve_utils import compute_topk, vote_topk, load_image_vector, load_once
from arguments import args

def construct_image_library(args, imageRecallModel):
    # insert data to image library
    image_vector, libid2dataid, libid2nodeid = load_image_vector(args)
    np.save(os.path.join("image_retrieval_result", 'libid2nodeid.npy'), libid2nodeid)
    np.save(os.path.join("image_retrieval_result", 'libid2dataid.npy'), libid2dataid)
    imageRecallModel.construct_library(image_vector)

def construct_text_library(args, textRecallModel):
    data = json.load(open(args.data_path))
    textRecallModel.clear_library()
    textRecallModel.construct_library(data)

def query_image_library_per_item(args, imageRecallModel, item):
    labels = imageRecallModel.query_library(item)
    return labels

def query_image_library(args, imageRecallModel, data):
    labels = imageRecallModel.query_library(data)
    return labels

def query_text_library_per_item(args, textRecallModel, item):
    results = textRecallModel.query_library(item['text'])
    return results

def query_text(args):
    # data
    data = json.load(open(args.data_path))
    # model 
    textRecallModel = TextRecallModel(args)
    # loop for query 
    labels = []
    preds = []
    texts = []
    scores = []
    for item in tqdm(data):
        text_query_labels, score = query_text_library_per_item(args, textRecallModel, item)
        pred = []
        text = []
        for result in text_query_labels:
            pred.append(result['node'])
            text.append(result['text'])
        preds.append(pred)
        texts.append(text)
        scores.append(score)
        labels.append(item['node'])
    print(f"Hit@1: {compute_topk(preds,labels, 1)}")
    print(f"Hit@3: {compute_topk(preds,labels, 3)}")
    print(f"Hit@10: {compute_topk(preds,labels, 10)}")
    print(f"Hit@100: {compute_topk(preds,labels, 100)}")
    # save result
    result = dict()
    for i in trange(len(labels)):
        label = labels[i]
        key = f"{label}-{i}"
        pred = preds[i]
        text = texts[i]
        score = scores[i]
        result[key] = {
            'sucess': [],
            'fail': []
        }
        for j in range(len(pred)):
            if pred[j] == label:
                result[key]['sucess'].append([pred[j], text[j], score[j]])
            else:
                result[key]['fail'].append([pred[j], text[j], score[j]])
    if not os.path.exists("text_retrieval_result"):
        os.mkdir("text_retrieval_result")
    prefix = args.prefix
    json.dump(result, open(os.path.join("text_retrieval_result", prefix+'text_recall_result.json'), 'w'))


def query_image(args):
    # data 
    libid2nodeid = np.load(os.path.join("image_retrieval_result", 'libid2nodeid.npy'))
    libid2dataid = np.load(os.path.join("image_retrieval_result", 'libid2dataid.npy'))
    data = json.load(open("../data/library_image.json"))
    image_vector, queryid2dataid, labels = load_image_vector(args)

    node2id = json.load(open("../data/node2id.json"))
    id2node = {v: k for k, v in node2id.items()}
    # model
    imageRecallModel = ImageRecallModel(args)
    preds, distances = query_image_library(args, imageRecallModel, image_vector)
    # compute metric
    gt_preds = []
    for pred in preds:
        gt_pred = []
        for p in pred:
            gt_pred.append(libid2nodeid[p])
        gt_preds.append(gt_pred)
    print(f"Hit@1: {compute_topk(gt_preds, labels, 1)}")
    print(f"Hit@3: {compute_topk(gt_preds, labels, 3)}")
    print(f"Hit@10: {compute_topk(gt_preds, labels, 10)}")
    print(f"Hit@100: {compute_topk(gt_preds, labels, 100)}")
    result = dict()
    for i in trange(len(labels)):
        label = id2node[labels[i]]
        key = f"{label}-{queryid2dataid[i]}" 
        pred = preds[i]
        distance = distances[i]
        result[key] = {
            'sucess': [],
            'fail': []
        }
        for j in range(len(pred)):
            nodeid = libid2nodeid[pred[j]]
            dataid = libid2dataid[pred[j]]
            assert id2node[nodeid] == data[dataid]['node'], "Error: Not Align!"
            if id2node[nodeid] == label:
                result[key]['sucess'].append([id2node[nodeid], data[dataid]['image'], float(distance[j])])
            else:
                result[key]['fail'].append([id2node[nodeid], data[dataid]['image'], float(distance[j])])
    prefix = args.prefix
    json.dump(result, open(os.path.join("image_retrieval_result", prefix+'image_recall_result.json'), 'w'))


if __name__ == "__main__":
    print(args)
    if args.mode == 'com_img_rep':
        compute_multi_image_rep(args)
    elif args.mode == 'con_img_lib':
        load_once(args)
        imageRecallModel = ImageRecallModel(args)
        construct_image_library(args, imageRecallModel)
    elif args.mode == 'con_text_lib':
        textRecallModel = TextRecallModel(args)
        construct_text_library(args, textRecallModel)
    elif args.mode == 'query_text':
        query_text(args)
    elif args.mode == 'query_image':
        if not os.path.exists(os.path.join("image_retrieval_result", args.img_save_name+'.npy')):
            load_once(args)
            print(f"{args.img_save_name} loaded.")
        query_image(args)
    




    










