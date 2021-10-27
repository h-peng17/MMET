import os 
import gc
import pdb 
import json
from pdb import set_trace 
import random
from collections import defaultdict
from random import uniform

import clip

import torch 
import numpy as np
from PIL import Image
from transformers import BertTokenizer, LxmertTokenizer

DIR = "../data"

def get_per_image(image, transform):
    image_path = DIR + "/images/" + image[:2] + "/" + image
    if not image.endswith('jpg'):
        image_path += ".jpg"
    return transform(Image.open(image_path).convert('RGB'))

def get_per_text(text, max_length, tokenizer):
    if isinstance(tokenizer, (BertTokenizer, LxmertTokenizer)):
        input_ids = torch.zeros(max_length, dtype=torch.long)
        att_mask = torch.zeros(max_length, dtype=torch.float32)
        _ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(text) + ['[SEP]'])
        length = min(max_length, len(_ids))
        input_ids[:length] = torch.tensor(np.array(_ids[:length]), dtype=torch.long)
        att_mask[:length] = 1
    else:
        att_mask = torch.zeros(77, dtype=torch.float32)
        input_ids = tokenizer(text, truncate=True).squeeze(0)
    return input_ids, att_mask

def get_per_text_pair(gt_text, text, max_length, tokenizer):
    input_ids = torch.zeros(max_length, dtype=torch.long)
    att_mask = torch.zeros(max_length, dtype=torch.float32)
    _ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(gt_text) + ['[SEP]'] + tokenizer.tokenize(text))
    length = min(max_length, len(_ids))
    input_ids[:length] = torch.tensor(np.array(_ids[:length]), dtype=torch.long)
    att_mask[:length] = 1
    return input_ids, att_mask

def sample_item(length, k):
    uniform = torch.ones(length)
    return torch.multinomial(uniform, num_samples=k)

def sample_retrievals(query_retrieval, ng_count, ps_count):
    sample_count = ng_count + ps_count
    retrievals = torch.zeros((sample_count), dtype=torch.int32)
    if ps_count > 0:
        positive_retrieval = sample_item(len(query_retrieval['sucess']), k=ps_count)
        retrievals[:ps_count] = positive_retrieval
    if len(query_retrieval['fail']) > 0:
        negative_retrieval = sample_item(len(query_retrieval['fail']), k=sample_count-ps_count)
        retrievals[ps_count:] = negative_retrieval
    return retrievals, sample_count 

def get_retrievals(library_data, tokenizer, transform, node2id,
                query, query_retrieval, 
                ng_count, ps_count=0, max_length=64, 
                retrieval_image=True, text_pair=False, only_text=False):
    gt_node, gt_text = query['node'], query['text']
    retrievals, sample_count = sample_retrievals(query_retrieval, ng_count, ps_count)
    text_inputs = torch.zeros((sample_count, max_length), dtype=torch.long)
    text_masks = torch.zeros((sample_count, max_length), dtype=torch.float32)
    labels = torch.zeros((sample_count), dtype=torch.long)
    preds = torch.zeros((sample_count), dtype=torch.long)
    for i, idx in enumerate(retrievals):
        if i < ps_count:
            node, retrieval, _ = query_retrieval['sucess'][idx]
        else:
            node, retrieval, _  = query_retrieval['fail'][idx]
        text = library_data[node]['text'][random.randint(0, len(library_data[node]['text'])-1)] if retrieval_image else retrieval 
        text_input, text_mask = get_per_text(text, max_length, tokenizer) if not text_pair else \
                                get_per_text_pair(gt_text, text, max_length, tokenizer)
        text_inputs[i] = text_input
        text_masks[i] = text_mask 
        labels[i] = 1 if gt_node == node else 0
        preds[i] = node2id[node]
        assert (labels[i] == 1 and i < ps_count) or (labels[i] == 0 and i >= ps_count)
    if only_text:
        image_inputs = None
    else:
        image_inputs = torch.zeros((sample_count, 3, 224, 224), dtype=torch.float32)
        for i, idx in enumerate(retrievals):
            if i < ps_count:
                node, retrieval, _ = query_retrieval['sucess'][idx]
            else:
                node, retrieval, _  = query_retrieval['fail'][idx]
            image = retrieval if retrieval_image else library_data[node]['image'][random.randint(0, len(library_data[node]['image'])-1)]
            image_input = get_per_image(image, transform)
            image_inputs[i] = image_input
    return image_inputs, text_inputs, text_masks, labels, preds


class CPDataset(torch.utils.data.Dataset):
    def __init__(self, args, transform):
        self.args = args
        if args.model == 'clip': 
            print("Let's use CLIP config...")
            self.tokenizer = clip.tokenize
            _, preprocess = clip.load("ViT-B/16", jit=False)
            self.transform = preprocess
        else: 
            self.transform = transform
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.text_retrieval_data = json.load(open(os.path.join("../retrieval/text_retrieval_result", f"{args.data_prefix}text_retrieval_result.json")))
        self.image_retrieval_data = json.load(open(os.path.join("../retrieval/image_retrieval_result", f"{args.data_prefix}image_retrieval_result.json")))

        self.node2id = json.load(open(os.path.join(os.path.join(DIR, "node2id.json"))))
        
        library_text_data = json.load(open(os.path.join(DIR, "library_text.json")))
        library_image_data = json.load(open(os.path.join(DIR, "library_image.json")))
        self.library_data = {}
        for item in library_text_data:
            self.library_data[item['node']] = {
                'name': item['name'],
                'image': [],
                'text': []
            }
        for item in library_text_data:
            self.library_data[item['node']]['text'].append(item['text'])
        for item in library_image_data:
            self.library_data[item['node']]['image'].append(item['image'])
        
        # filter entity which has not retrievals 
        self.data = json.load(open(os.path.join(DIR, f"{args.data_prefix}data.json")))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        anchor = self.data[index]
        key = f"{anchor['node']}-{index}"

        image_retrieval = self.image_retrieval_data[key]
        ps_count = min(len(image_retrieval['sucess']), 16) if self.args.training else len(image_retrieval['sucess'])
        ng_count = min(self.args.retrieval_count//2, len(image_retrieval['fail'])) if self.args.training else len(image_retrieval['fail'])
        image_inputs_im, text_inputs_im, text_masks_im, labels_im, preds_im = get_retrievals(self.library_data, self.tokenizer, self.transform, self.node2id,
                                                                                anchor, image_retrieval, 
                                                                                ng_count, ps_count, self.args.max_seq_length, 
                                                                                retrieval_image=True, text_pair=False, only_text=False)
        anchor_image_input = get_per_image(anchor['image'], self.transform)
        anchor_text_input, anchor_text_mask = get_per_text(anchor['text'], self.args.max_seq_length, self.tokenizer)
        text_retrieval = self.text_retrieval_data[key]
        ps_count = min(len(text_retrieval['sucess']), 16) if self.args.training else len(text_retrieval['sucess'])
        ng_count = min(self.args.retrieval_count//2, len(text_retrieval['fail'])) if self.args.training else len(text_retrieval['fail'])
        image_inputs_tx, text_inputs_tx, text_masks_tx, labels_tx, preds_tx = get_retrievals(self.library_data, self.tokenizer, self.transform, self.node2id,
                                                                                anchor, text_retrieval, 
                                                                                ng_count, ps_count, self.args.max_seq_length, 
                                                                                retrieval_image=False, text_pair=False, only_text=False)
        
        image_inputs = torch.cat((anchor_image_input.unsqueeze(0), image_inputs_im, image_inputs_tx), dim=0)
        text_inputs = torch.cat((anchor_text_input.unsqueeze(0), text_inputs_im, text_inputs_tx), dim=0)
        text_masks = torch.cat((anchor_text_mask.unsqueeze(0), text_masks_im, text_masks_tx), dim=0)
        labels = torch.cat((labels_im, labels_tx), dim=0)
        preds = torch.cat((preds_im, preds_tx), dim=0)
        if self.args.training:
            return image_inputs, text_inputs, text_masks, labels, preds 
        else:
            return image_inputs, text_inputs, text_masks, torch.tensor([self.node2id[anchor['node']]], dtype=torch.int32), preds
        
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, args, transform):
        self.args = args
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.text_retrieval_data = json.load(open(os.path.join("../retrieval/text_retrieval_result", f"{args.data_prefix}text_retrieval_result.json")))
        self.image_retrieval_data = json.load(open(os.path.join("../retrieval/image_retrieval_result", f"{args.data_prefix}image_retrieval_result.json")))

        self.node2id = json.load(open(os.path.join(os.path.join(DIR, "node2id.json"))))
        
        library_text_data = json.load(open(os.path.join(DIR, "library_text.json")))
        library_image_data = json.load(open(os.path.join(DIR, "library_image.json")))
        self.library_data = {}
        for item in library_text_data:
            self.library_data[item['node']] = {
                'name': item['name'],
                'image': [],
                'text': []
            }
        for item in library_text_data:
            self.library_data[item['node']]['text'].append(item['text'])
        for item in library_image_data:
            self.library_data[item['node']]['image'].append(item['image'])
        
        self.data = json.load(open(os.path.join(DIR, f"{args.data_prefix}data.json")))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        anchor = self.data[index]
        key = f"{anchor['node']}-{index}"

        image_retrieval = self.image_retrieval_data[key]
        ps_count = min(len(image_retrieval['sucess']), 16) if self.args.training else len(image_retrieval['sucess'])
        ng_count = min(self.args.retrieval_count//2, len(image_retrieval['fail'])) if self.args.training else len(image_retrieval['fail'])
        _, text_inputs_im, text_masks_im, labels_im, preds_im = get_retrievals(self.library_data, self.tokenizer, self.transform, self.node2id,
                                                                                anchor, image_retrieval, 
                                                                                ng_count, ps_count, self.args.max_seq_length, 
                                                                                retrieval_image=True, text_pair=True, only_text=True)
        text_retrieval = self.text_retrieval_data[key]
        ps_count = min(len(text_retrieval['sucess']), 16) if self.args.training else len(text_retrieval['sucess'])
        ng_count = min(self.args.retrieval_count//2, len(text_retrieval['fail'])) if self.args.training else len(text_retrieval['fail'])
        _, text_inputs_tx, text_masks_tx, labels_tx, preds_tx = get_retrievals(self.library_data, self.tokenizer, self.transform, self.node2id,
                                                                                anchor, text_retrieval, 
                                                                                ng_count, ps_count, self.args.max_seq_length, 
                                                                                retrieval_image=False, text_pair=True, only_text=True)
        text_inputs = torch.cat((text_inputs_im, text_inputs_tx), dim=0)
        text_masks = torch.cat((text_masks_im, text_masks_tx), dim=0)
        labels = torch.cat((labels_im, labels_tx), dim=0)
        preds = torch.cat((preds_im, preds_tx), dim=0)
        if self.args.training:
            return text_inputs, text_masks, labels, preds 
        else:
            return text_inputs, text_masks, torch.tensor([self.node2id[anchor['node']]], dtype=torch.int32), preds
