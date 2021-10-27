import os 
import gc
import pdb 
import copy
import json
from pdb import set_trace 
import random
from collections import defaultdict
from random import sample, uniform

import clip

import torch 
import numpy as np
from PIL import Image
from transformers import BertTokenizer

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
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

def get_retrievals(args, library_data, tokenizer, transform, node2id,
                query, query_retrieval, 
                ng_count, ps_count=0, max_length=64, 
                retrieval_image=True, text_pair=False, only_text=False):
    gt_node, gt_text = query['node'], query['text']
    sucess_count, fail_count = len(query_retrieval['sucess']), len(query_retrieval['fail'])
    sample_count = sucess_count + fail_count
    retrievals = torch.zeros(sample_count, dtype=torch.int32)
    retrievals[:sucess_count] = torch.tensor(list(range(sucess_count)))
    retrievals[sucess_count:] = torch.tensor(list(range(fail_count)))    

    expert_count = args.expert_count
    text_inputs = torch.zeros((sample_count, expert_count, max_length), dtype=torch.long)
    text_masks = torch.zeros((sample_count, expert_count, max_length), dtype=torch.float32)
    expert_masks = torch.zeros((sample_count, expert_count), dtype=torch.int32)
    labels = torch.zeros((sample_count), dtype=torch.long)
    preds = torch.zeros((sample_count), dtype=torch.long)

    for i, idx in enumerate(retrievals):
        if i < sucess_count:
            node, retrieval, _ = query_retrieval['sucess'][idx]
        else:
            node, retrieval, _  = query_retrieval['fail'][idx]

        if retrieval_image:
            all_texts = library_data[node]['text'][:expert_count]
            for j, text in enumerate(all_texts):
                text_input, text_mask = get_per_text(text, max_length, tokenizer) if not text_pair else \
                                get_per_text_pair(gt_text, text, max_length, tokenizer)
                text_inputs[i][j] = text_input
                text_masks[i][j] = text_mask
                expert_masks[i][j] = 1
        else:
            text = retrieval 
            text_input, text_mask = get_per_text(text, max_length, tokenizer) if not text_pair else \
                                get_per_text_pair(gt_text, text, max_length, tokenizer)
            text_inputs[i][0] = text_input
            text_masks[i][0] = text_mask
            expert_masks[i][0] = 1
            all_texts = copy.deepcopy(library_data[node]['text'])
            all_texts.remove(retrieval)
            all_texts = all_texts[:expert_count-1]
            for j, text in enumerate(all_texts):
                text_input, text_mask = get_per_text(text, max_length, tokenizer) if not text_pair else \
                                get_per_text_pair(gt_text, text, max_length, tokenizer)
                text_inputs[i][j+1] = text_input
                text_masks[i][j+1] = text_mask
                expert_masks[i][j+1] = 1
        labels[i] = 1 if gt_node == node else 0
        preds[i] = node2id[node]
        assert (labels[i] == 1 and i < ps_count) or (labels[i] == 0 and i >= ps_count)
    if only_text:
        image_inputs = None
        image_expert_masks = None
    else:
        image_expert_count = args.image_expert_count
        image_expert_masks = torch.zeros((sample_count, image_expert_count), dtype=torch.int32)
        image_inputs = torch.zeros((sample_count, image_expert_count, 3, 224, 224), dtype=torch.float32)
        for i, idx in enumerate(retrievals):
            if i < ps_count:
                node, retrieval, _ = query_retrieval['sucess'][idx]
            else:
                node, retrieval, _  = query_retrieval['fail'][idx]
            if retrieval_image:
                image = retrieval 
                image_input = get_per_image(image, transform)
                image_inputs[i][0] = image_input
                image_expert_masks[i][0] = 1

                all_images = copy.deepcopy(library_data[node]['image'])
                all_images.remove(retrieval)
                all_images = all_images[:image_expert_count-1]
                for j, image in enumerate(all_images):
                    image_input = get_per_image(image, transform)
                    image_inputs[i][j+1] = image_input
                    image_expert_masks[i][j+1] = 1
            else:
                all_images = library_data[node]['image'][:image_expert_count]
                for j, image in enumerate(all_images):
                    image_input = get_per_image(image, transform)
                    image_inputs[i][j] = image_input
                    image_expert_masks[i][j] = 1
        image_inputs = image_inputs.reshape(-1, 3, 224, 224)
    return image_inputs, text_inputs.reshape(-1, max_length), text_masks.reshape(-1, max_length), labels, preds, expert_masks, image_expert_masks


class CPDataset(torch.utils.data.Dataset):
    def __init__(self, args, transform):
        self.args = args
        if args.model == 'clip': 
            print("Let's use CLIP config...")
            self.tokenizer = clip.tokenize
            _, preprocess = clip.load("ViT-B/32", jit=False)
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
        image_inputs_im, text_inputs_im, text_masks_im, labels_im, preds_im, expert_masks_im, image_expert_masks_im = get_retrievals(self.args, self.library_data, self.tokenizer, self.transform, self.node2id,
                                                                                anchor, image_retrieval, 
                                                                                ng_count, ps_count, self.args.max_seq_length, 
                                                                                retrieval_image=True, text_pair=False, only_text=False)
        anchor_image_input = get_per_image(anchor['image'], self.transform)
        anchor_text_input, anchor_text_mask = get_per_text(anchor['text'], self.args.max_seq_length, self.tokenizer)
        text_retrieval = self.text_retrieval_data[key]
        ps_count = min(len(text_retrieval['sucess']), 16) if self.args.training else len(text_retrieval['sucess'])
        ng_count = min(self.args.retrieval_count//2, len(text_retrieval['fail'])) if self.args.training else len(text_retrieval['fail'])
        image_inputs_tx, text_inputs_tx, text_masks_tx, labels_tx, preds_tx, expert_masks_tx, image_expert_masks_tx = get_retrievals(self.args, self.library_data, self.tokenizer, self.transform, self.node2id,
                                                                                anchor, text_retrieval, 
                                                                                ng_count, ps_count, self.args.max_seq_length, 
                                                                                retrieval_image=False, text_pair=False, only_text=False)
        
        image_inputs = torch.cat((anchor_image_input.unsqueeze(0), image_inputs_im, image_inputs_tx), dim=0)
        text_inputs = torch.cat((anchor_text_input.unsqueeze(0), text_inputs_im, text_inputs_tx), dim=0)
        text_masks = torch.cat((anchor_text_mask.unsqueeze(0), text_masks_im, text_masks_tx), dim=0)
        labels = torch.cat((labels_im, labels_tx), dim=0)
        preds = torch.cat((preds_im, preds_tx), dim=0)
        expert_masks = torch.cat((expert_masks_im, expert_masks_tx), dim=0)
        image_expert_masks = torch.cat((image_expert_masks_im, image_expert_masks_tx), dim=0)

        if self.args.training:
            return image_inputs, text_inputs, text_masks, labels, preds, expert_masks, image_expert_masks
        else:
            return image_inputs, text_inputs, text_masks, torch.tensor([self.node2id[anchor['node']]], dtype=torch.int32), preds, expert_masks, image_expert_masks
        
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
        _, text_inputs_im, text_masks_im, labels_im, preds_im, expert_masks_im, _ = get_retrievals(self.args, self.library_data, self.tokenizer, self.transform, self.node2id,
                                                                                anchor, image_retrieval, 
                                                                                ng_count, ps_count, self.args.max_seq_length, 
                                                                                retrieval_image=True, text_pair=True, only_text=True)
        text_retrieval = self.text_retrieval_data[key]
        ps_count = min(len(text_retrieval['sucess']), 16) if self.args.training else len(text_retrieval['sucess'])
        ng_count = min(self.args.retrieval_count//2, len(text_retrieval['fail'])) if self.args.training else len(text_retrieval['fail'])
        _, text_inputs_tx, text_masks_tx, labels_tx, preds_tx, expert_masks_tx, _ = get_retrievals(self.args, self.library_data, self.tokenizer, self.transform, self.node2id,
                                                                                anchor, text_retrieval, 
                                                                                ng_count, ps_count, self.args.max_seq_length, 
                                                                                retrieval_image=False, text_pair=True, only_text=True)
        text_inputs = torch.cat((text_inputs_im, text_inputs_tx), dim=0)
        text_masks = torch.cat((text_masks_im, text_masks_tx), dim=0)
        labels = torch.cat((labels_im, labels_tx), dim=0)
        preds = torch.cat((preds_im, preds_tx), dim=0)
        expert_masks = torch.cat((expert_masks_im, expert_masks_tx), dim=0)
        if self.args.training:
            return text_inputs, text_masks, labels, preds, expert_masks 
        else:
            return text_inputs, text_masks, torch.tensor([self.node2id[anchor['node']]], dtype=torch.int32), preds, expert_masks

