import torch 
import pdb
import sys
sys.path.append('../')
import numpy as np 
from torch import einsum
import torch.nn as nn 
import torch.nn.functional as F

import clip 

from backbone.resnet import *
from transformers import BertModel, BertTokenizer, LxmertModel

inf = 100000000000

class FocalLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        logits and label have same shape, and label data type is long
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        Usage is like this:
            >>> criteria = FocalLoss()
            >>> logits = torch.randn(8, 19, 384, 384)# nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nchw, int64_t
            >>> loss = criteria(logits, lbs)
        '''

        # compute loss
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

class MatchingModel(nn.Module):
    def __init__(self, args):
        super(MatchingModel, self).__init__()
        self.args = args 
        
        # set encoder
        if self.args.encoder == 'br':
            self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
            self.classifier = nn.Linear(768, 1)
            self.image_encoder = resnet152(pretrained=True)
        elif self.args.encoder == 'clip':
            self.clip, _ = clip.load("ViT-B/16", jit=False)

        # set loss fn
        if self.args.loss == 'focal':
            print("Let's use focal loss.")
            self.loss_fn = FocalLoss(alpha=0.96, gamma=2)
        elif self.args.loss == 'bce':
            print("Let's use bce loss.")
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, gloss=None, image=None, gloss_attention_mask=None, labels=None, to_squeeze=True):
        if self.args.model == 'text_cross_encoder':
            return self.text_cross_encoder(gloss, gloss_attention_mask, labels, to_squeeze)
        elif self.args.model == 'text_bi_encoder':
            return self.text_bi_encoder(gloss, gloss_attention_mask, labels, to_squeeze)
        elif self.args.model == 'image_bi_encoder':
            return self.image_bi_encoder(gloss, image, gloss_attention_mask, labels, to_squeeze)
        elif self.args.model == 'clip':
            return self.clip_encoder(gloss, image, gloss_attention_mask, labels, to_squeeze)
        else:
            raise Exception("No such model")
    
    def text_cross_encoder(self, gloss, gloss_attention_mask, labels=None, to_squeeze=True):
        if to_squeeze:
            gloss = gloss.squeeze(0)
            gloss_attention_mask = gloss_attention_mask.squeeze(0)
            labels = labels.squeeze(0) if labels is not None else None
        # gloss feature
        gloss_outputs = self.text_encoder(input_ids=gloss, attention_mask=gloss_attention_mask)
        gloss_feature = gloss_outputs[0][:, 0, :]          # [batch, dim1]
        # loss
        scores = self.classifier(gloss_feature).squeeze(1)
        if self.training:
            loss = self.loss_fn(scores, labels)
            return loss, torch.sigmoid(scores)
        else:
            return torch.sigmoid(scores)
    
    def text_bi_encoder(self, gloss, gloss_attention_mask, labels=None, to_squeeze=True):
        if to_squeeze:
            gloss = gloss.squeeze(0)
            gloss_attention_mask = gloss_attention_mask.squeeze(0)
            labels = labels.squeeze(0) if labels is not None else None
        gloss_outputs = self.text_encoder(input_ids=gloss, attention_mask=gloss_attention_mask)
        gloss_feature = gloss_outputs[0][:, 0, :]          
        
        text_features = gloss_feature / gloss_feature.norm(dim=-1, keepdim=True)
        anchor_text = text_features[0:1]
        text_features = text_features[1:]
        # loss 
        scores = torch.matmul(anchor_text, text_features.t()).squeeze(0)
        if self.training:
            loss = self.loss_fn(scores, labels)
            return loss, torch.sigmoid(scores)
        else:
            return torch.sigmoid(scores)
    
    def image_bi_encoder(self, gloss, image, gloss_attention_mask, labels=None, to_squeeze=True):
        if to_squeeze:
            gloss = gloss.squeeze(0)
            image = image.squeeze(0)
            gloss_attention_mask = gloss_attention_mask.squeeze(0)
            labels = labels.squeeze(0) if labels is not None else None
        image_outputs = self.image_encoder(image)
        
        # v_feature = image_outputs / image_outputs.norm(dim=-1, keepdim=True)
        # anchor_v_feature = v_feature[0:1]
        # ins_v_feature = v_feature[1:]
        # scores = torch.matmul(anchor_v_feature, ins_v_feature.t()).squeeze(0)

        anchor_v_feature = image_outputs[0:1]
        ins_v_feature = image_outputs[1:]
        scores = F.cosine_similarity(anchor_v_feature, ins_v_feature, dim=-1, eps=1e-5)
        if self.training:
            loss = self.loss_fn(scores, labels)
            return loss, torch.sigmoid(scores)
        else:
            return torch.sigmoid(scores)
        
    def clip_encoder(self, gloss, image, gloss_attention_mask, labels=None, to_squeeze=True):
        if to_squeeze:
            gloss = gloss.squeeze(0)
            image = image.squeeze(0)
            labels = labels.squeeze(0) if labels is not None else None
        image_features = self.clip.encode_image(image)
        text_features = self.clip.encode_text(gloss)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        anchor_image = image_features[0:1]
        anchor_text = text_features[0:1]

        image_features = image_features[1:]
        text_features = text_features[1:]

        # cosine similarity as logits
        logits_per_image = (anchor_image @ text_features.t()).squeeze(0)
        logits_per_text = (anchor_text @ image_features.t()).squeeze(0)

        logits = torch.sigmoid(logits_per_image) + torch.sigmoid(logits_per_text)
        # logits = torch.sigmoid(logits_per_image)
        if self.training:
            if self.args.loss == 'bce':
                # Rebalance positive loss and negative loss
                i_loss = self.loss_fn(logits_per_image, labels.to(torch.float32)) 
                t_loss= self.loss_fn(logits_per_text, labels.to(torch.float32))
                i_loss += t_loss
                p_loss = (i_loss * labels).sum() / (labels.sum() + 1e-5)
                neg_labels = 1 - labels
                n_loss = (i_loss * neg_labels).sum() / (neg_labels.sum() + 1e-5)
                loss = (p_loss + n_loss) / 2
            elif self.args.loss == 'focal':
                i_loss = self.loss_fn(logits_per_image + logits_per_text, labels)
                # t_loss = self.loss_fn(logits_per_text, labels)
                # loss = (i_loss + t_loss) / 2
                loss = i_loss
            return loss, logits 
        else:
            return torch.sigmoid(logits_per_image), torch.sigmoid(logits_per_text)



        

