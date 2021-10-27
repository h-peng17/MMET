import os 
import pdb 
import json 
import random
from collections import defaultdict, deque
from tqdm import tqdm

import torch 
import torch.nn as nn
from PIL import Image
import numpy as np
import numpy
from transformers import BertTokenizer

from elasticsearch import Elasticsearch, helpers
from nltk.tokenize import word_tokenize


class TextRecallModel:
    def __init__(self, args):
        self.args = args 
        self.es = Elasticsearch(timeout=5000)
        
    def construct_library(self, data):
        # 创建mapping
        mapping = {      
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text"
                    },
                    "node": {
                        "type": "text"
                    }
                }
            }
        }
        def insert_data(data): # data generator, 返回数据
            for item in tqdm(data):
                body = {'text': item['text'], 'node': item['node']}
                yield {'_index': 'doc', '_source': body}

        self.es.indices.create(index='doc', body=mapping) # 创建索引
        deque(helpers.parallel_bulk(self.es, insert_data(data)), maxlen=0) # 多线程插入
    
    def query_library(self, item):
        body = {
            'query':{
                'match':{
                    'text': item
                }
            },
            'size': 100
        }
        _result = self.es.search(index='doc', body=body)
        result = []
        score = []
        for item in _result['hits']['hits']:
            result.append(item["_source"])
            score.append(item['_score'])

        return result, score
    
    def clear_library(self):
        try:
            res = self.es.indices.delete(index='doc')
            print(res)
        except:
            print("Not index doc")






