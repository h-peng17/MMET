import os
from os import terminal_size 
import pdb 
import sys 
import json
import random
from PIL import Image
import numpy as np 
from tqdm import tqdm
from collections import defaultdict

DIR = "../data"

def filter_node():
    """We will filter nodes:
    - less than 3 glosses
    - less than 3 images
    """
    nodes = json.load(open(os.path.join(DIR, "nodes.json")))
    filter_nodes = {}
    for key, node in nodes.items():
        if len(node['gl']) < 3 or len(node['ims']) < 3:
            continue
        filter_nodes[key] = node
    print(len(filter_nodes))
    json.dump(filter_nodes, open(os.path.join(DIR, "filter_nodes.json"), 'w'))

def gen_node_to_idx():
    nodes = json.load(open(os.path.join(DIR, "filter_nodes.json")))
    node2id = {}
    for key in nodes.keys():
        node2id[key] = len(node2id)
    json.dump(node2id, open(os.path.join(DIR, "node2id.json"), 'w'))

def split_data():
    nodes = json.load(open(os.path.join(DIR, "filter_nodes.json")))
    train_data = []
    extra_data = {}
    images_to_exclude = dict()
    texts_to_exclude = dict()
    for key, node in tqdm(nodes.items()):
        name = ' '.join(node['ms'].split(':')[-1].split('_')).lower()
        # train
        text = node['gl'][0]
        image = node['ims'][0]   
        train_data.append({
            'text': text,
            'node': key,
            'image': image,
            'name': name,
        })
        images_to_exclude[image] = 0
        texts_to_exclude[text] = 0
        # library
        images = []
        for image in node['ims']:
            if images_to_exclude.get(image, -1) == -1:
                images.append(image)
        if len(images) == 0:
            train_data = train_data[:-1]
            continue
        texts = []
        for text in node['gl']:
            if texts_to_exclude.get(text, -1) == -1:
                texts.append(text)
        extra_data[key] = {
            'ims': images,
            'gl': texts,
            'name': name, 
        }
    test_data = []
    test_keys = random.sample(list(extra_data.keys()), k=2000)
    for key in test_keys:
        images = extra_data[key]['ims']
        image = None
        for im in images:
            if images_to_exclude.get(im, -1) == -1:
                image = im 
                break
        if image is None:
            continue

        texts = extra_data[key]['gl']
        text = None 
        for tx in texts:
            if texts_to_exclude.get(tx, -1) == -1:
                text = tx 
                break
        if text is None:
            continue
        images_to_exclude[image] = 0
        texts_to_exclude[text] = 0
        test_data.append({
            'text': text,
            'node': key,
            'image': image,
            'name': extra_data[key]['name']
        })
    
    dev_data = []
    dev_keys = random.sample(list(extra_data.keys()), k=2000)
    for key in dev_keys:
        images = extra_data[key]['ims']
        image = None
        for im in images:
            if images_to_exclude.get(im, -1) == -1:
                image = im 
                break
        if image is None:
            continue

        texts = extra_data[key]['gl']
        text = None 
        for tx in texts:
            if texts_to_exclude.get(tx, -1) == -1:
                text = tx 
                break
        if text is None:
            continue
        images_to_exclude[image] = 0
        texts_to_exclude[text] = 0
        dev_data.append({
            'text': text,
            'node': key,
            'image': image,
            'name': extra_data[key]['name']
        })

    library_image = []
    library_text = []
    filtered_keys = []
    for key in extra_data.keys():
        images = extra_data[key]['ims']
        texts = extra_data[key]['gl']
        name = extra_data[key]['name']
        filtered_images = []
        filtered_texts = []
        for image in images:
            if images_to_exclude.get(image, -1) == -1:
                filtered_images.append(image)
        for text in texts:
            if texts_to_exclude.get(text, -1) == -1:
                filtered_texts.append(text)
        if len(filtered_texts) == 0 or len(filtered_images) == 0:
            filtered_keys.append(key)
            continue
        for image in filtered_images:
            library_image.append({
                'image': image,
                'node': key,
                'name': name
            })
        for text in filtered_texts:
            library_text.append({
                'text': text,
                'node':key,
                'name':name
            })

    filtered_train = [item for item in train_data if item['node'] not in filtered_keys]
    filtered_dev = [item for item in dev_data if item['node'] not in filtered_keys]
    filtered_test = [item for item in test_data if item['node'] not in filtered_keys]

    print(f"Train: {len(filtered_train)}, Dev: {len(filtered_dev)}, Test: {len(filtered_test)}, library_text: {len(library_text)}, library_image: {len(library_image)}")
    json.dump(filtered_train, open(os.path.join(DIR, "train_data.json"), 'w'))
    json.dump(filtered_dev, open(os.path.join(DIR, "dev_data.json"), 'w'))
    json.dump(filtered_test, open(os.path.join(DIR, "test_data.json"), 'w'))
    json.dump(library_text, open(os.path.join(DIR, "library_text.json"), 'w'))
    json.dump(library_image, open(os.path.join(DIR, "library_image.json"), 'w'))


if __name__ == "__main__":
    random.seed()
    filter_node()
    gen_node_to_idx()
    split_data()


    




