import os 
import pdb
import sys 
import json 
import time
import random 

from tqdm import trange

from collections import defaultdict
import numpy as np 

def load_and_sort(name, mode):
    preds = np.load(f"result/{mode}/{name}_preds.npy")
    scores = np.load(f"result/{mode}/{name}_scores.npy")
    labels = np.load(f"result/{mode}/{name}_labels.npy")
    preds, scores = sort_pred(preds, scores)
    return preds, scores, labels

def sort_pred(preds, scores):
    sorted_preds = []
    sorted_scores = []
    for i in range(preds.shape[0]):
        score = scores[i]
        pred = preds[i]
        sorted_index = np.argsort(pred)
        sorted_preds.append(pred[np.flip(sorted_index)])
        sorted_scores.append(score[np.flip(sorted_index)])
    return np.array(sorted_preds), np.array(sorted_scores)

def sort_by_scores(scores, preds, return_index=False):
    results = []
    result_index = []
    for i in range(scores.shape[0]):
        score = scores[i]
        pred = preds[i]
        sorted_index = np.flip(np.argsort(score))
        results.append(pred[sorted_index])
        result_index.append(sorted_index)
    if not return_index:
        return results 
    else:
        return results,  result_index

def compute_topk(preds, labels, k):
    tot = len(labels)
    cor = 0
    for i in range(len(labels)):
        pred = preds[i][:k]
        label = labels[i]
        if label in pred:
            cor += 1 
    return cor / tot

def search(all_scores, preds, labels):
    # preds: pred of the same entity of different models
    # scores: score of the same entitiy of different models
    tot = 10
    expert_cnt = len(all_scores)
    all_prop = np.zeros((expert_cnt))
    best_top1 = 0
    best_prop = []
    for i in trange(tot**expert_cnt):
        step = i
        for j in range(expert_cnt):
            all_prop[j] = (step % tot) / tot  
            step = int(step / tot)
        scores = np.mean(np.expand_dims(np.expand_dims(all_prop, axis=1), axis=2) * all_scores, axis=0)
        results = sort_by_scores(scores, preds)
        top1 = compute_topk(results, labels, 1)
        if top1 > best_top1:
            best_top1 = top1
            best_prop = []
            for p in all_prop:
                best_prop.append(p)
    print(best_top1)
    print(best_prop)
    return best_prop

def no_search(all_scores, preds, labels, all_prop):
    print(all_prop)
    scores = np.mean(np.expand_dims(np.expand_dims(all_prop, axis=1), axis=2) * all_scores, axis=0)
    results, result_index = sort_by_scores(scores, preds, True)
    # json dump
    np.save("result/full_model_pred_index.npy", np.array(result_index))

    top1 = compute_topk(results, labels, 1)
    top3 = compute_topk(results, labels, 3)
    top10 = compute_topk(results, labels, 10)
    top100 = compute_topk(results, labels, 100)
    print("Hits@1: %.3f" % top1)
    print("Hits@3: %.3f" % top3)
    print("Hits@10: %.3f" % top10)
    print("Hits@100: %.3f" % top100)

    return results, labels

def assemble(mode, assem):
    v1_preds, v1_scores, v1_labels = load_and_sort('v1', mode)
    image_preds, image_scores, image_labels = load_and_sort('image', mode)
    biencoder_preds, biencoder_scores, biencoder_labels = load_and_sort('biencoder', mode)
    clip_preds, clip_scores, clip_labels = load_and_sort('clip', mode)

    assert (v1_labels == image_labels).sum() == v1_labels.shape[0]
    assert (v1_labels == biencoder_labels).sum() == v1_labels.shape[0]
    assert (v1_labels == clip_labels).sum() == v1_labels.shape[0]

    assert (v1_preds == image_preds).sum() == v1_preds.shape[0] * v1_preds.shape[1]
    assert (v1_preds == biencoder_preds).sum() == v1_preds.shape[0] * v1_preds.shape[1]
    assert (v1_preds == clip_preds).sum() == v1_preds.shape[0] * v1_preds.shape[1]

    name2scores = {
        'v1': v1_scores,
        'bi': biencoder_scores,
        'image': image_scores,
        'clip': clip_scores
    }

    all_scores = []
    for m in ['v1', 'bi', 'image', 'clip']:
        if m in assem:
            all_scores.append(name2scores[m])

    all_scores = np.stack(all_scores)
    if mode == 'dev':
        search(all_scores, v1_preds, v1_labels)
    elif mode == 'test': 
        prop = np.array([0.2, 0.1, 0.3, 0.9])
        no_search(all_scores, v1_preds, v1_labels, prop)
    # prop = np.array([0.2, 0.1, 0.3, 0.9])
    # no_search(all_scores, v1_preds, v1_labels, prop)

def compute_metric(name, mode):
    # compute metric for single models
    preds = np.load(f"result/{mode}/{name}_preds.npy")
    scores = np.load(f"result/{mode}/{name}_scores.npy")
    labels = np.load(f"result/{mode}/{name}_labels.npy")
    results = []
    for i in range(scores.shape[0]):
        score = scores[i]
        pred = preds[i]
        sorted_index = np.argsort(score)
        results.append(pred[np.flip(sorted_index)])
    # results = preds
    result = f"Hit@1: {compute_topk(results, labels, 1)}\n"
    result += f"Hit@3: {compute_topk(results, labels, 3)}\n"
    result += f"Hit@10: {compute_topk(results, labels, 10)}\n"
    result += f"Hit@100: {compute_topk(results, labels, 100)}\n"
    result += f"Hit@200: {compute_topk(results, labels, 200)}\n"
    print(result)

    f = open("result.log", 'a+')
    cur_time = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()) 
    f.write(cur_time + '\n')
    f.write(name + '\n')
    f.write(result)

def convert_result():
    # load result 
    mode = 'test'
    v1_preds, v1_scores, v1_labels = load_and_sort('v1', mode)
    image_preds, image_scores, image_labels = load_and_sort('image', mode)
    biencoder_preds, biencoder_scores, biencoder_labels = load_and_sort('biencoder', mode)
    clip_preds, clip_scores, clip_labels = load_and_sort('clip', mode)
    # test data 
    ori_test_data = json.load(open("../data/test_data.json"))
    test_data = {}
    for item in ori_test_data:
        test_data[item['node']] = {
            'node': item['node'],
            'text': item['text'],
            'image': item['image'],
            'name': item['name']
        }
    # library
    library_text_data = json.load(open(os.path.join("../data", "library_text.json")))
    library_image_data = json.load(open(os.path.join("../data", "library_image.json")))
    library_data = {}
    for item in library_text_data:
        library_data[item['node']] = {
            'node': item['node'],
            'name': item['name'],
            'image': [],
            'text': []
        }
    for item in library_text_data:
        library_data[item['node']]['text'].append(item['text'])
    for item in library_image_data:
        library_data[item['node']]['image'].append(item['image'])
    
    # node2id 
    node2id = json.load(open("../data/node2id.json"))
    id2node = {}
    for k, v in node2id.items():
        id2node[v] = k
    # result
    pred_index = np.load("result/full_model_pred_index.npy")
    labels = np.load("result/test/v1_labels.npy")
    records = []
    for i in range(len(labels)):
        label = labels[i]
        pred = pred_index[i][0]
        v1, bi, im, cl = v1_scores[i][pred], biencoder_scores[i][pred], image_scores[i][pred], clip_scores[i][pred]
        record = {
            'score': [v1, bi, im, cl],
            'label': test_data[id2node[label]],
            'pred': library_data[id2node[v1_preds[i][pred]]]
            }
        records.append(record)
    json.dump(records, open("result/records.json", 'w'))

def sample_error_case():
    random.seed(42)
    data = json.load(open("result/records.json"))
    errors = []
    for item in data:
        if item['pred']['node'] != item['label']['node']:
            errors.append(item)
    print(len(errors) / len(data))
    sampled_errors = random.sample(errors, k=100)
    for item in sampled_errors:
        image = item['label']['image']
        image_path = "../data/images/" + image[:2] + "/" + image
        if not image.endswith('jpg'):
            image_path += ".jpg"
        os.system(f"cp {image_path} result/images/{image}.jpg")
    json.dump(sampled_errors, open("result/sampled_errors.json", 'w'))

def eval_sparsity(path):
    node2id = json.load(open("../data/node2id.json"))
    data = json.load(open(path))
    data_dict = {}
    for item in data:
        data_dict[node2id[item]] = 1
    
    v1_preds, v1_scores, v1_labels = load_and_sort('v1', "test")
    image_preds, image_scores, image_labels = load_and_sort('image', "test")
    biencoder_preds, biencoder_scores, biencoder_labels = load_and_sort('biencoder', "test")
    clip_preds, clip_scores, clip_labels = load_and_sort('clip', "test")

    assert (v1_labels == image_labels).sum() == v1_labels.shape[0]
    assert (v1_labels == biencoder_labels).sum() == v1_labels.shape[0]
    assert (v1_labels == clip_labels).sum() == v1_labels.shape[0]

    assert (v1_preds == image_preds).sum() == v1_preds.shape[0] * v1_preds.shape[1]
    assert (v1_preds == biencoder_preds).sum() == v1_preds.shape[0] * v1_preds.shape[1]
    assert (v1_preds == clip_preds).sum() == v1_preds.shape[0] * v1_preds.shape[1]

    all_scores = [v1_scores, biencoder_scores, image_scores, clip_scores]
    all_scores = np.stack(all_scores)
    prop = np.array([0.2, 0.1, 0.3, 0.9])
    results, labels = no_search(all_scores, v1_preds, v1_labels, prop)
    # results = sort_by_scores(clip_scores, clip_preds)
    # labels = clip_labels

    top1 = compute_topk(results, labels, 1)
    top3 = compute_topk(results, labels, 3)
    top10 = compute_topk(results, labels, 10)
    print('-'*10)
    print("Hits@1: %.3f" % top1)
    print("Hits@3: %.3f" % top3)
    print("Hits@10: %.3f" % top10)

    new_results, new_labels = [], [] 
    for i in range(len(labels)):
        if labels[i] in data_dict:
            new_results.append(results[i])
            new_labels.append(labels[i])
    print('-'*10)
    print(len(new_labels))
    print(len(new_labels) / len(labels))
    top1 = compute_topk(new_results, new_labels, 1)
    top3 = compute_topk(new_results, new_labels, 3)
    top10 = compute_topk(new_results, new_labels, 10)
    print('-'*10)
    print("Hits@1: %.3f" % top1)
    print("Hits@3: %.3f" % top3)
    print("Hits@10: %.3f" % top10)
    

if __name__ == "__main__":
    # name = sys.argv[1]
    # compute_metric(name)
    mode = sys.argv[1]
    assem = sys.argv[2]
    assemble(mode, assem)
    # convert_result()
    # sample_error_case()
    # eval_sparsity("result/image_sparsity_entities.json")