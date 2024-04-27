import pandas as pd
import pickle
# from collections import deque
import json
import argparse
# import csv
# import logging
from tqdm import tqdm
import os
import time
import string
from sentence_transformers import SentenceTransformer, util
import torch

def load_pickle(filename):
    with open(filename, 'rb') as file:
        res = pickle.load(file)
    print("Loaded!")
    return res

def store_pickle(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print("Stored!")

def compute_assertionSimilarity(A,B, C,D, itemSimilarity):
    similarities = torch.stack([itemSimilarity[(A,C)][0,0], itemSimilarity[(B,C)][0,0], itemSimilarity[(A,D)][0,0], itemSimilarity[(B,D)][0,0]])
    similarity = torch.min(similarities)
    return similarity



def hasOverlapConcept(A,B,C,D, itemsInfo):
    # pd.read_json("./data/items_simplified_byCate.jsonl",lines = True)
    A_info = list(itemsInfo[itemsInfo['id'] == A]["item_category"])[0]
    B_info = list(itemsInfo[itemsInfo['id'] == B]["item_category"])[0]
    C_info = list(itemsInfo[itemsInfo['id'] == C]["item_category"])[0]
    D_info = list(itemsInfo[itemsInfo['id'] == D]["item_category"])[0]
    AB = A_info + B_info
    AB = [word for string in AB for word in string.split(' ')]
    AB = set(AB)
    CD = C_info + D_info
    CD = [word for string in CD for word in string.split(' ')]
    CD = set(CD)
    if AB.intersection(CD):
        return True
    else:
        return False

def hasOverlapConcept_AB(A,B, itemsInfo):
    # pd.read_json("./data/items_simplified_byCate.jsonl",lines = True)
    A_info = list(itemsInfo[itemsInfo['id'] == A]["item_category"])[0]
    B_info = list(itemsInfo[itemsInfo['id'] == B]["item_category"])[0]

    A_words = [word for string in A_info for word in string.split(' ')]
    A_words = set(A_words)
    B_words = [word for string in B_info for word in string.split(' ')]
    B_words = set(B_words)
    if A_words.intersection(B_words):
        return True
    else:
        return False

def get_itemSimilarity(path_to_embedding_dict, out_dir):
    embedding_dict = load_pickle(path_to_embedding_dict)
    keys = list(embedding_dict)
    itemSimilarity_dict = dict()
    for i in tqdm(range(len(keys))):
        k1 = keys[i]
        for j in range(i+1, len(keys)):
            k2 = keys[j]
            cur_sim = util.cos_sim(embedding_dict[k1], embedding_dict[k2])
            # if k1 < k2:
            itemSimilarity_dict[(k1,k2)] = cur_sim
            # else:
            itemSimilarity_dict[(k2,k1)] = cur_sim
    store_pickle(out_dir, itemSimilarity_dict)

def update_itemSimilarity():
    print("Loading itemSimilarity dictionary...")
    start_time = time.time()
    dic = load_pickle("./data/QA/itemSimilarity.pickle")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("itemSimilarity loaded! Elapsed time:", elapsed_time/60, "minites")
    items_Info = pd.read_json("./data/items_simplified_byCate.jsonl", lines = True)
    keys = list(dic)
    for i in tqdm(range(len(keys))):
        k = keys[i]
        A = k[0]
        B = k[1]
        TF = hasOverlapConcept_AB(A,B,items_Info)
        dic[k] = (dic[k], TF)
    store_pickle("./data/QA/itemSimilarity_updated.pickle", dic)


def get_assertionSimilarity(itemSimilarity, itemsInfo, out_dir,path_to_assertionFile):
    assertions = pd.read_csv(path_to_assertionFile)
    relations = list(set(assertions['relation']))
    # for rel in relations:
    #     print("Computing assertionSim: {}".format(rel))
    #     out_file = out_dir + "/sim_"+rel+".pickle"
    #     cur_out_sim = dict()
    #     sub_assertions = assertions[assertions['relation'] == rel]
    #     cur_ids = list(sub_assertions['id'])
    #     for i in tqdm(range(len(cur_ids))):
    #         k1 = cur_ids[i]
    #         a1 = sub_assertions[sub_assertions['id'] == k1]
    #         itemId_A = list(a1['item_a_id'])[0]
    #         itemId_B = list(a1['item_b_id'])[0]
    #         for j in range(i+1, len(cur_ids)):
    #             k2 = cur_ids[j]
    #             a2 = sub_assertions[sub_assertions['id'] == k2]
    #             itemId_C = list(a2['item_a_id'])[0]
    #             itemId_D = list(a2['item_b_id'])[0]
    #             if itemId_C == itemId_A or itemId_C == itemId_B or itemId_D == itemId_A or itemId_D == itemId_B:
    #                 cur_out_sim[(k1, k2)] = (torch.tensor(1), True)
    #                 cur_out_sim[(k2, k1)] = (torch.tensor(1), True)
    #                 continue
    #             k1k2_similarity = compute_assertionSimilarity(itemId_A, itemId_B, itemId_C, itemId_D, itemSimilarity)
    #             k1k2_overlap = hasOverlapConcept(itemId_A, itemId_B, itemId_C, itemId_D, itemsInfo)
    #             cur_out_sim[(k1, k2)] = (k1k2_similarity, k1k2_overlap)
    #             cur_out_sim[(k2, k1)] = (k1k2_similarity, k1k2_overlap)
    #     store_pickle(out_file, cur_out_sim)

    print("Computing assertionSim: All")
    out_file = out_dir + "/sim_all.pickle"
    cur_out_sim = dict()
    sub_assertions = assertions
    cur_ids = list(sub_assertions['id'])
    for i in tqdm(range(len(cur_ids))):
        k1 = cur_ids[i]
        a1 = sub_assertions[sub_assertions['id'] == k1]
        itemId_A = list(a1['item_a_id'])[0]
        itemId_B = list(a1['item_b_id'])[0]
        for j in range(i+1, len(cur_ids)):
            k2 = cur_ids[j]
            a2 = sub_assertions[sub_assertions['id'] == k2]
            itemId_C = list(a2['item_a_id'])[0]
            itemId_D = list(a2['item_b_id'])[0]
            if itemId_C == itemId_A or itemId_C == itemId_B or itemId_D == itemId_A or itemId_D == itemId_B:
                cur_out_sim[(k1, k2)] = (torch.tensor(1), True)
                cur_out_sim[(k2, k1)] = (torch.tensor(1), True)
                continue
            k1k2_similarity = compute_assertionSimilarity(itemId_A, itemId_B, itemId_C, itemId_D, itemSimilarity)
            k1k2_overlap = hasOverlapConcept(itemId_A, itemId_B, itemId_C, itemId_D, itemsInfo)
            cur_out_sim[(k1, k2)] = (k1k2_similarity, k1k2_overlap)
            cur_out_sim[(k2, k1)] = (k1k2_similarity, k1k2_overlap)

    store_pickle(out_file, cur_out_sim)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default = 'all-MiniLM-L6-v2', type = str, required = False, )
    parser.add_argument("--embeddingDict", default = "./data/QA/embedding_dict_all-MiniLM-L6-v2.pickle", type = str, required = False)
    parser.add_argument("--itemSimilarity", default = "./data/QA/itemSimilarity.pickle", type = str, required = False)
    parser.add_argument("--itemsInfo", default = "./data/items_simplified_byCate.jsonl", type = str, required = False)
    parser.add_argument("--out_dir", default = "./data/QA/assertionSimilarity", type = str, required = False)
    parser.add_argument("--path_to_assertionFile", default = "./data/QA/typicality_filtered.csv", type = str, required = False)
    
    # parser.add_argument("--version", default = None, type = int, required = True)
#######################
    # args = parser.parse_args()
    # if os.path.exists(args.itemSimilarity):
    #     print("Loading itemSimilarity dict...")
    #     itemSimilarity_dict = load_pickle(args.itemSimilarity)
    #     print("itemSimilarity_dict loaded!")
    # else:
    #     itemSimilarity_dict = get_itemSimilarity(args.embeddingDict, args.itemSimilarity)
    #     print("itemSimilarity_dict computed!")
    
    # itemsInfo = pd.read_json(args.itemsInfo, lines = True)
    # get_assertionSimilarity(itemSimilarity_dict, itemsInfo, args.out_dir, args.path_to_assertionFile)
#######################

    update_itemSimilarity()



        