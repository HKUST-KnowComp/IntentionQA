import pandas as pd
import numpy as np
import networkx as nx
import pickle
import random
import numpy as np
import re
# from collections import deque
import itertools
import json
import argparse
# import csv
# import logging
from tqdm import tqdm
import os
import time
import string
from collections import Counter
from findNode_aser import rules
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

DATA_PATH = "./data"

'''
QA/embeding_dict.pickle: {item_id: [nodes in aser]} => {item_id: [(node, avg(sbert())]}
QA/typicality_filtered.csv: 5514 assertions

'''




def load_pickle(filename):
    with open(filename, 'rb') as file:
        res = pickle.load(file)
    print("Loaded!")
    return res

def store_pickle(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print("Stored!")





def write_data(filename, data):
    with open(filename, 'w') as fout:
        for sample in data:
            fout.write(json.dumps(sample))
            fout.write('\n')

def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def itemId2Events(id, items_info):
    info_rows = items_info[items_info['id'] == id]
    names = list(info_rows['item_category'])
    names_set = []
    for i in names:
        names_set += i
    names_set = list(set(names_set))
    events = rules(names_set)
    return events






def main(nodes):
    typicality_annotated = DATA_PATH + "/typicality_annotated.csv"
    out_file = DATA_PATH + "/QA/typicality_filtered.csv"
    if os.path.exists(out_file):
        print("{} exists!".format(out_file))
        typ_a = pd.read_csv(out_file, header = 0)
        print(typ_a.head())
        print(typ_a.at[0,'item_a_id'])
    else:
        typ_a = pd.read_csv(typicality_annotated, header = 0)
        relevant_cols = ['typicality','assertion','item_a_name','item_b_name','relation','item_a_id','item_b_id','item_a_cate','item_b_cate']
        typ_a = typ_a[relevant_cols]
        typ_a['relation'] = typ_a['relation'].replace('open', 'other')
        typ_a['relation'] = typ_a['relation'].replace('propertOf', 'propertyOf')
        typ_a = typ_a[typ_a['typicality'] >= 0.5]
        out_file = DATA_PATH + "/QA/typicality_filtered.csv"
        typ_a.to_csv(out_file, header=True, index = False)

    aser_node_id_file = DATA_PATH+'/rule_align/t3.txt'
    aser_node_ids = pd.read_csv(aser_node_id_file, sep='\t',header = 0)
    arr = aser_node_ids.to_numpy()
    found_ids = arr[arr[:,1] > 0][:,0]
    items_info = pd.read_json(DATA_PATH+"/items_simplified.jsonl", lines=True)
    res_dict = dict()
    for i in tqdm(range(len(typ_a))):
        item_a_id = typ_a.at[i, 'item_a_id']
        item_b_id = typ_a.at[i, 'item_b_id']
        if item_a_id in found_ids and item_b_id in found_ids:
            if item_a_id not in res_dict:
                events_a = itemId2Events(item_a_id, items_info)
                nodes_a = []
                for e in events_a:
                    if e in nodes:
                        nodes_a.append(e)
                res_dict[item_a_id] = nodes_a
            if item_b_id not in res_dict:
                events_b = itemId2Events(item_b_id, items_info)
                nodes_b = []
                for e in events_b:
                    if e in nodes:
                        nodes_b.append(e)
                res_dict[item_b_id] = nodes_b
        else:
            typ_a.at[i, 'assertion'] = "N/A"
            continue
    typ_a = typ_a[typ_a['assertion'] != "N/A"]
    typ_a.to_csv(out_file, header=True, index = False)
    print("Storing dictionary...")
    file_dict = DATA_PATH+'/QA/embedding_dict.pickle'
    with open(file_dict, 'wb') as file:
        pickle.dump(res_dict, file)
    print("Stored!")

def filterItems():
    embedding_dict = load_pickle("./data/QA/embedding_dict_all-MiniLM-L6-v2.pickle")
    items_relevant = set(list(embedding_dict))
    full_items = read_data("./data/items_simplified_byCate_full.jsonl")
    res_jsonl = []
    for i in tqdm(range(len(full_items))):
        d = full_items[i]
        if d['id'] in items_relevant and "item_category_byCate" in d:
            res_jsonl.append(d)
    write_data("./data/items_simplified_byCate.jsonl", res_jsonl)

def filterAssertions():
    '''filter assertions: keep those with typicality >= 0.6; relation not in 
    {'createdBy','deriveFrom', 'distinctFrom','madeOf','partOf', 'similarTo', 'symbolOf', }
    {'propertyOf', 'relatedTo',  'capableOf', 
     , 'effect', 'cause', 'definedAs', 'usedFor',  'can', 'other',
      'mannerOf', 'isA'}
    {'symbolOf', 'propertyOf', 'relatedTo', 'madeOf', 'capableOf', 'partOf', 'distinctFrom',
     'createdBy', 'effect', 'deriveFrom', 'cause', 'definedAs', 'usedFor', 'similarTo', 'can', 'other',
      'mannerOf', 'isA'}'''
    df = pd.read_csv("./data/QA/typicality_filtered_allrel.csv", header = 0)
    df =  df[~df['relation'].isin(['createdBy','deriveFrom', 'distinctFrom','madeOf','partOf', 'similarTo', 'symbolOf',])]
    df = df[df['typicality']> 0.5]
    print("After filtering by typicality > 0.5 & relation not in [], len(df) = {}".format(len(df)))
    out_file = "./data/QA/typicality_filtered.csv"
    df.to_csv(out_file, header=True, index = False)

if __name__ == '__main__':


    # file_nodes = DATA_PATH+"/nodes_aser2.1_nodefilteronly_norm.pickle"
    # print("nodes set exists! Loading...")
    # with open(file_nodes, 'rb') as file:
    #     nodes = pickle.load(file)
    # print("Loaded!")
    # main(nodes)

    # filterItems()
    filterAssertions()
