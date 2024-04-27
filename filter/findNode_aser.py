# map person buy/shop/... PRODUCT to nodes in ASER
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import random
import numpy as np
import re
# from collections import deque
import itertools
import tqdm
import json
import argparse
# import csv
# import logging
from tqdm import tqdm
import os
import time
import string
from collections import Counter
# from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

DATA_PATH = "./data"

# stops = set(stopwords.words('english'))
puncs = list(string.punctuation)

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



def normalize_name(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(puncs)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    def remove_parentheses(text):
        pattern = r"\([^()]*\)"
        result = re.sub(pattern, "", text)
        return result

    return white_space_fix(remove_articles(remove_punc(lower(remove_parentheses(s)))))


def normalize_items(l):
    lemmatizer = WordNetLemmatizer()
    l_res = []
    for i in l:
        normalized_i = normalize_name(i)
        l_res.append(normalized_i)
        normalized_tokens = normalized_i.split()
        normalized_tokens[-1] = lemmatizer.lemmatize(normalized_tokens[-1])
        singular_i = ' '.join(normalized_tokens)
        if singular_i != normalized_i:
            l_res.append(singular_i)
    return l_res

def normalize_all_items():
    raw_file = "./data/items_simplified_raw.jsonl"
    out_file = "./data/items_simplified.jsonl"
    data = read_data(raw_file)
    res = []
    for i in tqdm(range(len(data))):
        d = data[i]
        if d['n_prompt'] == -1:
            continue
        # print("hi")
        d['item_category'] = normalize_items(d['item_category'])
        res.append(d)
        # print(d)
        # raise
    print("writing")
    write_data(out_file, res)


def rules(l):
    sub = ['PersonX','PersonY','PeopleX','PeopleY']
    verb = ['buy','shop','purchase','get','obtain','have']
    verb += ['buys','shops','purchases','gets','obtains','has']
    verb += ['bought','shopped','purchased','got','obtained','had']
    article = ['a','an','the','1','2']
    def rule_1(verb):
        verb += ['has bought','has shopped','has purchased','has got','has obtained','has had'] #version 1
        verb += ['have bought','have shopped','have purchased','have got','have obtained','have had'] #version 1
        verb += ['had bought','had shopped','had purchased','had got','had obtained','had had']
        return verb

    
    verb = rule_1(verb)
    cartesian_product = list(itertools.product(sub, verb, l))
    cartesian_product += list(itertools.product(verb, l)) # version 1
    cartesian_product += list(itertools.product(verb, article, l)) # version 2
    cartesian_product += list(itertools.product(sub, verb, article, l)) # version 2
    events = []
    for p in cartesian_product:
        events.append(' '.join(p))
    return events





def findNode(args, nodes):
    items_file = DATA_PATH+"/items_simplified.jsonl"
    data = read_data(items_file)
    ids = []
    ls_found = []
    for i in tqdm(range(len(data))):
        simplified_names = data[i]['item_category']
        events = rules(simplified_names)
        ids.append(data[i]['id'])
        n_found = 0
        for e in events:
            if e in nodes:
                n_found += 1
        ls_found.append(n_found)
    
    out_file = os.path.join(DATA_PATH+"/rule_align", "t"+str(args.version)+".txt")
    scores = {'id':ids, 'n_found':ls_found}
    df = pd.DataFrame(scores)
    df.to_csv(out_file, sep="\t", header=True, index = False)

    filename = DATA_PATH+"/rule_align/trial_summary.txt"
    avg_found = sum(ls_found)/len(ids)
    positive_count = len([num for num in ls_found if num > 0])
    print("Writing summary...")
    QA_count = count_QAs(args)
    with open(filename, 'a') as file:
        # Write the content to the file
        file.write("{}\n".format(out_file[44:]))
        file.write("avg_found: {}\n".format(avg_found))
        file.write("items_found (#[n_found > 0]):{}\n".format(positive_count))
        file.write("len(data): {}\n".format(len(ids)))
        file.write("Total number of assertions (typicality>=0.5) where both ids have correlated nodes in ASER = {}.\n\n".format(QA_count))
    return

def count_QAs(args):
    aser_node_id_file = DATA_PATH+'/rule_align/t'+str(args.version)+'.txt'
    aser_node_ids = pd.read_csv(aser_node_id_file, sep='\t',header = 0)
    arr = aser_node_ids.to_numpy()
    found_ids = arr[arr[:,1] > 0][:,0]
    assertions = pd.read_csv(DATA_PATH+'/typicality_annotated.csv',header=0)
    assertions = assertions[assertions['typicality'] >= 0.5]
    ids_1 = list(assertions['item_a_id'])
    ids_2 = list(assertions['item_b_id'])
    count_by_relation = dict()
    relations = list(assertions['relation'])
    counter_all = Counter(relations)
    print(counter_all)
    QA_count = 0
    for i in tqdm(range(len(ids_1))):
        if ids_1[i] in found_ids and ids_2[i] in found_ids:
            QA_count += 1
            if relations[i] not in count_by_relation:
                count_by_relation[relations[i]] = 1
            else:
                count_by_relation[relations[i]] += 1
    print(count_by_relation)
    print("Total number of assertions (typicality>=0.5) where both ids have correlated nodes in ASER = {}.".format(QA_count))
    return QA_count 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--f", default = None, type = str, required = True, choices=['findNode', 'countQA'])
    parser.add_argument("--version", default = None, type = int, required = True)

    args = parser.parse_args()
    if args.f == 'findNode':
        file_nodes = DATA_PATH+"/nodes_aser2.1_nodefilteronly_norm.pickle"
        print(file_nodes)
        if os.path.exists(file_nodes):
            print("nodes set exists! Loading...")
            with open(file_nodes, 'rb') as file:
                nodes = pickle.load(file)
            print("Loaded!")

        else:
            print("reading ASER...")
            T1 = time.perf_counter()
            G = nx.read_gpickle("./data/ASER2-1_Norm/G_aser2.1_nodefilteronly_norm.pickle")
            nodes = set(G.nodes())
            T2 = time.perf_counter()
            print("Loading time: {}s".format((T2-T1))) #197.9s
            print("Storing nodes set...")
            with open(file_nodes, 'wb') as file:
                pickle.dump(nodes, file)
            print("Stored!")
        
        findNode(args, nodes)
    else:
        count_QAs(args)



    

