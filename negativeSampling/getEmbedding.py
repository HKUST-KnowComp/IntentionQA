import pandas as pd
import numpy as np
import networkx as nx
import pickle
import random
import numpy as np
import json
import argparse
from tqdm import tqdm
import os
import time
import string
from collections import Counter
from sentence_transformers import SentenceTransformer
from aserRelationTemplate import templates

DATA_PATH = "./data"

'''
/QA/embeding_dict.pickle: {item_id: [nodes in aser]} => {item_id: [(node, avg(sbert())]}
/QA/typicality_filtered.csv: assertions

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


def get_neighborhood(origin, G):
    nodes = list(G.predecessors(origin)) + list(G.successors(origin)) + [origin]
    nodes = set(nodes)
    subG = G.subgraph(list(nodes))
    return subG

def get_embedding(g, model):
    sentences = []
    for u, v, relation in g.edges(data="relation"):
        for rel in relation:
            sentences.append(templates[rel].format(h=u,t=v)) #relation in natural language or not?
    embeddings = model.encode(sentences)
    embeddings = np.mean(embeddings, axis=0) # shape (len,)
    return embeddings

def main(args):
    print("reading ASER...")
    # raise
    T1 = time.perf_counter()
    G = nx.read_gpickle("./data/ASER2-1_Norm/G_aser2.1_nodefilteronly_norm.pickle")
    T2 = time.perf_counter()
    print("Loading time: {}s".format((T2-T1))) #197.9s
    # G = nx.DiGraph()
    res_dict_file = DATA_PATH + "/QA/embedding_dict_byCate.pickle"
    res_dict = load_pickle(res_dict_file)
    key_list = list(res_dict)

    model = SentenceTransformer(args.model)
    test_flag = 2
    for i in tqdm(range(len(key_list))):
        node_list = res_dict[key_list[i]]
        avg_embeddings = []
        for subls in node_list:
            subls_embedding = []
            for j in range(len(subls)):
                cur_node = subls[j]
                subG = get_neighborhood(cur_node, G)
                embedding = get_embedding(subG, model)
                if test_flag >0:
                    test_flag -= 1
                    print(embedding.shape)
                subls_embedding.append(embedding)
            cur_nodes_embedding = np.mean(subls_embedding, axis = 0)
            avg_embeddings.append(cur_nodes_embedding)
        avg_embeddings = np.mean(avg_embeddings, axis = 0)
        if test_flag >0:
            print(avg_embeddings.shape,"avg")
        res_dict[key_list[i]] = avg_embeddings
    print("Storing result dict...")
    store_pickle(DATA_PATH+"/QA/embedding_dict_"+args.model+".pickle", res_dict)
    print("{} stored!".format(DATA_PATH+"/QA/embedding_"+args.model+".pickle"))


    




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default = 'all-MiniLM-L6-v2', type = str, required = False, )

    args = parser.parse_args()
    main(args)