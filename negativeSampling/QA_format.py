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
    with open(filename, 'a') as fout:
        for sample in data:
            fout.write(json.dumps(sample))
            fout.write('\n')

def resonAfterBecause(assertions):
    for i in range(len(assertions)):
        index = assertions[i].find("because")
        if index == -1:
            index = assertions[i].find("as a result, ")
        if index == -1:
            print("WARNING! index (reasonAfterBecause/AsAResult) == -1!")
        assertions[i] = assertions[i][index:].strip()  
    return assertions

def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def insert_gold(distractors, answer):
    hasFalseNegative = False
    for i in range(len(distractors)):
        if distractors[i] == answer:
            distractors[i] = '-1'
            hasFalseNegative = True
    distractors = [i for i in distractors if i != '-1']
    gold_ind = random.randint(0, len(distractors))
    distractors.insert(gold_ind,answer)
    return gold_ind, distractors, hasFalseNegative

def main(args):
    def task_1(args):
        raw_file = args.raw_dir
        if args.mixRelation:
            raw_file += "QA_raw_mixRelation"
        else:
            raw_file += "QA_raw_byRelation"
        raw_file += "_{}_{}.jsonl".format(args.threshold_l, args.threshold_h)
        raw_data = read_data(raw_file)

        if args.mixRelation:
            out_file = args.out_dir+"QA_fullInfo_mixRelation_{}_{}.jsonl".format(args.threshold_l, args.threshold_h)
        else:
            out_file = args.out_dir+"QA_fullInfo_byRelation_{}_{}.jsonl".format(args.threshold_l, args.threshold_h)
        for i in tqdm(range(len(raw_data))):
            d = raw_data[i]
            if len(d['distractors_asserstion']) == 0:
                continue
            distractors_asserstion = resonAfterBecause(d['distractors_asserstion'])
            gold_answer = resonAfterBecause([d['assertion']])[0]
            gold_ind, options, hasFalseNegative = insert_gold(distractors_asserstion, gold_answer)
            d['gold_ind'] = chr(ord('A') + gold_ind)
            d['options'] = {chr(ord('A') + j): options[j] for j in range(len(options))}
            d['hadFalseNegative'] = int(hasFalseNegative)
            write_data(out_file, [d])
    def task_2(args):
        raw_file = args.raw_dir
        raw_file += "QA_raw_task2"
        raw_file += "_{}_{}.jsonl".format(args.threshold_l, args.threshold_h)
        raw_data = read_data(raw_file)

        out_file = args.out_dir+"QA_fullInfo_task2_{}_{}.jsonl".format(args.threshold_l, args.threshold_h)
        for i in tqdm(range(len(raw_data))):
            d = raw_data[i]
            distractors_products = d['distractors_product']
            gold_answer = d['item_b_name']
            gold_ind, options, hasFalseNegative = insert_gold(distractors_products, gold_answer)
            d['gold_ind'] = chr(ord('A') + gold_ind)
            d['options'] = {chr(ord('A') + j): options[j] for j in range(len(options))}
            d['hadFalseNegative'] = int(hasFalseNegative)
            write_data(out_file, [d])
    if args.task == 1:
        task_1(args)
    else:
        task_2(args)
'''
x = "A customer buys {item_a} and {item_b}. What is the most likely intention for buying them?\n".format(item_a=d['item_a_name'],item_b=d['item_b_name'])
for k, v in d['options'].items():
    x += k+'. '+v+'\n'
x = x[:-1]
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default = './data/QA_task2/QA_raw/', type = str, required = False)
    parser.add_argument("--out_dir", default = '/data/QA_task2/QA_fullInfo/', type = str, required = False)
    parser.add_argument("--mixRelation", action="store_true", help="whether to generate mixRelation QA dataset") 
    parser.add_argument("--threshold_h", default=0.5, type = float, help="whether to generate mixRelation QA dataset") 
    parser.add_argument("--threshold_l", default=0, type = float, help="whether to generate mixRelation QA dataset") 
    parser.add_argument("--task", default=1, required = True, type = int, help="1: predict intention given products; 2: predict product given one product and intention")
    
    args = parser.parse_args()
    main(args)
   

