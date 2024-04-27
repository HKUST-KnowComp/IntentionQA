from prompt_utils import LM_generation, count_tokens, toPrompt_task1
import pandas as pd
import numpy as np
import tqdm
import json
import argparse
import csv
import logging
from tqdm import tqdm
import os
import glob
import time

def write_data(filename, data):
    with open(filename, 'a') as fout:
        for sample in data:
            fout.write(json.dumps(sample))
            fout.write('\n')


def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def QA_count_tokens(args):
    print("Counting tokens for [{}]...".format(args.QA_path))
    data = read_data(args.QA_path)
    num_tokens = 0
    for i in tqdm(range(len(data))):
        d = data[i]
        x = toPrompt_task1(d['item_a_name'], d['item_b_name'], d['options'], mode = args.mode)
        num_tokens += count_tokens(x)
        if i < 3:
            print(x+"===============\n")
    print("Total num_tokens = {} (input)\nFor {} QAs\nAvg num_tokens = {}\nPricing/1000*0.03 = ${}".format(num_tokens, len(data), num_tokens/len(data), num_tokens/1000*0.03))

def QA_to_LM(args):
    if args.mode == 'cot':
        _max_token = 200
    else:
        _max_token = 10
    print("Generating answers with {} on [{}]...".format(args.LM, args.QA_path))
    data = read_data(args.QA_path)
    if "hard" in args.QA_path:
        difficulty = 'hard'
    else:
        difficulty = 'easy'
    out_dir = args.out_dir+"/{}/p_{}/{}".format(args.LM, args.mode, difficulty)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = out_dir+"/response.jsonl".format(args.LM,args.mode)
    for i in tqdm(range(len(data))):
        d = data[i]
        x = toPrompt_task1(d['item_a_name'], d['item_b_name'], d['options'], mode = args.mode)
        response, a, b = LM_generation(x, max_tokens = _max_token, temperature=0.1, model = args.LM)
        d['response'] = response
        d['prompt_tokens'] = a
        d['completion_tokens'] = b
        d['x'] = x
        write_data(out_file, [d])

def QA_to_LM_selfconsistency(args):
    _max_token = 200
    print("COTSC: Generating answers with {} on [{}]...".format(args.LM, args.QA_path))
    data = read_data(args.QA_path)
    if "hard" in args.QA_path:
        difficulty = 'hard'
    else:
        difficulty = 'easy'
    out_dir = args.out_dir+"/{}/p_{}/{}".format(args.LM, args.mode, difficulty)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = out_dir+"/response.jsonl".format(args.LM,args.mode)
    for i in tqdm(range(len(data))):
        d = data[i]
        x = toPrompt_task1(d['item_a_name'], d['item_b_name'], d['options'], mode = args.mode)
        d['x'] = x
        d['response'] = []
        d['prompt_tokens'] = []
        d['completion_tokens'] = []
        for j in range(5):
            response, a, b = LM_generation(x, max_tokens = _max_token, temperature=0.7, model = args.LM)
            d['response'].append(response)
            d['prompt_tokens'].append(a)
            d['completion_tokens'].append(b)
        write_data(out_file, [d])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--QA_path", default="./data/QA/QA_fullInfo.jsonl", type=str, required=False, help="path to QA_fullInfo")
    parser.add_argument("--mode", default=None, type=str, required=True,\
        choices=['0shot', 'cot','cotsc'])
    parser.add_argument("--response", default=0, type=int, required=False, help="1:get response from openai; 0: count tokens only")
    parser.add_argument("--out_dir", default="./data/QA_promptResult", type=str, required=False, help="Output dir")
    parser.add_argument("--LM", default="gpt-35-turbo", type=str, required=False,help="Langage model to use")
    parser.add_argument("--task", default=1, type = int, help="1: predict intention given products; 2: predict product given one product and intention")
    args = parser.parse_args()

    if args.response == 0:
        QA_count_tokens(args)
    else:
        print("response == 1")
        if args.mode == 'cotsc':
            QA_to_LM_selfconsistency(args)
        else:
            QA_to_LM(args)

