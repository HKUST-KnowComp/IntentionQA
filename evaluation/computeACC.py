# from prompt_utils import LM_generation, count_tokens, toPrompt
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
import pickle
from collections import Counter

HARD = 0.60
MEDIUM = 0.85


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


def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def answer_parser_0shot(response):
    response = response.strip(".")
    if len(response):
        return response[0].upper()
    else:
        return " "
def answer_parser_cot(response):
    response = response.strip(".")
    if len(response) == 0:
        return " "
    else:
        response = response.upper()
    index = response.find("STEP 2:")
    if index == -1:
        return " "
    if index + 7 == len(response):
        return " "
    response = response[index+7:].strip()[0]
    return response

def answer_parser_cotsc(response):
    # response = response.strip(".")
    result = []
    for r in response:
        result.append(answer_parser_cot(r))
    element_count = Counter(result)
    most_common_element, frequency = element_count.most_common(1)[0]
    return most_common_element
    
    
ANSWER_PARSERs = {
    '0shot': answer_parser_0shot,
    'cot': answer_parser_cot,
    'cotsc':answer_parser_cotsc,
}

def isAnswerCorrect(gold, answer):
    return gold == answer

def loadItemSimilarity():
    print("Loading itemSimilarity dictionary...")
    start_time = time.time()
    dic = load_pickle("./data/QA/itemSimilarity.pickle")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("itemSimilarity loaded! Elapsed time:", elapsed_time/60, "minites")
    return dic
def getDifficultyLevel():
    dic = loadItemSimilarity()
    df = pd.read_csv("./data/QA/typicality_filtered.csv",header = 0)
    id_a = list(df['item_a_id'])
    id_b = list(df['item_b_id'])
    fs_id = list(df['id'])
    difficulty_dict = dict()
    for i in tqdm(range(len(fs_id))):
        key = fs_id[i]
        difficulty_dict[key] = dic[(id_a[i],id_b[i])]
    print("Difficulty level computed! Storing to ./data/QA/fs_difficulty.pickle...")
    store_pickle("./data/QA/fs_difficulty.pickle", difficulty_dict)
    print("Stored!")

# PATH = "./data/QA_promptResult/gpt-35-turbo"
def id2difficulty(fs_id, difficulty_dict):
    score = difficulty_dict[fs_id]
    if score < HARD:
        return "hard"
    elif score < MEDIUM:
        return "medium"
    else:
        return "easy"


def main(args):
    difficulty_dict = load_pickle("./data/QA/fs_difficulty.pickle")

    data = read_data(args.p_response + '/response.jsonl')
    answer_parser = ANSWER_PARSERs[args.mode]
    id_list = []
    rel_list = []
    isCorrect_list = []
    difficulty_list = []
    j=0
    for i in tqdm(range(len(data))):
        d = data[i]
        if d['completion_tokens'] == -1 or d['response'] == 'Error':
            j=j+1
            continue
        if len(d['options']) < 4:
            continue
        gold = d['gold_ind']
        answer = answer_parser(d['response'])
        id_list.append(d['id'])
        rel_list.append(d['relation'])

        difficulty_list.append(id2difficulty(d['id'], difficulty_dict))
        if isAnswerCorrect(gold, answer):
            isCorrect_list.append(1)
        else:
            isCorrect_list.append(0)
    print (f"invalid response, { j/len(data)} ")
    df = pd.DataFrame({'id': id_list, 'relation': rel_list, 'acc': isCorrect_list, 'difficulty': difficulty_list})
    df.to_csv(args.p_response + '/scores.csv', index=False)

    difficulty_scores = df.groupby('difficulty')['acc'].mean()
    print("Difficulty: easy -> ({},1), medium -> ({}, {}), hard -> (0, {})".format(MEDIUM, HARD, MEDIUM, HARD))
    print(difficulty_scores)
    rel_list = list(set(rel_list))
    scores = []
    for rel in rel_list:
        sub_df = df[df['relation'] == rel]
        scores.append(sub_df['acc'].mean())
    rel_list.append('TOTAL')
    scores.append(df['acc'].mean())
    print('TOTAL:', df['acc'].mean())
    for diff in ['easy','medium','hard']:
        rel_list.append(diff)
        scores.append(difficulty_scores[diff])
    
    df = pd.DataFrame({'relation': rel_list, 'scores': scores})
    df.to_csv(args.p_response + '/score_report.csv', index=False)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--p_response", default=None, type=str, required=True, help="path to response folder")
    parser.add_argument("--mode", default="0shot", type=str, required=False, help="prompt method")
    args = parser.parse_args()
    main(args)
    # getDifficultyLevel()


