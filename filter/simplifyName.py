import pandas as pd
import csv
import pandas as pd
import numpy as np
import json
import argparse
from tqdm import tqdm, trange
import os
from collections import Counter
import tiktoken
import openai  # for OpenAI API calls
import re
import time

openai.api_key = "KEY_TO_BE_FILLED"
openai.api_base = "https://hkust.azure-api.net"
openai.api_type = "azure"
openai.api_version = "2023-05-15"
model = "gpt-35-turbo"



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

def getItems():
    print("df_all loading!")
    df_all = pd.read_csv("./data/typicality_annotated.csv")

    print("df_all loaded30")
    col_list = ['item_a_name','item_b_name','item_a_id','item_b_id','item_a_cate','item_b_cate']
    print("df_all loaded!")
    df_all = df_all[col_list]
    print("df_all loaded!")
    id_set = set()
    ids = []
    names = []
    cates = []
    for i in trange(len(df_all)):
        item_a_id = df_all.loc[i, 'item_a_id']
        item_b_id = df_all.loc[i, 'item_b_id']
        if item_a_id not in id_set:
            id_set.add(item_a_id)
            ids.append(df_all.loc[i, 'item_a_id'])
            names.append(df_all.loc[i, 'item_a_name'])
            cates.append(df_all.loc[i, 'item_a_cate'])
        if item_b_id not in id_set:
            id_set.add(item_b_id)
            ids.append(df_all.loc[i, 'item_b_id'])
            names.append(df_all.loc[i, 'item_b_name'])
            cates.append(df_all.loc[i, 'item_b_cate'])

    out_file = "./data/items_folkscope.csv"
    items_info = {'id':ids, 'name':names, 'cate':cates}
    df_out = pd.DataFrame(items_info)
    df_out.to_csv(out_file,index=False)

def count_tokens(text, encoding_name = "cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens

def getResponse(model, x, _max_tokens, _temperature):
    # print('\n',x)
    try:
        response = openai.ChatCompletion.create(
            engine=model,
            messages=[
                {"role": "user", "content": x},
            ],
            max_tokens= _max_tokens,
            temperature= _temperature,
        )
    except openai.error.APIError as e:
        print(f"OpenAI API returned an API Error: {e}")
        time.sleep(1)
        return "openai.error.APIError", -1, -1 
    except openai.error.InvalidRequestError as e:
        print(f"openai.error.InvalidRequestError: {e}")
        time.sleep(1)
        return "openai.error.InvalidRequestError", -1, -1 
    except openai.error.APIConnectionError as e:
        print(f"Failed to connect to OpenAI API: {e}")
        time.sleep(1)
        return "openai.error.APIConnectionError", -1, -1 
    except openai.error.RateLimitError as e:
        #Handle rate limit error (we recommend using exponential backoff)
        print(f"OpenAI API request exceeded rate limit: {e}")
        time.sleep(1)
        return "openai.error.RateLimitError", -1, -1 
    except openai.error.ServiceUnavailableError as e:
        print(f"openai.error.ServiceUnavailableError: {e}")
        time.sleep(3)
        return "openai.error.ServiceUnavailableError", -1, -1 
    except openai.error.Timeout as e:
        print(f"openai.error.Timeout: {e}")
        time.sleep(1)
        return "openai.error.Timeout", -1, -1 
    # print(response["choices"][0]["message"], response)
    if "content" in response["choices"][0]["message"]:
        content = response["choices"][0]["message"]["content"]
    else:
        return "", -1, -1
    return response["choices"][0]["message"]["content"], response["usage"]["prompt_tokens"],response["usage"]["completion_tokens"]

def parseByComma(s):
    substrings = s.split(',')
    res = [substring.strip() for substring in substrings]
    return res

def prompt_simplifyName(args):
    out_file = args.out_dir
    print("Loading dataset...")
    df = pd.read_csv(args.dataset)
    total_tokens = 0
    if args.response == 1:
        print("Generating item categories...")
    else:
        print("Counting tokens...")
    if args.last_id == "":
        flag_getResponse = True
    else:
        flag_getResponse = False
    for i in trange(len(df)):
        cur_d = dict()
        cur_d['id'] = df.loc[i, 'id']
        cur_d['name'] = df.loc[i,'name']
        cur_d['cate'] = df.loc[i,'cate']
        cur_d['item_category'] = []
        prompt = "Product name: {}; What is the category of the product? Generate three possible categories, each in 2 words, separated by comma.".format(cur_d['name'])
        if not flag_getResponse and cur_d['id'] != args.last_id:
            continue
        if not flag_getResponse and cur_d['id'] == args.last_id:
            flag_getResponse = True
            print("Continue after {}".format(cur_d['id']))
            continue
        if args.response == 1:
            # print(prompt)
            res, n_p, n_c = getResponse(model, prompt, 50, 0)
            # print(res)
            if n_p == -1:
                cur_d['item_category'], cur_d['n_prompt'], cur_d['n_completion'] = [res], n_p, n_c
                write_data(out_file, [cur_d])
                write_data(args.error_log, [cur_d])
            else:
                cur_d['item_category'], cur_d['n_prompt'], cur_d['n_completion'] = parseByComma(res), n_p, n_c
                write_data(out_file, [cur_d])
        else:
            total_tokens += count_tokens(prompt)
    if args.response == 1:
        print("All items simplified!")
    else:
        print("len(data) = {}, total_tokens = {}".format(len(df), total_tokens))
        

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str, required=True, help="dataset:path to KG file or the folder containing all KG files")
    parser.add_argument("--retry", default=False, type=bool, required=False)
    parser.add_argument("--response", default=1, type=int, required=False, help="1:get response from openai; 0: count tokens only")
    parser.add_argument("--size", default = -1, type = int, required = False, help="number of QA data")
    parser.add_argument("--last_id", default = "", type = str, required = False)
    parser.add_argument("--out_dir", default=None, type=str, required=False, help="Output dir")
    parser.add_argument("--error_log", default=None, type=str, required=False, help="error log dir")
    args = parser.parse_args()
    prompt_simplifyName(args)
# python simplifyName.py --dataset ./data/items_folkscope.csv --response 0 --out_dir ./data/items_simplified.jsonl --error_log ./data/items_error_log.jsonl