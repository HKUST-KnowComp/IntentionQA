import pandas as pd
import numpy as np
import pickle
import random
import numpy as np
import json
import argparse
from tqdm import tqdm


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


def hasOverlapConcept(A,B,C,D, itemsInfo):
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


def negativeSampling_task1(path_to_assertionFile, similarity_byrel_dir, out_file, threshold_h, threshold_l, args):
    print(out_file)
    if args.mixRelation:
        out_file += "QA_raw_mixRelation"
        print(out_file)
    else:
        out_file += "QA_raw_byRelation"
        print(out_file)
    out_file += "_{}_{}.jsonl".format(threshold_l, threshold_h)
    print(out_file)
    
    print("Generating distractors to {}...".format(out_file))
    
    assertions = pd.read_csv(path_to_assertionFile)
    relations = list(set(assertions['relation']))
    
    if not args.mixRelation:
        for rel in relations:
            print("Generating QA for relation: {}\n".format(rel))
            cur_rel_dict = load_pickle(similarity_byrel_dir+"/sim_"+rel+".pickle")
            cur_rel_assertions = assertions[assertions['relation'] == rel]
            cur_rel_assertionIds = list(cur_rel_assertions['id'])
            for i in tqdm(range(len(cur_rel_assertionIds))):
                cur_id = cur_rel_assertionIds[i]
                cur_record = assertions[assertions['id'] == cur_id]
                sub_dict = {k:v for k,v in cur_rel_dict.items() if k[0] == cur_id and v[1] == True and v[0] >= threshold_l}
                sorted_sub_dict = sorted(sub_dict.items(), key=lambda x:x[1][0])

                if len(sorted_sub_dict) > 128 and sorted_sub_dict[128][1][0] < threshold_h:
                    distractors = random.sample(sorted_sub_dict[:128+1], 3)
                elif len(sorted_sub_dict) > 64 and sorted_sub_dict[64][1][0] < threshold_h:
                    distractors = random.sample(sorted_sub_dict[:64+1], 3)
                elif len(sorted_sub_dict) > 32 and sorted_sub_dict[32][1][0] < threshold_h:
                    distractors = random.sample(sorted_sub_dict[:32+1], 3)
                elif len(sorted_sub_dict) > 16 and sorted_sub_dict[16][1][0] < threshold_h:
                    distractors = random.sample(sorted_sub_dict[:16+1], 3)
                elif len(sorted_sub_dict) > 8 and sorted_sub_dict[8][1][0] < threshold_h:
                    distractors = random.sample(sorted_sub_dict[:8+1], 3)
                elif len(sorted_sub_dict) > 4 and sorted_sub_dict[4][1][0] < threshold_h:
                    distractors = random.sample(sorted_sub_dict[:4+1], 3)
                elif len(sorted_sub_dict) > 3 and sorted_sub_dict[3][1][0] < threshold_h:
                    distractors = sorted_sub_dict[:4]
                elif len(sorted_sub_dict) > 2 and sorted_sub_dict[2][1][0] < threshold_h:
                    distractors = sorted_sub_dict[:3]
                elif len(sorted_sub_dict) > 1 and sorted_sub_dict[1][1][0] < threshold_h:
                    distractors = sorted_sub_dict[:2]
                elif len(sorted_sub_dict) > 0 and sorted_sub_dict[0][1][0] < threshold_h:
                    distractors = sorted_sub_dict[:1]
                else:
                    distractors = []

                if len(distractors):
                    distractors_id = [x[0][1] for x in distractors]
                    distractors_score = [x[1][0].item() for x in distractors]
                    distractors_overlap = [int(x[1][1]) for x in distractors]
                    distractors_asserstion = list(assertions[assertions['id'].isin(distractors_id)]['assertion'])
                else:
                    distractors_id = []
                    distractors_score = []
                    distractors_overlap = []
                    distractors_asserstion = []
                cur_res_QA = {'id': cur_id, 'item_a_id': list(cur_record['item_a_id'])[0], 'item_b_id':list(cur_record['item_b_id'])[0], 'item_a_name': list(cur_record['item_a_name'])[0],'item_b_name': list(cur_record['item_b_name'])[0],'relation': rel, 'assertion': list(cur_record['assertion'])[0], \
                            'distractors_id':distractors_id, 'distractors_score':distractors_score, 'distractors_overlap':distractors_overlap, 'distractors_asserstion':distractors_asserstion,\
                            'typicality': list(cur_record['typicality'])[0]}
                write_data(out_file, [cur_res_QA])
    
    if args.mixRelation:
        for rel in ['all']:
            print("Generating QA for relation: {}\n".format(rel))
            cur_rel_dict = load_pickle(similarity_byrel_dir+"/sim_"+rel+".pickle")
            cur_rel_assertions = assertions
            cur_rel_assertionIds = list(cur_rel_assertions['id'])
            num_no_distractors = 0
            for i in tqdm(range(len(cur_rel_assertionIds))):
                cur_id = cur_rel_assertionIds[i]
                cur_record = assertions[assertions['id'] == cur_id]
                if args.mixRelation:
                    sub_dict = {k:v for k,v in cur_rel_dict.items() if k[0] == cur_id and v[1] == True and v[0] >= threshold_l}
                else:
                    sub_dict = {k:v for k,v in cur_rel_dict.items() if k[0] == cur_id and v[0] >= threshold_l}
                sorted_sub_dict = sorted(sub_dict.items(), key=lambda x:x[1][0])
                if len(sorted_sub_dict) > 128 and sorted_sub_dict[128][1][0] < threshold_h:
                    distractors = random.sample(sorted_sub_dict[:128+1], 3)
                elif len(sorted_sub_dict) > 64 and sorted_sub_dict[64][1][0] < threshold_h:
                    distractors = random.sample(sorted_sub_dict[:64+1], 3)
                elif len(sorted_sub_dict) > 32 and sorted_sub_dict[32][1][0] < threshold_h:
                    distractors = random.sample(sorted_sub_dict[:32+1], 3)
                elif len(sorted_sub_dict) > 16 and sorted_sub_dict[16][1][0] < threshold_h:
                    distractors = random.sample(sorted_sub_dict[:16+1], 3)
                elif len(sorted_sub_dict) > 8 and sorted_sub_dict[8][1][0] < threshold_h:
                    distractors = random.sample(sorted_sub_dict[:8+1], 3)
                elif len(sorted_sub_dict) > 4 and sorted_sub_dict[4][1][0] < threshold_h:
                    distractors = random.sample(sorted_sub_dict[:4+1], 3)
                elif len(sorted_sub_dict) > 3 and sorted_sub_dict[3][1][0] < threshold_h:
                    distractors = sorted_sub_dict[:4]
                elif len(sorted_sub_dict) > 2 and sorted_sub_dict[2][1][0] < threshold_h:
                    distractors = sorted_sub_dict[:3]
                elif len(sorted_sub_dict) > 1 and sorted_sub_dict[1][1][0] < threshold_h:
                    distractors = sorted_sub_dict[:2]
                elif len(sorted_sub_dict) > 0 and sorted_sub_dict[0][1][0] < threshold_h:
                    distractors = sorted_sub_dict[:1]
                else:
                    distractors = []
                if len(distractors):
                    distractors_id = [x[0][1] for x in distractors]
                    distractors_score = [x[1][0].item() for x in distractors]
                    distractors_overlap = [int(x[1][1]) for x in distractors]
                    distractors_asserstion = list(assertions[assertions['id'].isin(distractors_id)]['assertion'])
                else:
                    distractors_id = []
                    distractors_score = []
                    distractors_overlap = []
                    distractors_asserstion = []
                    num_no_distractors += 1
                cur_res_QA = {'id': cur_id, 'item_a_id': list(cur_record['item_a_id'])[0], 'item_b_id':list(cur_record['item_b_id'])[0], 'item_a_name': list(cur_record['item_a_name'])[0],'item_b_name': list(cur_record['item_b_name'])[0],'relation': rel, 'assertion': list(cur_record['assertion'])[0], \
                            'distractors_id':distractors_id, 'distractors_score':distractors_score, 'distractors_overlap':distractors_overlap, 'distractors_asserstion':distractors_asserstion,\
                            'typicality': list(cur_record['typicality'])[0]}

                write_data(out_file, [cur_res_QA])


def negativeSampling_task2(path_to_assertionFile, item_similarity_dir, out_file, threshold_h, threshold_l, args):


    print(out_file)
    out_file += "QA_raw_task2"
    out_file += "_{}_{}.jsonl".format(threshold_l, threshold_h)
    print(out_file)
    
    print("Generating distractors for task 2 to {}...".format(out_file))
    assertions = pd.read_csv(path_to_assertionFile)
    relations = list(set(assertions['relation']))
    itemsInfo = pd.read_json(args.itemsInfo, lines = True)
    
    
    rel = 'all'
    print("Generating QA for relation: {}\n".format(rel))
    cur_rel_dict = load_pickle(item_similarity_dir)
    cur_rel_assertions = assertions
    cur_rel_assertionIds = list(cur_rel_assertions['id'])
    cur_rel_ItemAIds = list(cur_rel_assertions['item_a_id'])
    num_no_distractors = 0
    for i in tqdm(range(len(cur_rel_ItemAIds))):
        cur_itemA_id = cur_rel_ItemAIds[i]
        cur_id = cur_rel_assertionIds[i]
        cur_record = assertions[assertions['id'] == cur_id]

        sub_dict = {k:v for k,v in cur_rel_dict.items() if k[0] == cur_itemA_id and v[1] == True and v[0] >= threshold_l and v[0] < threshold_h}

        sorted_sub_dict = sorted(sub_dict.items(), key=lambda x:x[1][0])
        num_distractors_to_sample = min(3, len(sorted_sub_dict))
        distractors = random.sample(sorted_sub_dict, num_distractors_to_sample)
    
        if len(distractors):
            distractors_id = [x[0][1] for x in distractors]
            distractors_score = [x[1][0].item() for x in distractors]
            distractors_products = list(itemsInfo[itemsInfo['id'].isin(distractors_id)]['name'])
        else:
            distractors_id = []
            distractors_score = []
            distractors_products = []
            num_no_distractors += 1
        cur_res_QA = {'id': cur_id, 'item_a_id': list(cur_record['item_a_id'])[0], 'item_b_id':list(cur_record['item_b_id'])[0], 'item_a_name': list(cur_record['item_a_name'])[0],'item_b_name': list(cur_record['item_b_name'])[0],'relation': rel, 'assertion': list(cur_record['assertion'])[0], \
                        'distractors_id':distractors_id, 'distractors_score':distractors_score, 'distractors_product':distractors_products,\
                        'typicality': list(cur_record['typicality'])[0]}
        write_data(out_file, [cur_res_QA])


            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--itemsInfo", default = "./data/items_simplified_byCate.jsonl", type = str, required = False)
    parser.add_argument("--similarity_byrel_dir", default = "./data/QA_task1/assertionSimilarity", type = str, required = False)
    parser.add_argument("--itemSimilarity_dir", default='./data/QA/itemSimilarity_updated.pickle', type=str, required=False)
    parser.add_argument("--path_to_assertionFile", default = "./data/QA/typicality_filtered.csv", type = str, required = False)
    parser.add_argument("--out_dir", default = './data/QA_task2/QA_raw/', type = str, required = False)
    parser.add_argument("--mixRelation", action="store_true", help="whether to generate QA dataset for main experiment or byRelation") 
    parser.add_argument("--threshold_h", default=0.5, type = float,) 
    parser.add_argument("--threshold_l", default=0, type = float, ) 
    parser.add_argument("--task", default=1, type = int, help="1: predict intention given products; 2: predict product given one product and intention")
    args = parser.parse_args()
    if args.mixRelation:
        print("Generating QA dataset whose options are mixed-relations ...")
    if args.task == 1:
        negativeSampling_task1(args.path_to_assertionFile, args.similarity_byrel_dir, args.out_dir, args.threshold_h, args.threshold_l, args)
    if args.task == 2: 
        negativeSampling_task2(args.path_to_assertionFile, args.itemSimilarity_dir, args.out_dir, args.threshold_h, args.threshold_l, args)
