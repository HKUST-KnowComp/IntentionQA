import pandas as pd
import numpy as np
import pickle
import random
import numpy as np
import json
import argparse
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


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

def loadItemSimilarity():
    print("Loading itemSimilarity dictionary...")
    start_time = time.time()
    dic = load_pickle("./data/QA/itemSimilarity.pickle")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("itemSimilarity loaded! Elapsed time:", elapsed_time/60, "minites")
    return dic





if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--raw_dir", default = './data/QA/QA_raw/', type = str, required = False)
    # parser.add_argument("--out_dir", default = './data/QA/QA_fullInfo/', type = str, required = False)
    # parser.add_argument("--hard", action="store_true", help="whether to generate hard QA dataset") 
    # parser.add_argument("--threshold_h", default=0.5, type = float, help="whether to generate hard QA dataset") 
    # parser.add_argument("--threshold_l", default=0, type = float, help="whether to generate hard QA dataset") 
    # args = parser.parse_args()
    dic = loadItemSimilarity()
    df = pd.read_csv("./data/QA/typicality_filtered.csv",header = 0)
    id_a = list(df['item_a_id'])
    id_b = list(df['item_b_id'])

    similarities = []
    for i in tqdm(range(len(id_a))):
        similarities.append(dic[(id_a[i],id_b[i])])


    numbers = similarities
    slots = {}
    for number in numbers:
        slot = int(number / 0.1)  # Determine the slot index
        slots.setdefault(slot, []).append(number)

    x = list(slots.keys())
    y = [len(values) for values in slots.values()]

    # Plot the histogram
    plt.bar(x, y, width=0.8, align='center')

    # Customize the plot
    plt.xlabel('Slots')
    plt.ylabel('Count')
    plt.title('Number Distribution')
    plt.xticks(x)
    plt.grid(True)

    # Display the plot
    plt.savefig('./plot.png')
    plt.show()
            




    


