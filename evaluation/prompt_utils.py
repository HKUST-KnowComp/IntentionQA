import json
import re
import sys
import time
import warnings
from random import sample

import openai
from openai import AzureOpenAI
import pandas as pd
from tqdm import tqdm
import tiktoken

warnings.filterwarnings("ignore")


client = AzureOpenAI(
    api_key = "KEY_TO_BE_FILLED",
    api_version = "2023-12-01-preview",
    azure_endpoint = "https://hkust.azure-api.net"
)


def LM_generation(prompt, max_tokens = 10, temperature = 0.1, model = "gpt-35-turbo"):
    retry_attempt = 5
    retry_num = 0
    generation_success = False
    messages = [
        {'role':'user', 'content':prompt},
    ]
    while retry_num < retry_attempt and not generation_success:
        try:
            gen = client.chat.completions.create(
                model = model,
                messages = messages,
                max_tokens = max_tokens,
                temperature = temperature,
            ) 
            generation_success = True
            input_tokens = gen.usage.prompt_tokens
            output_tokens = gen.usage.completion_tokens
        except openai.APIError as e:
            retry_num += 1
            generation_success = False
            time.sleep(5)
            # return "openai.error.APIError", -1, -1 
        except openai.RateLimitError as e:
            retry_num += 1
            generation_success = False
            time.sleep(30)
        except:
            retry_num += 1
            generation_success = False
            time.sleep(10)
    
    if generation_success:
        if gen == None or gen.choices[0] == None or gen.choices[0].message == None or gen.choices[0].message.content == None:
            return "", 0, 0
        return gen.choices[0].message.content.strip(), input_tokens, output_tokens
    else:
        return "Error", -1,-1

def resonAfterBecause(assertions):
    if type(assertions) == str:
        assertions = [assertions]
    for i in range(len(assertions)):
        index = assertions[i].find("because")
        if index == -1:
            index = assertions[i].find("as a result, ")
        if index == -1:
            print("WARNING! index (reasonAfterBecause/AsAResult) == -1!")
        assertions[i] = assertions[i][index:].strip()  
    return assertions


def toPrompt_task1(a,b, options, mode = '0shot'):
    if mode == '0shot':
        x = "A customer buys {} and {}. What is the most likely intention for buying them?\n".format(a,b)
        for k, v in options.items():
            x += k+'. '+ v+ '\n'
        x += "Answer {} only without any other word.".format(" or ".join(options))
        return x
    elif mode == 'cot' or mode == 'cotsc':
        x = "A customer buys {} and {}. What is the most likely intention for buying them?\n".format(a,b)
        for k, v in options.items():
            x += k+'. '+ v+ '\n'
        x += "Formulate your answer in this way.\nStep 1: Give a short and brief rationale by thinking step by step.\nStep 2: Answer {} only without any other word.".format(" or ".join(options))
        return x
    else:
        raise ValueError("Invalid mode")

def toPrompt_task2(a,assertions, options, mode = '0shot'):
    if mode == '0shot':
        x = "A customer buys {}, {}".format(a,resonAfterBecause(assertions)[0])
        if x[-1] == '.':
            x += '\n'
        else:
            x += '.\n'
        x += "What is the customer's most probable additional purchase?\n"
        # #######"Which product will the customer most likely purchase?"
        # "Which product does the customer most likely want to purchase?"
        # "What is the customer's most probable additional purchase?"
        # "Which product is the customer most inclined to buy alongside?"
        # "What is the customer's likely additional purchase?"
        # "Which product is the customer most expected to buy as well?"
        for k, v in options.items():
            x += k+'. '+ v+ '\n'
        x += "Answer {} only without any other word.".format(" or ".join(options))
        return x
    elif mode == 'cot' or mode == 'cotsc':
        x = "A customer buys {}, {}".format(a,resonAfterBecause(assertions)[0])
        if x[-1] == '.':
            x += '\n'
        else:
            x += '.\n'
        x += "What is the customer's most probable additional purchase?\n"
        for k, v in options.items():
            x += k+'. '+ v+ '\n'
        x += "Formulate your answer in this way.\nStep 1: Give a short and brief rationale by thinking step by step.\nStep 2: Answer {} only without any other word.".format(" or ".join(options))
        return x
    else:
        raise ValueError("Invalid mode")

def count_tokens(text, encoding_name = "cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    # print(text)
    num_tokens = len(encoding.encode(text))
    return num_tokens








# '''A customer buys StarTech.com 60x10mm Replacement Ball Bearing Computer Case Fan w/ TX3 Connector - 3 pin case Fan - TX3 Fan - 60mm Fan and StarTech 8-Inch 24 Pin ATX 2.01 Power Extension Cable (ATX24POWEXT). What is the most likely intention for buying them?
# A. because they both are defined as "Space Suit Costume" in the same category.
# B. because they both are defined as "costumes" and both are "fun".
# C. because they both are defined as "Balaclava" in the product title.
# D. because they both are defined as "Computer Accessories".'''

# '''A customer buys StarTech.com 60x10mm Replacement Ball Bearing Computer Case Fan w/ TX3 Connector - 3 pin case Fan - TX3 Fan - 60mm Fan and StarTech 8-Inch 24 Pin ATX 2.01 Power Extension Cable (ATX24POWEXT). What is the most likely intention for buying them?
# A. because they both are defined as ""Space Suit Costume"" in the same category.
# B. because they both are defined as ""costumes"" and both are ""fun"".
# C. because they both are defined as ""Balaclava"" in the product title.
# D. because they both are defined as ""Computer Accessories"".'''
# '''A customer buys ELAC C5 Debut Series 5.25&quot; Center Speaker by Andrew Jones and ELAC Debut B4 Bookshelf Speaker (Black, Pair). What is the most likely intention for buying them? Answer A or B or C or D only without any other word.
# A. because they both are capable of being worn by a pirate.
# B. because they both are capable of keeping him warm.
# C. because they both are capable of delivering high quality sound.
# D. because they both are capable of making him look more handsome and more charming.'''

# '''A customer buys ELAC C5 Debut Series 5.25&quot; Center Speaker by Andrew Jones and ELAC Debut B4 Bookshelf Speaker (Black, Pair). What is the most likely intention for buying them?
# A. because they both are capable of being worn by a pirate.
# B. because they both are capable of keeping him warm.
# C. because they both are capable of delivering high quality sound.
# Answer A or B or C or D only without any other word.'''