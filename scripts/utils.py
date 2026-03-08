import os
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np
import csv
import json
from collections import Counter
import argparse
import random
import matplotlib.pyplot as plt
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)
device = utils.get_device()


ROOT_DIR = os.getcwd()[:os.getcwd().rfind('/')]

Verbs = {}
with open(f'{ROOT_DIR}/datasets/Sinclair_vocab/verblist_DT_usf_freq.csv', mode ='r')as f:
    csvFile = csv.reader(f, delimiter =';')
    for line in csvFile:
       if not line[0] == "\ufeffV":
          Verbs[line[0].upper()] = {'pres': line[0],
                                    'past': line[-3],
                                    '3sg': line[-2],
                                    'participle': line[-1],
                                    'prep': line[-4]}
Verbs_inv = {}
for verb in Verbs:
    for form in list(Verbs[verb].values())[:4]:
        Verbs_inv[form] = verb
      
      
# Define the regions of interest  
noPrime_regions = {
                'SUBJ': {'PD': (0, 2), 'DO': (0, 2)},
                'VERB': {'PD': (2, 3), 'DO': (2, 3)},
                'DOBJ': {'PD': (3, 5), 'DO': (5, 7)},
                'IOBJ': {'PD': (6, 8), 'DO': (3, 5)},
                'PREP': {'PD': (5, 6), 'DO': (0, 0)},
                'EOS': {'PD': (8, 9), 'DO': (7, 8)}}

noPrime_regions_pronoun = {
                'SUBJ': {'PD': (0, 2), 'DO': (0, 2)},
                'VERB': {'PD': (2, 3), 'DO': (2, 3)},
                'DOBJ': {'PD': (3, 5), 'DO': (4, 6)},
                'IOBJ': {'PD': (6, 7), 'DO': (3, 4)},
                'PREP': {'PD': (5, 6), 'DO': (0, 0)},
                'EOS': {'PD': (7, 8), 'DO': (6, 7)}}

POS_MAP = {'Period':0, 'V':3, 'DP1_mid':4, 'DP1_end':5, 'DP1_after':6}
POS_MAP_PronounDO = {'Period':0, 'V':3, 'DP1_end':4, 'DP1_after':5}     

# Define the regions of interest for priming
prime_regions = {
                'BOS': {'PD': (0, 1), 'DO': (0, 1)},
                'SUBJ': {'PD': (1, 3), 'DO': (1, 3)},
                'VERB': {'PD': (3, 4), 'DO': (3, 4)},
                'DOBJ': {'PD': (4, 6), 'DO': (6, 8)},
                'IOBJ': {'PD': (7, 9), 'DO': (4, 6)},
                'PREP': {'PD': (6, 7), 'DO': (0, 0)},
                'EOS': {'PD': (9, 10), 'DO': (8, 9)},
                'TARGET_SUBJ': {'PD': (10, 12), 'DO': (9, 11)},
                'TARGET_VERB': {'PD': (12, 13), 'DO': (11, 12)}}

prime_regions_pronoun = {
            'BOS': {'PD': (0, 1), 'DO': (0, 1)},
            'SUBJ': {'PD': (1, 3), 'DO': (1, 3)},
            'VERB': {'PD': (3, 4), 'DO': (3, 4)},
            'DOBJ': {'PD': (4, 6), 'DO': (5, 7)},
            'IOBJ': {'PD': (7, 8), 'DO': (4, 7)},
            'PREP': {'PD': (6, 7), 'DO': (0, 0)},
            'EOS': {'PD': (8, 9), 'DO': (7, 8)},
            'TARGET_SUBJ': {'PD': (9, 11), 'DO': (8, 10)},
            'TARGET_VERB': {'PD': (11, 12), 'DO': (10, 11)}}

trial_names = []
for i in range(15000):
    trial_names.append(f'{i}-DO-DO.pt')
    trial_names.append(f'{i}-DO-PD.pt')
    trial_names.append(f'{i}-PD-DO.pt')
    trial_names.append(f'{i}-PD-PD.pt')

with open(f'{ROOT_DIR}/results/corpus_verb_bias/OWT1Verbs.json', 'r') as file:
    temp = json.load(file)
verbs = list(temp.keys())
DO = [temp[key]['DO'] for key in temp]
PD = [temp[key]['PD'] for key in temp]
DO_ratio1 = [d/(d+p) for d, p in zip(DO, PD)]

with open(f'{ROOT_DIR}/results/corpus_verb_bias/OWT2Verbs.json', 'r') as file:
    temp = json.load(file)
verbs = list(temp.keys())
DO = [temp[key]['DO'] for key in temp]
PD = [temp[key]['PD'] for key in temp]
DO_ratio2 = [d/(d+p) for d, p in zip(DO, PD)]

DO_bias = (np.array(DO_ratio1) + np.array(DO_ratio2))/2 - np.ones(22)/2

Verbs_Subsets = {'All': None,
                 'DO_biased_Model': ['SHOW', 'FEED', 'BUY', 'DELIVER', 'PROMISE'],
                 'DO_biased_Corpus': [verb for verb, bias in zip(list(Verbs.keys()), DO_bias) if bias > 0],
                 'Large_PD_biased': [verb for verb, bias in zip(list(Verbs.keys()), DO_bias) if bias < 0.1],
                 'Large_DO_biased': [verb for verb, bias in zip(list(Verbs.keys()), DO_bias) if bias > 0.3]}


# input the structure of the structure of itself!!!
def find_counterpart(text, structure, PRONOUN):
    counterpart = []
    temp = text.split(' ')
    counterpart = temp[0:3]
    if PRONOUN: # pronoun version
        if structure == 'DO':
            counterpart.extend(temp[4:6])
            counterpart.append(Verbs[Verbs_inv[temp[2]]]['prep'])  
            counterpart.append(temp[3])      
        elif structure == 'PD':
            counterpart.extend(temp[6:7])
            counterpart.extend(temp[3:5])
        else:
            print("Wrong structure input.")
    else: # Det-N version
        if structure == 'DO':
            counterpart.extend(temp[5:7])
            counterpart.append(Verbs[Verbs_inv[temp[2]]]['prep'])        
        elif structure == 'PD':
            counterpart.extend(temp[6:8])
        else:
            print("Wrong structure input.")
        counterpart.extend(temp[3:5])
    counterpart.append('.')
    return ' '.join(counterpart)


def open_result_as_df(model_name, PRIMING=True, PRONOUN=False):
    file_Priming = 'Priming' if PRIMING else 'NoPriming'
    file_Pronoun = 'Pronoun' if PRONOUN else 'NoPronoun'
    
    path = f'{ROOT_DIR}/results/GPT2/GPT2-{model_name}_{file_Pronoun}_{file_Priming}.csv'
    df = pd.read_csv(path)
    
    return df

def find_verbset_name(verbs):
    if verbs == None: return 'All'
    elif verbs == ['SHOW', 'FEED', 'BUY', 'DELIVER', 'PROMISE']: return 'DO_biased_Model'
    elif verbs == [verb for verb, bias in zip(list(Verbs.keys()), DO_bias) if bias > 0]: return 'DO_biased_Corpus'
    elif verbs == [verb for verb, bias in zip(list(Verbs.keys()), DO_bias) if bias < 0.1]: return 'Large_PD_biased'
    elif verbs == [verb for verb, bias in zip(list(Verbs.keys()), DO_bias) if bias > 0.3]: return 'Large_DO_biased'
    else: return 'NOT FOUND'
    
    
    
def get_verb_bias(MODEL, SIZE, PRONOUN):
    file_Priming = 'NoPriming'
    file_Pronoun = 'Pronoun' if PRONOUN else 'NoPronoun'
    input_path = f'{ROOT_DIR}/results/{MODEL}/{MODEL}-{SIZE}_{file_Pronoun}_{file_Priming}.csv'
    if MODEL == 'Llama' or MODEL == 'GPT2':
        prob_column = 'sen_prob_nospace'
    elif MODEL == 'GPT3':
        prob_column = 'log_prob'
    else:
        print("Wrong model name.")
        return None
    
    df = pd.read_csv(input_path)
    
    PD_biases = [] # list of PD biases as the absolute probability: P(PD) / (P(PD) + P(DO)
    PD_biases_ratio = [] # list of PD biases as the ratio of the two log probabilities: log_P(PD) / log_P(DO) = log_P baseDO (PD)

    for verb in Verbs:
        # get the subdataframe for the current verb
        df_verb = df[df['verb'] == verb]

        # check that each sentence does have a counterpart
        for _, row in df_verb.iterrows():
            if not df_verb['text'].isin([find_counterpart(row['text'], row['structure'], PRONOUN)]).any():
                print(row['text'])
                break

        
        temp = 0
        temp_ratio = 0
        for _, row in df_verb[df_verb['structure'] == 'PD'].iterrows():
            # summing absolute probs (by taking exp) for all DO sentences and PD sentences
            PD_prob = torch.exp(torch.tensor(row[f'{prob_column}']))
            DO_prob = torch.exp(torch.tensor(df_verb[df_verb['text'] == find_counterpart(row['text'], row['structure'], PRONOUN)][f'{prob_column}'].values[0]))
            temp += PD_prob / (PD_prob + DO_prob)
            
            # compute the ratio of the two log probabilities
            PD_log_prob = row[f'{prob_column}']
            DO_log_prob = df_verb[df_verb['text'] == find_counterpart(row['text'], row['structure'], PRONOUN)][f'{prob_column}'].values[0]
            temp_ratio += PD_log_prob / DO_log_prob
        
        PD_bias = temp / len(df_verb[df_verb['structure'] == 'PD'])
        PD_biases.append(PD_bias)
        
        PD_bias_ratio = temp_ratio / len(df_verb[df_verb['structure'] == 'PD'])
        PD_biases_ratio.append(PD_bias_ratio)
        
    return PD_biases, PD_biases_ratio


def compute_prob(df, prime_structure, target_structure, exp=True):
    prob_xx = df[f'{prime_structure}-{target_structure}']
    opp_structure = 'PD' if target_structure == 'DO' else 'DO'
    prob_xy = df[f'{prime_structure}-{opp_structure}']
    if exp:
        return (torch.exp(torch.tensor(prob_xx.values)) / (torch.exp(torch.tensor(prob_xx.values)) + torch.exp(torch.tensor(prob_xy.values)))).mean().item()
    else:
        return (prob_xx / (prob_xx + prob_xy)).mean().item()