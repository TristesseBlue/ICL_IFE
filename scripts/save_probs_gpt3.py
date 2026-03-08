import sys
from utils import *
import random
import time
import os
import openai

OPENAI_API_KEY="_YOUR_OPENAI_API_KEY_"
client = openai.OpenAI(api_key=OPENAI_API_KEY)

MODEL = sys.argv[1]
PRONOUN = False if sys.argv[2].lower() == 'false' else bool(sys.argv[2])
PRIMING = False if sys.argv[3].lower() == 'false' else bool(sys.argv[3])

# Set up input and output files
file_Priming = 'Priming' if PRIMING else 'NoPriming'
file_Pronoun = 'Pronoun' if PRONOUN else 'NoPronoun'
input_file = f'{ROOT_DIR}/datasets/Corpus_{file_Pronoun}_{file_Priming}.csv' 
df = pd.read_csv(input_file)
col_names = ['prime_sentence', 'prime_structure', 'prime_verb', 'prime_det', 'prime_prep','target_sentence', 'target_structure', 'target_verb', 'target_det', 'target_prep'] if PRIMING else ['text', 'structure', 'verb', 'det', 'prep']

PERIOD_IDX = 29889
BATCH_SIZE = 100
TOTAL = df.shape[0]



STARTING_IDX = 0
END_IDX = min((STARTING_IDX + BATCH_SIZE), TOTAL)

while STARTING_IDX < TOTAL:
    print(f'Currently at {STARTING_IDX}-{END_IDX} batch.')
    output_dir = f'{ROOT_DIR}/results/GPT3/GPT3-{MODEL}_{file_Pronoun}_{file_Priming}_{STARTING_IDX}.csv'
    
    # do saveProbs
    output_dics = []
    sentences = []
    
    for i in range(STARTING_IDX, END_IDX):
        result = {key: df.loc[i, key] for key in col_names}
        output_dics.append(result)

        if PRIMING:
            prime_text = df.loc[i, 'prime_sentence'].capitalize()[:-2]+'.'
            target_text = df.loc[i, 'target_sentence'].capitalize()[:-2]+'.'
            text = '<|endoftext|>' + prime_text + " " + target_text
            sentences.append(text)
        else:
            text = df.loc[i, 'text'].capitalize()[:-2]+'.'
            sentences.append('<|endoftext|>' + text)

    try:
        print(f"Try to request log_probs for batch {STARTING_IDX}")
        response = client.completions.create(
            model=MODEL,  
            max_tokens=0,
            prompt=sentences,
            temperature=0.0,
            echo=True,
            logprobs=0,
        )
    except:
        print(f"Timeout: sleep for 10 seconds and try batch {STARTING_IDX} again")
        time.sleep(10)
        
    for i in range(END_IDX - STARTING_IDX):
        tokens = response.choices[i].logprobs.tokens[1:]
        probs = response.choices[i].logprobs.token_logprobs[1:]
        if PRIMING:
            target_idx = tokens.index('.')+1
            target_log_prob = torch.tensor(probs[target_idx:]).sum().item()
        else:
            target_log_prob = torch.tensor(probs).sum().item()
        
        output_dics[i]['tokens'] = tokens
        output_dics[i]['log_prob_tokens'] = probs
        output_dics[i]['log_prob'] = target_log_prob
    
    with open(output_dir, "w") as csv_f:
      writer = csv.DictWriter(csv_f, output_dics[0].keys())
      writer.writeheader()
      writer.writerows(output_dics)

    # update the indices
    STARTING_IDX += BATCH_SIZE
    END_IDX = min((STARTING_IDX + BATCH_SIZE), TOTAL)


