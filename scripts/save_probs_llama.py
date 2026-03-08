import sys
from utils import *
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

# Adding argument from command lines
SIZE = sys.argv[1] # 7b, 13b, 7b-chat
PRONOUN = False if sys.argv[2].lower() == 'false' else bool(sys.argv[2])
PRIMING = False if sys.argv[3].lower() == 'false' else bool(sys.argv[3])
STARTING_IDX = int(sys.argv[4])

PERIOD_IDX = 29889
BATCH_SIZE = int(92400 / 25)

# Set up input and output files
file_Priming = 'Priming' if PRIMING else 'NoPriming'
file_Pronoun = 'Pronoun' if PRONOUN else 'NoPronoun'
input_file = f'{ROOT_DIR}/datasets/Corpus_{file_Pronoun}_{file_Priming}.csv' 
output_dir = f'{ROOT_DIR}/results/Llama/Llama-{SIZE}_{file_Pronoun}_{file_Priming}_{STARTING_IDX}.csv'

# load the input data as df
df = pd.read_csv(input_file)
END_IDX = min((STARTING_IDX + BATCH_SIZE), df.shape[0])
col_names = ['prime_sentence', 'prime_structure', 'prime_verb', 'prime_det', 'prime_prep','target_sentence', 'target_structure', 'target_verb', 'target_det', 'target_prep'] if PRIMING else ['text', 'structure', 'verb', 'det', 'prep']
data = []

# load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(f'meta-llama/Llama-2-{SIZE}-hf', use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(f'meta-llama/Llama-2-{SIZE}-hf', use_auth_token=True, device_map='auto', offload_folder="offload")


if not PRIMING:
    for i in tqdm(range(STARTING_IDX, END_IDX), desc=f'LLAMA-{SIZE}, pronoun={PRONOUN}, priming={PRIMING}'):
        # prepare the dictionary to write the result to
        result = {key: df.loc[i, key] for key in col_names}
    
        # compute log probability
        text = df.loc[i, 'text']
        text = text.capitalize()[:-2]+'.'
        tokens = tokenizer.tokenize(text, add_special_tokens=True)
        token_ids = tokenizer(text, add_special_tokens=True, return_tensors='pt')["input_ids"]
        output = model(token_ids)["logits"]
        log_probs = F.log_softmax(output, dim=-1)
    
        # put each token's log_prob into a list
        next_word_logs = []
        for i, distr in enumerate(log_probs[0]):
            if i == log_probs.size()[1]-1:
                continue
            log_prob = distr[token_ids[0][i+1]]
            next_word_logs.append(log_prob.item())
    
        # write the tokens, probs, sen_prob into the dictionary
        result['sen_prob_nospace'] = torch.tensor(next_word_logs).sum().item()
        result["tokens"] = "|".join(tokens[1:])
        result["log_prob_tokens"] = next_word_logs
        data.append(result)
else:
    for i in tqdm(range(STARTING_IDX, END_IDX), desc=f'LLAMA-{SIZE}, pronoun={PRONOUN}, priming={PRIMING}'):
        # prepare the dictionary to write the result to
        result = {key: df.loc[i, key] for key in col_names}
    
        # compute log probability
        prime_text = df.loc[i, 'prime_sentence'].capitalize()[:-2]+'.'
        target_text = df.loc[i, 'target_sentence'].capitalize()[:-2]+'.'
        text = prime_text + " " + target_text
        
        tokens = tokenizer.tokenize(text, add_special_tokens=True)
        token_ids = tokenizer(text, add_special_tokens=True, return_tensors='pt')["input_ids"]
        output = model(token_ids)["logits"]
        log_probs = F.log_softmax(output, dim=-1)
    
        # this is the idx of the period in token_ids, which contains one extra <s> token
        # thus, this is actually the correct index where the target sentence logits start
        target_idx = np.where(token_ids[0] == PERIOD_IDX)[0][0]
    
        # put each token's log_prob into a list
        next_word_logs = []
        for i, distr in enumerate(log_probs[0][target_idx:]):
            if i == log_probs[:, target_idx:, :].size()[1]-1:
                continue
            log_prob = distr[token_ids[0][target_idx:][i+1]]
            next_word_logs.append(log_prob.item())
    
        # write the tokens, probs, sen_prob into the dictionary
        result['log_prob'] = torch.tensor(next_word_logs).sum().item()
        result["tokens"] = "|".join(tokens[1:])
        result["log_prob_tokens"] = next_word_logs
        data.append(result)

# write to file
with open(output_dir, "w") as outputf:
    writer = csv.DictWriter(outputf, data[0].keys())
    writer.writeheader()
    writer.writerows(data)
