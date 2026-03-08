from utils import *
import sys

SIZE = sys.argv[1]
PRONOUN = False if sys.argv[2].lower() == 'false' else bool(sys.argv[2])
PRIMING = False if sys.argv[3].lower() == 'false' else bool(sys.argv[3])

model_name = f'gpt2-{SIZE}'
model = HookedTransformer.from_pretrained(model_name, device=device)

# define file paths
file_Priming = 'Priming' if PRIMING else 'NoPriming'
file_Pronoun = 'Pronoun' if PRONOUN else 'NoPronoun'
input_file = f'{ROOT_DIR}/datasets/Corpus_{file_Pronoun}_{file_Priming}.csv' 
output_csv = f'{ROOT_DIR}/results/GPT2/GPT2-{SIZE}_{file_Pronoun}_{file_Priming}.csv'
output_json = f'{ROOT_DIR}/results/GPT2/GPT2-{SIZE}_{file_Pronoun}_{file_Priming}.json'

# load the dataset as a dataframe
df = pd.read_csv(input_file)

# define the indices of each region
# notice that for priming, since we are only looking at target sentences,
# the only distinction is DO vs PD, so using the same set of indices makes no difference
regions = noPrime_regions_pronoun if PRONOUN else noPrime_regions

# define a temporary storage space for results from regiosn
result = {key: [] for key in list(regions.keys())}
if PRIMING:
    df = df[['prime_sentence', 'prime_structure', 'prime_verb', 'prime_det', 'prime_prep','target_sentence', 'target_structure', 'target_verb', 'target_det', 'target_prep']]
    result['log_prob'] = []
    result['perplexity'] = []
else:
    df = df[['text', 'structure', 'verb', 'det', 'prep']]
    result['sen_prob_nospace'] = []
    result['perplexity_nospace'] = []

if not PRIMING:
    for i in tqdm(range(len(df)), desc=f'GPT2-{SIZE}, pronoun={PRONOUN}, priming={PRIMING}'):
        text = df.loc[i, 'text']
        structure = df.loc[i, 'structure']

        # get the model's prediction on each token
        toks = [' '+s for s in text.capitalize().split(' ')]
        toks[-1] = '.'
        toks[0] = toks[0].lstrip()
        token_ids = [model.to_single_token(tok) for tok in toks]

        logits = model(text.capitalize()[:-2] + '.')
        probs = nn.functional.log_softmax(logits.squeeze()[:-1, :], dim=-1)
        probs = probs[range(probs.size(0)), token_ids]

        # compute the total sen_prob and perplexity
        sen_prob = torch.sum(probs)
        perplexity = torch.exp(sen_prob) ** (-1 / len(toks))
        result['sen_prob_nospace'].append(sen_prob.item())
        result['perplexity_nospace'].append(perplexity.item())

        # take probs by region and save the corresponding log_prob by word into dictionary
        dic = {'text': text, 'structure': structure, 'verb': df.loc[i, 'verb']}
        for region in regions:
            start = regions[region][structure][0]
            end = regions[region][structure][1]
            result[region].append(torch.sum(probs[start:end]).item())
            dic[region] = [(toks[j], probs[j].item()) for j in range(start, end)]
        
        with open(output_json, "a") as outfile:
            json.dump(dic, outfile)
            outfile.write('\n')

    # saving the temp result into the dataframe and save the dataframe as a file
    for item in result:
        df[item] = result[item]
    df.to_csv(output_csv, index=False)
else:
    for i in tqdm(range(len(df)), desc=f'GPT2-{SIZE}, pronoun={PRONOUN}, priming={PRIMING}'):
        prime_text = df.loc[i, 'prime_sentence'].capitalize()[:-2]+'.'
        target_text = df.loc[i, 'target_sentence'].capitalize()[:-2]+'.'
        target_structure = df.loc[i, 'target_structure']

        prime_len = len(prime_text.split(' ')) + 1
        target_len = len(target_text.split(' ')) + 1

        toks = [' '+s for s in df.loc[i, 'target_sentence'].capitalize().split(' ')]
        toks[-1] = '.'
        #toks[0] = toks[0].lstrip()
        token_ids = [model.to_single_token(tok) for tok in toks]


        logits = model(prime_text + ' ' + target_text)
        probs = nn.functional.log_softmax(logits.squeeze()[prime_len:-1, :], dim=-1)
        probs = probs[range(probs.size(0)), token_ids]

        sen_prob = torch.sum(probs)
        perplexity = torch.exp(sen_prob) ** (-1 / target_len)
        result['log_prob'].append(sen_prob.item())
        result['perplexity'].append(perplexity.item())

        # take probs by region and save the corresponding log_prob by word into dictionary
        dic = {'prime_sentence': df.loc[i, 'prime_sentence'], 
            'target_sentence': df.loc[i, 'target_sentence'], 
            'prime_structure': df.loc[i, 'prime_structure'],
            'target_structure': df.loc[i, 'target_structure'],
            'prime_verb': df.loc[i, 'prime_verb'],
            'target_verb': df.loc[i, 'target_verb']}
        for region in regions:
            start = regions[region][target_structure][0]
            end = regions[region][target_structure][1]
            result[region].append(torch.sum(probs[start:end]).item())
            dic[region] = [(toks[j], probs[j].item()) for j in range(start, end)]
        
        with open(output_json, "a") as outfile:
            json.dump(dic, outfile)
            outfile.write('\n')


    # saving the temp result into the dataframe and save the dataframe as a file
    for item in result:
        df[item] = result[item]
    df.to_csv(output_csv, index=False)