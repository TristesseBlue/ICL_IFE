from utils import *
import spacy
import lzma
import re

def detect(sen, root):
    """
    Detects whether the input sentence contains one of the 22 ditransitive
    verbs, and if so, whether it is a DO or PD structure. The heuristics here
    also consider the passive/active voice of the sentence.

    Args:
        sen (str): The sentence to be analyzed
        root (str): The ditransitive verb that heads the DO/PD structure

    Returns:
        Tuple[str, bool, str]: (DO/PD/Neither, whether there are >1 DO/PD
            structures, passive/active)
    """
    for token in nlp(sen):
        if token.text==root:
            deps = [child.dep_ for child in token.children]
            dobj = [idx for idx, val in enumerate(deps) if val == 'dobj']
            dative = [idx for idx, val in enumerate(deps) if val == 'dative']
            dobj_min = min(dobj) if len(dobj)>0 else -1
            dative_min = min(dative) if len(dative)>0 else -1

            multiple = 1 if len(dobj) > 1 or len(dative)>1 else 0

            if 'nsubjpass' in deps: # passive
                voice = 'passive'
                if len(dobj)>0 and len(dative)==0: return 'DO', multiple, voice
                elif len(dative)>0 and len(dobj)==0: return 'PD', multiple, voice
                else: return 'Neither', multiple, voice
            else: # active
                voice = 'active'
                if dobj_min>=0 and dative_min>=0:
                    if dobj_min < dative_min: return 'PD', multiple, voice
                    elif dobj_min > dative_min: return 'DO', multiple, voice
                else: return 'Neither', multiple, voice               
    
def plot_verb_bias(data, dataset_name):
    """
    Plot the verb bias for the dataset and save the png file

    Args:
        data (dict): a dictionary containing DO and PD counts for each verb
        dataset_name (str): name of the saved png file
    """
    verbs = list(data.keys())
    DO = [data[key]['DO'] for key in data]
    PD = [data[key]['PD'] for key in data]
    DO_ratio = [d/(d+p) for d, p in zip(DO, PD)]
    PD_ratio = [p/(d+p) for d, p in zip(DO, PD)]

    # Define the width of the bars and the index for the items
    bar_width = 0.25
    index = np.arange(len(verbs))
    plt.figure(figsize = (20,5))

    # Create the bar plot
    plt.bar(index, DO_ratio, bar_width, label='DO ratio')
    plt.bar(index + bar_width, PD_ratio, bar_width, label='PD ratio')

    # Labeling and customization
    plt.xlabel('Ditransitive Verbs')
    plt.ylabel('Ratio of each structure')
    plt.title('Verb Bias')
    plt.xticks(index + bar_width / 2, verbs)
    plt.legend()

    # Show and save plot
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{ROOT_DIR}/plots/corpus_verb_bias/{dataset_name}Verbs.png')            
                
if __name__ == "__main__":
    ########################################################################
    # parse arguments
    ########################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=True)
    parser.add_argument('--plot_bias_comparison', help='Whether to plot the two verb biases from papers', type=bool, required=False, default=False)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    PLOT_BIAS_COMPARISON = args.plot_bias_comparison
    
    ########################################################################
    # set up dataset path
    ########################################################################
    # options: Wiki, OWT1, OWT2
    if dataset_name == 'Wiki':
        dataset_path = f'{ROOT_DIR}/datasets/EngWiki100M/train.txt'
    elif dataset_name == 'OWT1':
        dataset_path = f'{ROOT_DIR}/datasets/openwebtext1'
    elif dataset_name == 'OWT2':
        dataset_path = f'{ROOT_DIR}/datasets/openwebtext2'


    ########################################################################
    # clean up data
    ########################################################################
    # load the spacy model
    nlp = spacy.load("en_core_web_trf")

    # create a list of verb forms to search for
    all_verb_forms = []
    for key in Verbs:
        all_verb_forms.extend(list(Verbs[key].values())[:4])
    all_verb_forms = list(set(all_verb_forms))

    # load and clean up the dataset, and convert each sentence into a list
    if dataset_name == 'OWT1' or dataset_name == 'OWT2':
        threshold = 250 # how many .xz files are we decompressing
        lines_clean = []

        for i in range(1,threshold):
            with lzma.open(f'{dataset_path}/urlsf_subset11-{i}_data.xz', mode='rt', encoding='utf-8') as f:
                samples = f.readlines()

                lines_clean.append(samples[0][[m.start() for m in re.finditer('\x00', samples[0])][-1]+1 : -1].split(' '))

                for item in samples[1:]:
                    if item != '\n':
                        occurrences = [m.start() for m in re.finditer('\x00', item)]
                        if len(occurrences)>1:
                            lines_clean.append(item[:occurrences[0]].split(' '))
                            lines_clean.append(item[occurrences[-1]+1:-1].split(' '))
                        else:
                            lines_clean.append(item[:-1].split(' '))

        # check the total number of tokens
        print(f'Total number of sentences = {len(lines_clean)}.')
    else: # for Wiki dataset
        with open(dataset_path) as f:
            lines = f.readlines()

        lines_clean = []
        for line in lines:
            if len([tok for tok in line.split(' ')if tok=='<unk>']) <= 2:
                line = line.replace('<unk>', 'I')
                line = line.replace('<eos>', '')
                lines_clean.append(line.split(' ')[:-1])

    # check the total number of tokens
    total_tokens = 0
    for line in lines_clean:
        total_tokens += len(line)
    print(f'Total number of tokens = {total_tokens}.')
    
    # set up the total occurrence of each verbs in the dataset
    for verb in Verbs:
        count = 0
        for sen in lines_clean:
            if any(item in sen for item in list(Verbs[verb].values())[:3]): count += 1
        print(f'{verb:<10}{count}')
        Verbs[verb]['total'] = count
    print(f"Total number of sentences with target verbs = {sum(d['total'] for d in list(Verbs.values()))}")
    
    ########################################################################
    # compute the meta-information for each sentence and save as json files
    ########################################################################
    # iterate through each sentence in the dataset
    for sen in tqdm(lines_clean): 
        for token in sen:
            if token in all_verb_forms: # if a sentence has one of the target verbs
                structure, multiple, voice = detect(' '.join(sen), token) # parse it and retrieve the info
                Verbs[Verbs_inv[token]][structure] += 1
                Verbs[Verbs_inv[token]][voice] += 1
                Verbs[Verbs_inv[token]]['multiple'] += multiple

                # create a dictionary for json file
                sentence = {
                    'text': ' '.join(sen),
                    'verb_form': token,
                    'verb': Verbs_inv[token],
                    'structure': structure,
                    'voice': voice,
                    'multiple': multiple
                }

                with open(f"{ROOT_DIR}/results/corpus_verb_bias/{dataset_name}Metadata.json", "a") as outfile:
                    json.dump(sentence, outfile)
                    outfile.write('\n')
                
                break
            
    with open(f'{ROOT_DIR}/results/corpus_verb_bias/{dataset_name}Verbs.json', 'a') as out:
        json.dump(Verbs, out)
        
        
        
    ########################################################################
    # plotting the verb bias and save the figures as png files
    ########################################################################
    with open(f'{ROOT_DIR}/results/corpus_verb_bias/{dataset_name}Verbs.json', 'r') as file:
        data = json.load(file)
        
    plot_verb_bias(data, dataset_name)
    
    
    if PLOT_BIAS_COMPARISON:
        ########################################################################
        # plotting the verb bias results from Zhou & Frank (2023) for comparison
        ########################################################################
        # verb, DO, PD
        raw = [('BRING', 580, 4927),
                ('DRAW', 0, 0),
                ('FIND', 0, 0),
                ('GET', 0, 0),
                ('GVIE', 15311, 8402),
                ('KEEP', 0, 0),
                ('LEAVE', 0, 0),
                ('MAKE', 0, 0),
                ('PURCHASE', 0, 0),
                ('SAVE', 0, 0),
                ('SELL', 190, 1288),
                ('SEND', 658, 3134),
                ('SHOW', 502, 571),
                ('SUPPLY', 0, 0),
                ('DESIGN', 0, 0),
                ('TAKE', 2044, 5620),
                ('FEED', 52, 96),
                ('BUY', 0, 0),
                ('CATCH', 0, 0),
                ('DELIVER', 0, 0),
                ('THROW', 25, 222),
                ('PROMISE', 43, 43)]
        
        data = {}
        for tup in raw:
            data[tup[0]] = {'DO':tup[1], 'PD':tup[2]}
        plot_verb_bias(data, 'Zhou&Frank2023')
        
        ########################################################################
        # plotting the verb bias results from Hawkins et al. (2020) for comparison
        ########################################################################
        df = pd.read_csv(f'{ROOT_DIR}/datasets/Sinclair_corpora/Hawkins_verblists.csv')
        df_vlist = df['verb'].to_list()
        verbs = [Verbs[item]['past'] for item in Verbs]
        verb_both = [v for v in verbs if v in df_vlist]
        
        bias = []
        for verb in verb_both:
            DO_pref = df[df['verb'] == verb]['DOpreference'].mean()
            bias.append((DO_pref/100, 1-DO_pref/100))
            
        DO = [item[0] for item in bias]
        PD = [item[1] for item in bias]

        # Define the width of the bars and the index for the items
        bar_width = 0.25
        index = np.arange(len(verb_both))
        plt.figure(figsize = (10,5))

        # Create the bar plot
        plt.bar(index, DO, bar_width, label='DO ratio')
        #plt.bar_label(DO_ratio)
        plt.bar(index + bar_width, PD, bar_width, label='PD ratio')
        #plt.bar_label(PD_ratio)

        # Labeling and customization
        plt.xlabel('Ditransitive Verbs')
        plt.ylabel('Ratio of each structure')
        plt.title('Verb Bias from Hawkins et al. 2020')
        plt.xticks(index + bar_width / 2, verb_both)
        plt.legend()

        # Show plot
        plt.tight_layout()
        plt.show()
        plt.savefig(f'{ROOT_DIR}/plots/corpus_verb_bias/Hawkins2020Verbs.png')