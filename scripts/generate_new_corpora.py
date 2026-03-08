from utils import *


def check_diff(target, prime, tolerate=True):
    """
    Return True if there is no overlap between the two sentences

    Args:
        target (str): the target sentence
        prime (str): the prime sentence
        tolerate (bool, optional): doesn't compare prepositions if True. Defaults to True.

    Returns:
        _type_: _description_
    """
    target = set(target.split()[:-1])
    prime = set(prime.split()[:-1])
    if not tolerate:
        return len(target.intersection(prime))==0
    else:
        intersect = [x for x in list(prime.intersection(target)) if x not in ['to', 'for']]
        return len(intersect) == 0
    
def convert_PD_to_Pronoun(sen):
    """
    Convert a PD sentence to its Pronoun version by replacing the indirect object by a pronoun according to the frequency distribution of the target pronouns

    Args:
        sen (str): the input PD sentence

    Returns:
        str: the output Pronoun sentence
    """
    temp = sen.split(' ')[:-3]
    temp.extend([random.choices(Pronouns, freq, k=1)[0], '.'])
    return ' '.join(temp)

if __name__ == "__main__":
    corpus = pd.read_csv(f'{ROOT_DIR}/datasets/Sinclair_corpora/CORE_dative.csv')
    do_sen = list(corpus[' tdo'].unique())
    do_sen.extend(list(corpus[' pdo'].unique()))
    pd_sen = list(corpus[' tpo'].unique())
    pd_sen.extend(list(corpus['ppo'].unique()))
    for sen in pd_sen:
        do_sen.append(find_counterpart(sen, 'PD', False))
    unique_do_sen = list(set(do_sen))

    Det_inv = {'the': 'THE', 'a': 'A', 'an': 'A'}
    temp = []
    for sen in unique_do_sen:
        temp.append([sen, Verbs_inv[sen.split()[2]], Det_inv[sen.split()[0]]])
    temp_df = pd.DataFrame(temp, columns=['text', 'verb', 'det'])
    gb = temp_df.groupby('verb')
    
    ########################################################################
    # Construct (25*2) * 21 * 22 = 23100 pairs of (Prime, Target) Sentences
    ########################################################################
    prime_sentences = []
    target_sentences = []

    for target_verb in tqdm(Verbs):
        verb_group = gb.get_group(target_verb)
        A_samples = verb_group[verb_group['det']=='A']['text'].drop_duplicates().sample(25).values
        T_samples = verb_group[verb_group['det']=='THE']['text'].drop_duplicates().sample(25).values
        
        for verb in [x for x in Verbs if x != target_verb]:
            A_primes = []
            T_primes = []
            
            verb_group = gb.get_group(verb)
            A_group = verb_group[verb_group['det']=='A']
            T_group = verb_group[verb_group['det']=='THE']
            
            for target in A_samples:
                valid = [x for x in T_group['text'].values if check_diff(target, x)]
                if len(valid) > 0: A_primes.append(random.choice(valid))
                else:
                    print(f'No valid prime for target sentence {target}.')
                    break
            
            for target in T_samples:
                valid = [x for x in A_group['text'].values if check_diff(target, x)]
                if len(valid) > 0: T_primes.append(random.choice(valid))
                else:
                    print(f'No valid prime for target sentence {target}.')
                    break
        
            prime_sentences.extend(A_primes)
            prime_sentences.extend(T_primes)
            target_sentences.extend(list(A_samples))
            target_sentences.extend(list(T_samples))


    ########################################################################
    # Construct the full 23100*4 (Prime, Target) pairs with Structures
    ########################################################################
    df_list = []
    for prime, target in zip(prime_sentences,target_sentences):
        prime_verb = Verbs_inv[prime.split()[2]]
        target_verb = Verbs_inv[target.split()[2]]
        prime_det = Det_inv[prime.split()[0]]
        target_det = Det_inv[target.split()[0]]
        
        df_list.append([prime, target, 'DO', 'DO', prime_verb, target_verb, prime_det, target_det])
        df_list.append([prime, find_counterpart(target, 'DO', False), 'DO', 'PD', prime_verb, target_verb, prime_det, target_det])
        df_list.append([find_counterpart(prime, 'DO', False), target, 'PD', 'DO', prime_verb, target_verb, prime_det, target_det])
        df_list.append([find_counterpart(prime, 'DO', False), find_counterpart(target, 'DO', False), 'PD', 'PD', prime_verb, target_verb, prime_det, target_det])
        
    df = pd.DataFrame(df_list, columns=['prime_sentence', 'target_sentence', 'prime_structure', 'target_structure', 'prime_verb', 'target_verb', 'prime_det', 'target_det'])

    ########################################################################
    # Save it as the basis file
    ########################################################################
    df.to_csv(f'{ROOT_DIR}/datasets/Corpus_noPronoun.csv', index=False)


    ########################################################################
    # Prepare Pronoun-relevant Information
    ########################################################################
    Pronouns = ['me', 'you', 'us', 'him', 'them', 'her', 'it']

    corpus = []
    with open(f'{ROOT_DIR}/results/corpus_verb_bias/OWT1Metadata.json', 'r') as file:
        for line in file:
            corpus.append(json.loads(line))
    with open(f'{ROOT_DIR}/results/corpus_verb_bias/OWT2Metadata.json', 'r') as file:
        for line in file:
            corpus.append(json.loads(line))
    df = pd.DataFrame(corpus)
    dfDO = df[(df['structure']=='DO') & (df['voice']=='active')]

    first = []
    second = []
    tup = []
    for _, row in dfDO.iterrows():
        toks = row['text'].split(' ')
        idx = toks.index(row['verb_form'])
        first.append(toks[idx+1])
        if len(toks) > idx+2:
            second.append(toks[idx+2]) 
            tup.append((toks[idx+1], toks[idx+2])) 

    counter = Counter(tup).most_common(200)
    freq = []
    for pro in Pronouns:
        occ = 0
        for item in counter:
            if item[0][0]==pro: occ+=item[1]
        freq.append(occ)

    ########################################################################
    # Get the pool of unique PD sentences
    ########################################################################
    # load the original sentences
    corpus = pd.read_csv(f'{ROOT_DIR}/datasets/Sinclair_corpora/CORE_dative.csv')

    # get all the unique sentences in PD structure
    do_sen = list(corpus[' tdo'].unique())
    do_sen.extend(list(corpus[' pdo'].unique()))
    pd_sen = list(corpus[' tpo'].unique())
    pd_sen.extend(list(corpus['ppo'].unique()))
    for sen in do_sen:
        pd_sen.append(find_counterpart(sen, 'DO', False))
    unique_pd_sen = list(set(pd_sen))

    # get a version of unqiue PD sentences with Pronoun
    pd_sen_pronoun = [convert_PD_to_Pronoun(sen) for sen in unique_pd_sen]
    unique_pd_sen_pronoun = list(set(pd_sen_pronoun))


    ########################################################################
    # For each target sentence, find a valid prime sentence
    ########################################################################
    # create groupby objects for the unique sentences
    temp = []
    for sen in unique_pd_sen:
        temp.append([sen, Verbs_inv[sen.split()[2]], Det_inv[sen.split()[0]]])
    temp_df = pd.DataFrame(temp, columns=['text', 'verb', 'det'])
    gb = temp_df.groupby('verb')

    temp = []
    for sen in unique_pd_sen_pronoun:
        temp.append([sen, Verbs_inv[sen.split()[2]], Det_inv[sen.split()[0]]])
    temp_df = pd.DataFrame(temp, columns=['text', 'verb', 'det'])
    gb_pronoun = temp_df.groupby('verb')

    ########################################################################
    # Generate pairs of (Prime, Target) Sentences
    ########################################################################
    prime_sentences = []
    target_sentences = []
    prime_sentences_pronoun = []
    target_sentences_pronoun = []

    # iterate through each target verb
    for target_verb in tqdm(Verbs):
        # STEP 1: sample 25 A and 25 THE sentences for the target verb
        verb_group = gb.get_group(target_verb)
        A_samples = verb_group[verb_group['det']=='A']['text'].drop_duplicates().sample(25).values
        T_samples = verb_group[verb_group['det']=='THE']['text'].drop_duplicates().sample(25).values
        # get the pronoun version of the samples
        verb_group_pronoun = gb_pronoun.get_group(target_verb)
        A_samples_pronoun = [convert_PD_to_Pronoun(sen) for sen in A_samples]
        T_samples_pronoun = [convert_PD_to_Pronoun(sen) for sen in T_samples]
        
        # STEP 2: iterate through each prime verb
        for verb in [x for x in Verbs if x != target_verb]:
            A_primes = []
            T_primes = []
            A_primes_pronoun = []
            T_primes_pronoun = []
            
            # STEP 2.1: fetch the prime verb subgroups for A and THE
            verb_group = gb.get_group(verb)
            A_group = verb_group[verb_group['det']=='A']
            T_group = verb_group[verb_group['det']=='THE']
            # do the same thing for pronoun version
            verb_group_pronoun = gb_pronoun.get_group(verb)
            A_group_pronoun = verb_group_pronoun[verb_group_pronoun['det']=='A']
            T_group_pronoun = verb_group_pronoun[verb_group_pronoun['det']=='THE']
            
            # STEP 2.2: find the valid prime sentences for each target sentence
            for target, target_pronoun in zip(A_samples, A_samples_pronoun):
                valid = [x for x in T_group['text'].values if check_diff(target, x)]
                if len(valid) > 0: prime = random.choice(valid)
                else:
                    print(f'No valid prime for target sentence {target}.')
                    break
                prime_pronoun = convert_PD_to_Pronoun(prime)
                while not check_diff(target_pronoun, prime_pronoun):
                    prime_pronoun = convert_PD_to_Pronoun(prime)
                A_primes.append(prime)
                A_primes_pronoun.append(prime_pronoun)
            
            for target, target_pronoun in zip(T_samples, T_samples_pronoun):
                valid = [x for x in A_group['text'].values if check_diff(target, x)]
                if len(valid) > 0: prime = random.choice(valid)
                else:
                    print(f'No valid prime for target sentence {target}.')
                    break
                prime_pronoun = convert_PD_to_Pronoun(prime)
                while not check_diff(target_pronoun, prime_pronoun):
                    prime_pronoun = convert_PD_to_Pronoun(prime)
                T_primes.append(prime)
                T_primes_pronoun.append(prime_pronoun)
        
            # STEP 2.3: append the valid prime and target sentences to the lists
            prime_sentences.extend(A_primes)
            prime_sentences.extend(T_primes)
            prime_sentences_pronoun.extend(A_primes_pronoun)
            prime_sentences_pronoun.extend(T_primes_pronoun)

            # STEP 2.4: logging target sentences
            target_sentences.extend(list(A_samples))
            target_sentences.extend(list(T_samples))
            target_sentences_pronoun.extend(list(A_samples_pronoun))
            target_sentences_pronoun.extend(list(T_samples_pronoun))

    ########################################################################
    # Saving to .csv files
    ########################################################################
    # No Pronoun version
    df_list = []
    for prime, target in zip(prime_sentences,target_sentences):
        prime_verb = Verbs_inv[prime.split()[2]]
        target_verb = Verbs_inv[target.split()[2]]
        prime_det = Det_inv[prime.split()[0]]
        target_det = Det_inv[target.split()[0]]
        prime_prep = prime.split()[5]
        target_prep = target.split()[5]
        
        df_list.append([prime, target, 'PD', 'PD', prime_verb, target_verb, prime_det, target_det, prime_prep, target_prep])
        df_list.append([prime, find_counterpart(target, 'PD', False), 'PD', 'DO', prime_verb, target_verb, prime_det, target_det, prime_prep, target_prep])
        df_list.append([find_counterpart(prime, 'PD', False), target, 'DO', 'PD', prime_verb, target_verb, prime_det, target_det, prime_prep, target_prep])
        df_list.append([find_counterpart(prime, 'PD', False), find_counterpart(target, 'PD', False), 'DO', 'DO', prime_verb, target_verb, prime_det, target_det, prime_prep, target_prep])
        
    df = pd.DataFrame(df_list, columns=['prime_sentence', 'target_sentence', 'prime_structure', 'target_structure', 'prime_verb', 'target_verb', 'prime_det', 'target_det', 'prime_prep', 'target_prep'])
    df.to_csv(f'{ROOT_DIR}/datasets/Corpus_NoPronoun_Priming.csv', index=False)

    # Pronoun version
    df_list_pronoun = []
    for prime, target in zip(prime_sentences_pronoun,target_sentences_pronoun):
        prime_verb = Verbs_inv[prime.split()[2]]
        target_verb = Verbs_inv[target.split()[2]]
        prime_det = Det_inv[prime.split()[0]]
        target_det = Det_inv[target.split()[0]]
        prime_prep = prime.split()[5]
        target_prep = target.split()[5]
        
        df_list_pronoun.append([prime, target, 'PD', 'PD', prime_verb, target_verb, prime_det, target_det, prime_prep, target_prep])
        df_list_pronoun.append([prime, find_counterpart(target, 'PD', True), 'PD', 'DO', prime_verb, target_verb, prime_det, target_det, prime_prep, target_prep])
        df_list_pronoun.append([find_counterpart(prime, 'PD', True), target, 'DO', 'PD', prime_verb, target_verb, prime_det, target_det, prime_prep, target_prep])
        df_list_pronoun.append([find_counterpart(prime, 'PD', True), find_counterpart(target, 'PD', True), 'DO', 'DO', prime_verb, target_verb, prime_det, target_det, prime_prep, target_prep])
        
    df_pronoun = pd.DataFrame(df_list_pronoun, columns=['prime_sentence', 'target_sentence', 'prime_structure', 'target_structure', 'prime_verb', 'target_verb', 'prime_det', 'target_det', 'prime_prep', 'target_prep'])

    df_pronoun.to_csv(f'{ROOT_DIR}/datasets/Corpus_Pronoun_Priming.csv', index=False)

    ########################################################################
    # Deriving the Unprimed Corpora
    ########################################################################
    # No Pronoun version
    corpus = pd.read_csv(f'{ROOT_DIR}/datasets/Corpus_noPronoun.csv')

    targets = corpus[['target_sentence', 'target_structure', 'target_verb', 'target_det', 'target_prep']].drop_duplicates()
    targets.columns = ['sentence','structure','verb','det','prep']

    primes = corpus[['prime_sentence', 'prime_structure', 'prime_verb', 'prime_det', 'prime_prep']].drop_duplicates()
    primes.columns = ['sentence','structure','verb','det','prep']

    uniques = pd.concat([targets, primes], ignore_index=True).drop_duplicates()
    uniques.to_csv(f'{ROOT_DIR}/datasets/Corpus_NoPronoun_NoPriming.csv', index=False)

    # Pronoun version
    corpus_pronoun = pd.read_csv(f'{ROOT_DIR}/datasets/Corpus_Pronoun.csv')

    targets = corpus_pronoun[['target_sentence', 'target_structure', 'target_verb', 'target_det', 'target_prep']].drop_duplicates()
    targets.columns = ['sentence','structure','verb','det','prep']

    primes = corpus_pronoun[['prime_sentence', 'prime_structure', 'prime_verb', 'prime_det', 'prime_prep']].drop_duplicates()
    primes.columns = ['sentence','structure','verb','det','prep']

    uniques = pd.concat([targets, primes], ignore_index=True).drop_duplicates()
    uniques.to_csv(f'{ROOT_DIR}/datasets/Corpus_Pronoun_NoPriming.csv', index=False)