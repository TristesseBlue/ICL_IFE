import sys
from sklearn.metrics import r2_score, mean_squared_error
from utils import *


################################################################
# Step 1: concatenate the results of all batches into a single DF
################################################################
BatchSize = 2000
SIZE = 'small'

for PRONOUN in [True, False]:
    for SQUARED in [True, False]:
        file_Pronoun = 'Pronoun' if PRONOUN else 'NoPronoun'
        str_SQUARED = 'True' if SQUARED else 'False'
        file_Squared = 'Squared' if SQUARED else 'NotSquared'
        corpus_size = 40036 if PRONOUN else 24392
        combined_df = None
        
        for START_IDX in range(0, corpus_size, BatchSize):
            path = f'{ROOT_DIR}/results/FT/GPT2-{SIZE}/{file_Pronoun}/probs_{str_SQUARED}-{START_IDX}.csv'
            df = pd.read_csv(path)
            if START_IDX == 0:
                combined_df = df
            else:
                combined_df = pd.concat([combined_df, df], ignore_index=True)

        print(PRONOUN, SQUARED, combined_df.shape)
        combined_df.to_csv(f'{ROOT_DIR}/results/FT/FT-{file_Squared}_{file_Pronoun}_Priming.csv', index=False)
        


################################################################
# Step 2: Create intermediate representations
################################################################

for PRONOUN in [True, False]:
    for SQUARED in [True, False]:        
        file_Pronoun = 'Pronoun' if PRONOUN else 'NoPronoun'
        file_Squared = 'Squared' if SQUARED else 'NotSquared'
        
        input_path = f'{ROOT_DIR}/results/FT/FT-{file_Squared}_{file_Pronoun}_Priming.csv'
        df = pd.read_csv(input_path)
        df_sub = df[ (df['prime_structure']=='DO') & (df['target_structure']=='DO') ]
        df_stats = df_sub[['prime_sentence', 'prime_verb', 'prime_det', 'prime_prep', 'target_sentence', 'target_verb', 'target_det', 'target_prep','log_prob']]

        df_stats["PD-PD"] = np.nan
        df_stats["PD-DO"] = np.nan
        df_stats["DO-PD"] = np.nan
        df_stats["DO-DO"] = np.nan

        # compute log_prob of the 4 structures of Priming
        for idx, row in tqdm(df_stats.iterrows()):
            prime_PD = find_counterpart(row['prime_sentence'], 'DO', PRONOUN)
            target_PD = find_counterpart(row['target_sentence'], 'DO', PRONOUN)

            df_stats.loc[idx, 'PD-PD'] = df[(df['prime_sentence']==prime_PD) & (df['target_sentence']==target_PD)]['log_prob'].values[0]
            df_stats.loc[idx, 'PD-DO'] = df[(df['prime_sentence']==prime_PD) & (df['target_sentence']==row['target_sentence'])]['log_prob'].values[0]
            df_stats.loc[idx, 'DO-PD'] = df[(df['prime_sentence']==row['prime_sentence']) & (df['target_sentence']==target_PD)]['log_prob'].values[0]
            df_stats.loc[idx, 'DO-DO'] = row['log_prob']

        # save the result
        df_stats = df_stats.drop(columns=['log_prob'])
        df_stats.to_csv(f'{ROOT_DIR}/results/FT/FT-{file_Squared}_{file_Pronoun}_IntermediatePriming.csv', index=False)
        
###################################################################################
# Note: the IFE_FT, IFE_FT_Stats, IFE_FT_perTarget, IFE_FT_perTarget_Stats files are computed through the intermediate_representation.py, where the input arguments are:

# MODEL = 'FT'
# SIZES = ['Squared, 'NotSquared']
# PRONOUN = [True, False]       
###################################################################################    