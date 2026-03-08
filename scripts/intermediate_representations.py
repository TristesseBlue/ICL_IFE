import sys
from sklearn.metrics import r2_score , mean_squared_error
from scipy import stats
from utils import *

MODEL = sys.argv[1] # options = GPT2, GPT3, Llama, or FT
SIZE = sys.argv[2]
PRONOUN = False if sys.argv[3].lower() == 'false' else bool(sys.argv[3])

# set up directory-related variables
file_Priming = 'Priming'
file_Pronoun = 'Pronoun' if PRONOUN else 'NoPronoun'
input_path = f'{ROOT_DIR}/results/{MODEL}/{MODEL}-{SIZE}_{file_Pronoun}_{file_Priming}.csv'
intermediate_path = f'{ROOT_DIR}/results/{MODEL}/'

if MODEL == 'GPT2':
    sizes = ['small', 'medium', 'large']
elif MODEL == 'GPT3':
    sizes = ['davinci-002']
elif MODEL == 'Llama':
    sizes = ['7b', '7b-chat', '13b']
elif MODEL == 'FT':
    sizes = ['NotSquared', 'Squared']
else:
    raise ValueError('Invalid MODEL')

################################################################
# Define functions
################################################################
def confidence_interval(x, y): # x and y are np arrays
    # Perform linear regression with numpy.polyfit and return covariance matrix
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Number of data points
    n = len(x)

    # Degrees of freedom: n - 2 (because we estimated two parameters: slope and intercept)
    dof = n - 2

    # t-critical value for 95% confidence interval
    t_critical = stats.t.ppf(1 - 0.005, dof)

    # Calculate confidence intervals
    slope_CI_lower = slope - t_critical * std_err
    slope_CI_upper = slope + t_critical * std_err
    
    return slope, intercept, r_value, p_value, std_err, slope_CI_lower, slope_CI_upper

def fit_IFE(df):    
    data1 = pd.DataFrame({'x': df['pd_bias'],
                        'y': df['PD-PD'],
                        'label': df['prime_verb'],
                        'structure': 'PD-PD'})

    data2 = pd.DataFrame({'x': df['pd_bias'],
                        'y': df['DO-PD'],
                        'label': df['prime_verb'],
                        'structure': 'DO-PD'})
    
    result = {'PD-PD_slope':-100, 'PD-PD_intercept':-100, 'PD-PD_R2':-100, 'PD-PD_RMSE':-100, 'DO-PD_slope':-100, 'DO-PD_intercept':-100, 'DO-PD_R2':-100, 'DO-PD_RMSE':-100}

    # Plot each scatter plot with regression line and label each point
    for data in [data1, data2]:
        # Fit a line (perform linear regression) and calculate R-squared and RMSE
        coefficients = np.polyfit(df['pd_bias'], data['y'], 1)
        poly = np.poly1d(coefficients)
        y_pred = poly(df['pd_bias']) # Calculate the predicted values (y_pred)
        r_squared = r2_score(data['y'], y_pred) # Calculate R-squared value
        rmse = np.sqrt(mean_squared_error(data['y'], y_pred)) # Calculate Root Mean Squared Error (RMSE)
        
        # Find std_err and 95% confidence interval
        slope, intercept, r_value, p_value, std_err, slope_CI_lower, slope_CI_upper = confidence_interval(df['pd_bias'].to_numpy(), data['y'].to_numpy())
        
        # record relevant info
        result[f"{data['structure'].tolist()[0]}_slope"] = slope
        result[f"{data['structure'].tolist()[0]}_intercept"] = intercept
        result[f"{data['structure'].tolist()[0]}_std_err"] = std_err
        result[f"{data['structure'].tolist()[0]}_CI_lower"] = slope_CI_lower
        result[f"{data['structure'].tolist()[0]}_CI_upper"] = slope_CI_upper
        result[f"{data['structure'].tolist()[0]}_r_value"] = r_value
        result[f"{data['structure'].tolist()[0]}_p_value"] = p_value
        result[f"{data['structure'].tolist()[0]}_R2"] = r_squared
        result[f"{data['structure'].tolist()[0]}_RMSE"] = rmse

    return result

def fit_IFE_perTarget(df):
    result = {'PD-PD_slope':-100, 'PD-PD_intercept':-100, 'DO-PD_slope':-100, 'DO-PD_intercept':-100}
    
    # Plot each subplot
    for _, col in enumerate(['PD-PD', 'DO-PD', 'DO-DO', 'PD-DO']):
        # Fit a line (perform linear regression)
        coefficients = np.polyfit(df['pd_bias'], df[col], 1)
        slope = coefficients[0]
        intercept = coefficients[1]
        poly = np.poly1d(coefficients)

        y_pred = poly(df['pd_bias'])
        r_squared = r2_score(df[col], y_pred)
        rmse = np.sqrt(mean_squared_error(df[col], y_pred))
        
        if col == 'PD-PD' or col == 'DO-PD':
            result[f'{col}_slope'] = slope
            result[f'{col}_intercept'] = intercept
            result[f'{col}_R2'] = r_squared
            result[f'{col}_RMSE'] = rmse
            
    return result


################################################################
# Compute InterpediatePriming representations
################################################################
# load the data
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
df_stats.to_csv(intermediate_path+f'{MODEL}-{SIZE}_{file_Pronoun}_Intermediate{file_Priming}.csv', index=False)

################################################################
# compute IFE representations
################################################################
col_names = ['size', 'pronoun', 'prime_verb', 'pd_bias', 'pd_bias_ratio', 'DO-DO', 'DO-PD', 'PD-DO', 'PD-PD']
temp = []

for SIZE in sizes:
    for PRONOUN in [True, False]:
        df_stats = pd.read_csv(intermediate_path+f'{MODEL}-{SIZE}_{file_Pronoun}_Intermediate{file_Priming}.csv')
        PD_biases, PD_biases_ratio = get_verb_bias(SIZE, PRONOUN)
    
        for i, prime_verb in enumerate(Verbs):
            prime_df = df_stats.groupby('prime_verb').get_group(prime_verb)
            dodo = compute_prob(prime_df, 'DO', 'DO')
            dopd = compute_prob(prime_df, 'DO', 'PD')
            pddo = compute_prob(prime_df, 'PD', 'DO')
            pdpd = compute_prob(prime_df, 'PD', 'PD')
            temp.append([SIZE, PRONOUN, prime_verb, PD_biases[i].item(), PD_biases_ratio[i], dodo, dopd, pddo, pdpd])
            
df_bias = pd.DataFrame(temp, columns=col_names)
df_bias.to_csv(intermediate_path+f'IFE_{MODEL}.csv', index=False)

################################################################
# compute IFE_stats representations
################################################################
data = pd.read_csv(intermediate_path+f'IFE_{MODEL}.csv')
temp = []
col_names = ['size', 'pronoun',
            'PD-PD_slope', 'PD-PD_intercept', 'PD-PD_std_err', 'PD-PD_CI_lower', 'PD-PD_CI_upper', 'PD-PD_r_value', 'PD-PD_p_value', 'PD-PD_R2', 'PD-PD_RMSE',
            'DO-PD_slope', 'DO-PD_intercept', 'DO-PD_std_err', 'DO-PD_CI_lower', 'DO-PD_CI_upper', 'DO-PD_r_value', 'DO-PD_p_value', 'DO-PD_R2', 'DO-PD_RMSE']
df_IFE = pd.DataFrame(temp, columns=col_names)
    
for SIZE in sizes:
    for PRONOUN in [True, False]:
        if MODEL == 'FT':
            str_sq = True if SIZE == 'Squared' else False
            data_sub = data[(data['squared'] == str_sq) & (data['pronoun'] == PRONOUN)]
        else:
            data_sub = data[(data['size'] == SIZE) & (data['pronoun'] == PRONOUN)]
        dic = {'size':SIZE, 'pronoun':PRONOUN}
        result_dic = fit_IFE(data_sub)
        merged_dic = {**dic, **result_dic}
        merged_list = {key: [value] for key, value in merged_dic.items()}
        df_IFE = pd.concat([df_IFE, pd.DataFrame(merged_list)], ignore_index=True)
            
df_IFE.to_csv(intermediate_path+f'IFE_{MODEL}_Stats.csv', index=False)


################################################################
# compute IFE_perTarget representations
################################################################
col_names = ['size', 'pronoun', 'prime_verb', 'pd_bias', 'pd_bias_ratio', 'target_verb', 'DO-DO', 'DO-PD', 'PD-DO', 'PD-PD']
temp = []

for SIZE in sizes:
    for PRONOUN in [True, False]:
        df_stats = pd.read_csv(intermediate_path+f'{MODEL}-{SIZE}_{file_Pronoun}_Intermediate{file_Priming}.csv')
        
        PD_biases, PD_biases_ratio = get_verb_bias(SIZE, PRONOUN)
    
        for i, prime_verb in tqdm(enumerate(Verbs)):
            for target_verb in Verbs:
                if prime_verb != target_verb:
                    prime_df = df_stats.groupby('prime_verb').get_group(prime_verb).groupby('target_verb').get_group(target_verb)
                    dodo = compute_prob(prime_df, 'DO', 'DO')
                    dopd = compute_prob(prime_df, 'DO', 'PD')
                    pddo = compute_prob(prime_df, 'PD', 'DO')
                    pdpd = compute_prob(prime_df, 'PD', 'PD')
                    temp.append([SIZE, PRONOUN, prime_verb, PD_biases[i].item(), PD_biases_ratio[i], target_verb, dodo, dopd, pddo, pdpd])
            
df_bias = pd.DataFrame(temp, columns=col_names)
df_bias.to_csv(intermediate_path+f'IFE_{MODEL}_perTarget.csv', index=False)

################################################################
# compute IFE_perTarget_stats representations
################################################################
data = pd.read_csv(intermediate_path+f'IFE_{MODEL}_perTarget.csv')
temp = []
col_names = ['size', 'pronoun', 'target_verb', 'PD-PD_slope', 'PD-PD_intercept', 'DO-PD_slope', 'DO-PD_intercept']
df_IFE = pd.DataFrame(temp, columns=col_names)
    
for SIZE in sizes:
    for PRONOUN in [True, False]:
        for target_verb in Verbs:
            data_sub = data[(data['size'] == SIZE) & (data['pronoun'] == PRONOUN) & (data['target_verb'] == target_verb)]
            dic = {'size':SIZE, 'pronoun':PRONOUN, 'target_verb':target_verb}
            result_dic = fit_IFE_perTarget(data_sub)
            merged_dic = {**dic, **result_dic}
            merged_list = {key: [value] for key, value in merged_dic.items()}
            df_IFE = pd.concat([df_IFE, pd.DataFrame(merged_list)], ignore_index=True)
            
df_IFE.to_csv(intermediate_path+f'IFE_{MODEL}_perTarget_Stats.csv', index=False)