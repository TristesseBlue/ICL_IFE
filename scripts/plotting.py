import sys
import seaborn as sns
from utils import *

MODEL = sys.argv[1] # options = GPT2, GPT3, or Llama
SIZE = sys.argv[2]
PRONOUN = False if sys.argv[3].lower() == 'false' else bool(sys.argv[3])

# set up directory-related variables
file_Pronoun = 'Pronoun' if PRONOUN else 'NoPronoun'
title_Pronoun = 'with' if PRONOUN else 'without'

################################################################
# Verb Biases
################################################################
def plot_verb_bias(MODEL, SIZE, PRONOUN):
    PD_biases, _ = get_verb_bias(MODEL, SIZE, PRONOUN)

    sorted_indices = np.argsort(PD_biases)
    V_sort = [list(Verbs.keys())[i] for i in sorted_indices]
    Values_sort = [float(PD_biases[i]) for i in sorted_indices]

    #plt.bar(V_sort, Values_sort)
    sns.barplot(x=V_sort, y=Values_sort, color='#4C72B0')

    plt.xlabel('Prime Verbs', family='serif', style='normal', weight='bold', fontsize=15)
    plt.ylabel('PD Bias', family='serif', style='normal', weight='bold', fontsize=15)
    plt.ylim(0, 1)
    plt.xticks(rotation=90, family='serif', style='normal', weight='normal', fontsize=12)
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.title(f'{MODEL} Verb Bias {title_Pronoun} Pronoun', family='serif', style='normal', weight='bold', fontsize=18)
    
    plt.savefig(f'{ROOT_DIR}/plots/model_verb_biases/VerbBias_{MODEL}_{SIZE}_{file_Pronoun}.png', bbox_inches='tight')
    plt.show()
    
################################################################
# IFE
################################################################

def IFE_subplots(df, df_stats, title, ax):

    data_PD = pd.DataFrame({'x': df['pd_bias'],
                        'y': df['PD-PD'],
                        'label': df['prime_verb'],
                        'legend': 'PD prime, PD target'})

    data_DO = pd.DataFrame({'x': df['pd_bias'],
                        'y': df['DO-PD'],
                        'label': df['prime_verb'],
                        'legend': 'DO prime, PD target'})

    ########################################################################################
    # Plot PD scatter plot
    sns.regplot(x='x', y='y', data=data_PD, ax=ax, scatter_kws={"s": 100}, label=data_PD['legend'].iloc[0])
    for i in range(len(data_PD)):
        ax.text(data_PD['x'].iloc[i], data_PD['y'].iloc[i], data_PD['label'].iloc[i], fontsize=13, ha='center', va='bottom', family='serif', style='normal', weight='light', color='black')
        
    slope = df_stats['PD-PD_slope'].iloc[0]
    intercept = df_stats['PD-PD_intercept'].iloc[0]
    std_err = df_stats['PD-PD_std_err'].iloc[0]
    CI = (df_stats['PD-PD_CI_lower'].iloc[0], df_stats['PD-PD_CI_upper'].iloc[0])

    ax.text(0.05, 0.85, f"Slope: {slope}\nIntercept: {intercept}\nStd_Err: {std_err}\n95%CI: {CI}", transform=ax.transAxes, fontsize=18, ha='left', va='bottom', family='serif', style='normal', weight='normal', color='#4C72B0')
    
    ########################################################################################
    # Plot DO scatter plot
    sns.regplot(x='x', y='y', data=data_DO, ax=ax, scatter_kws={"s": 100}, label=data_DO['legend'].iloc[0])
    for i in range(len(data_DO)):
        ax.text(data_DO['x'].iloc[i], data_DO['y'].iloc[i], data_DO['label'].iloc[i], fontsize=13, ha='center', va='bottom', family='serif', style='normal', weight='light', color='black')
        
    slope = df_stats['PD-PD_slope'].iloc[0]
    intercept = df_stats['PD-PD_intercept'].iloc[0]
    std_err = df_stats['PD-PD_std_err'].iloc[0]
    CI = (df_stats['PD-PD_CI_lower'].iloc[0], df_stats['PD-PD_CI_upper'].iloc[0])

    ax.text(0.05, 0.85, f"Slope: {slope}\nIntercept: {intercept}\nStd_Err: {std_err}\n95%CI: {CI}", transform=ax.transAxes, fontsize=18, ha='left', va='bottom', family='serif', style='normal', weight='normal', color='#FFA07A')

    ######################################################################################### Set format info for this axis
    ax.legend(prop={'size': 17})
    ax.set_xlabel('PD Biases of Prime Verbs', family='serif', style='normal', weight='bold', fontsize=20)
    ax.set_ylabel('PrimeBias', family='serif', style='normal', weight='bold', fontsize=20)
    ax.set_title(title, family='serif', style='normal', weight='bold', fontsize=25)
    # ax.set_ylim(0, 0.8)
    # ax.set_xlim(0, 0.8)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=17)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=17)
    
    
    
def plot_IFE_merged(MODEL, SIZE):
    """
    Return a plot of two subplots: left = without pronoun, right = with pronoun;
    Each subplot contains two scatter plots: PD-PD and DO-PD;
    Each scatter plot is fitted with a linear regression line, and the slope, intercept, and R2 are displayed;

    Args:
        MODEL (_type_): _description_
        SIZE (_type_): _description_
        PRONOUN (_type_): _description_
    """
    # initialize the outmost fig
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # First fill in the lef axis where PRONOUN = False
    PRONOUN = False
    df = pd.read_csv(f'{ROOT_DIR}/results/{MODEL}/IFE_{MODEL}.csv')
    df = df[(df['size'] == SIZE) & (df['pronoun'] == PRONOUN)]
    df_stats = pd.read_csv(f'{ROOT_DIR}/results/{MODEL}/IFE_{MODEL}_Stats.csv')
    df_stats = df_stats[(df_stats['size'] == SIZE) & (df_stats['pronoun'] == PRONOUN)]
    
    subplot_title = f'{MODEL}-{SIZE} IFE without Pronoun'
    IFE_subplots(df, df_stats, subplot_title, axes[0])
    
    # Second fill in the right axis where PRONOUN = True
    PRONOUN = True
    df = pd.read_csv(f'{ROOT_DIR}/results/{MODEL}/IFE_{MODEL}.csv')
    df = df[(df['size'] == SIZE) & (df['pronoun'] == PRONOUN)]
    df_stats = pd.read_csv(f'{ROOT_DIR}/results/{MODEL}/IFE_{MODEL}_Stats.csv')
    df_stats = df_stats[(df_stats['size'] == SIZE) & (df_stats['pronoun'] == PRONOUN)]
    
    subplot_title = f'{MODEL}-{SIZE} IFE with Pronoun'
    IFE_subplots(df, df_stats, subplot_title, axes[1])
    
    # Finally, display the plot and save it
    #fig.suptitle(f'Inverse Frequency Effect With Pronoun', family='serif', style='normal', weight='bold', fontsize=20)
    plt.savefig(f'{ROOT_DIR}/plots/ife/IFE_{MODEL}-{SIZE}.png')
    plt.show()
    
def plot_FT_IFE_merged(MODEL, SIZE, SQUARED):
    """
    Return a plot of two subplots: left = without pronoun, right = with pronoun;
    Each subplot contains two scatter plots: PD-PD and DO-PD;
    Each scatter plot is fitted with a linear regression line, and the slope, intercept, and R2 are displayed;

    Args:
        MODEL (_type_): _description_
        SIZE (_type_): _description_
        PRONOUN (_type_): _description_
    """
    file_Squared = 'Squared' if SQUARED else 'NotSquared'
    # initialize the outmost fig
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # First fill in the lef axis where PRONOUN = False
    PRONOUN = False
    df = pd.read_csv(f'{ROOT_DIR}/results/{MODEL}/IFE_{MODEL}.csv')
    df = df[(df['size'] == SIZE) & (df['pronoun'] == PRONOUN) & (df['squared'] == SQUARED)]
    df_stats = pd.read_csv(f'{ROOT_DIR}/results/{MODEL}/IFE_{MODEL}_Stats.csv')
    df_stats = df_stats[(df_stats['size'] == SIZE) & (df_stats['pronoun'] == PRONOUN) & (df_stats['squared'] == SQUARED)]
    df_stats = df_stats.round(3)
    
    subplot_title = f'{MODEL} GPT2-{SIZE} IFE without Pronoun' #, {file_Squared}
    IFE_subplots(df, df_stats, subplot_title, axes[0])
    
    # Second fill in the right axis where PRONOUN = True
    PRONOUN = True
    df = pd.read_csv(f'{ROOT_DIR}/results/{MODEL}/IFE_{MODEL}.csv')
    df = df[(df['size'] == SIZE) & (df['pronoun'] == PRONOUN) & (df['squared'] == SQUARED)]
    df_stats = pd.read_csv(f'{ROOT_DIR}/results/{MODEL}/IFE_{MODEL}_Stats.csv')
    df_stats = df_stats[(df_stats['size'] == SIZE) & (df_stats['pronoun'] == PRONOUN) & (df_stats['squared'] == SQUARED)]
    df_stats = df_stats.round(3)
    
    subplot_title = f'{MODEL} GPT2-{SIZE} IFE with Pronoun'#, {file_Squared}
    IFE_subplots(df, df_stats, subplot_title, axes[1])
    
    # Adjust layout
    #plt.tight_layout()
    
    # CODE FOR SAVING A SUBPLOT: Save only the right subplot (axes[1]) with all labels and titles
    # Get the bounding box of the right subplot, but expand it to include labels/titles
    # extent = axes[1].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    # fig.savefig("right_subplot_with_labels.png", bbox_inches=extent.expanded(1.1, 1.1))  # Expand slightly to include labels
    
    plt.savefig(f'{ROOT_DIR}/plots/ife/IFE_{MODEL}-GPT2-{SIZE}, {file_Squared}.png')
    plt.show()
    
if __name__ == '__main__':
    # plot_FT_IFE_merged(MODEL, SIZE, PRONOUN)
    # plot_IFE_merged(MODEL, SIZE)
    # plot_verb_bias(MODEL, SIZE, PRONOUN)