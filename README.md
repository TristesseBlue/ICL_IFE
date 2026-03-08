This repository contains the datasets, scripts, and plots for paper [Is In-Context Learning a Type of Error-Driven Learning? Evidence from the Inverse Frequency Effect in Structural Priming](https://aclanthology.org/2025.naacl-long.586/), published on the main-track in NAACL 2025. All questions should correspond to Zhenghao Herbert Zhou, `herbert.zhou@yale.edu`.


# Structure of this Repository

- **`datasets/`: contains data for running inferences on LLMs, helper data stored for multiple reference, and corpora under investigation for naturalistic distribution of verb biases.**
  - `Sinclair_corpora/` and `Sinclair_vocab/` contains raw data from [Sinclair et al. (2022)](https://aclanthology.org/2022.tacl-1.60/), used to derive the current dataset;
  - `FT_fillers` contains the filler sentences used for fine-tuning GPT2-small with a single prime sentence;
  - `Corpus_*.csv` are the adapted datasets for the current main study, where `NoPriming` files contain the metadata of each raw sentence and `Priming` files contain concatenated prime and target sentences along with the metadata of the pair. Both `Pronoun` (replacing the indirect object with a pronoun) and `NoPronoun` versions are presented.

- **`results/`: contains interemediate data files along the study that are used in computing the final results, statistical tests, and plotting;**
  - `GPT2/`, `GPT3/`, `Llama/`, and `LSTM/` contain intermeidate results of the 4 model familys of computing priming effects and the IFE;
  - `FT/` contains the intermediate results of the Fine-tuning mode of priming on GPT2-small;
  - `corpus_verb_bias/` contains computed occurrence information of relavant dative strucutres in each corpus.

- **`scripts/`: contains code that runs all studies:**
  - ***Main Priming and IFE experiments:***
    - `save_probs_*.py`: scripts for running inferences of a language model on a designated dataset and recording the probabilities;
    - `intermediate_representations.py`: computing data aggregations for later IFE and statistical analaysis computation;
  - ***Fine-tuning mode experiments:***
    - `FT_hyperparam_search.py`: the script for finding the set of hyperparameters to fine-tune GPT2-small on one single prime sentence (with fillers);
    - `FT_save_probs.py` and `FT_intermediate_representations.py`: same as above, running inference on fine-tuned models and aggregate data for later computation;
  - ***Helper Scripts***
    - `generate_new_corpora.py`: the script for generating the `datasets/Corpus_*.csv` files from the raw files from Sinclair et al. (2022), including sampling logic of replacing noun phrases with pronouns;
    - `compute_corpus_verb_bias.py`: the script for counting the number of occurrances of dative structures and verbs in a given corpus;
    - `utils.py`: contains pre-defined variables and helper functions for most other scripts;
    - `plotting.py`: the scripts for generating the main figures in the paper.
- **`plots/`: contains plots and jupyter notebooks for generating those plots.**

# Environment Setup
To replicate the study, the relevant conda environment requirements are included in the `environment.yml` and `requirements.txt` files. You can create the Conda environment from the former and then install pip dependencies from the latter via the following commands:

```bash
# 1) Create the conda environment
conda env create -f environment.yml

# 2) Activate it (replace <ENV_NAME> with the name in environment.yml)
conda activate <ENV_NAME>

# 3) Install pip-only dependencies
pip install -r requirements.txt
```

# Main Results 

### 1. Measuring priming effect in the Concatenation mode
1. Run the script of inference running on LLMs with appropriate arguments for model and dataset selection: e.g., `python scripts/save_probs_llama.py 13b False True 0`. 
   1. This will produce results in the corresponding directory, e.g., `results/Llama/Llama-13b_NoPronoun_Priming.csv`.
   2. Use different argument combinations to generate results for each model of the following: `NoPronoun_NoPriming`, `NoPronoun_Priming`, `Pronoun_NoPriming`, `Pronoun_Priming`.
2. Run the script of computing aggregation and generating intermediate representations for later analysis: e.g., `python scripts/intermediate_representations.py Llama 13b False`.
   1. This will produce intermediate files in the corresponding directory, e.g., `results/Llama/Llama-13b_NoPronoun_IntermediatePriming.csv`, `results/Llama/IFE_Llama_Stats`, `results/Llama/IFE_Llama_perTarget_Stats.csv`.
3. The intermediate results are used in the plotting scripts.

### 2. Fine-tuning mode of priming
1. Run the script of testing the hyperparameter configurations: e.g., `python scripts/FT_hyperparam_search.py small tiny True` (meaning fine-tuning GPT2-small with tiny filler sentences with padding).
   1. Results will be saved to directory `results/FT/Hyperparameter/Hypersearch_small_tiny_Padding.csv`.
   2. Try all argument combinations to generate full results in `results/FT/Hyperparameter/`.
2. Run the script of fine-tuning GPT2-small with selected hyperparameters and then do inference on the fine-tuned model: e.g., `python scripts/FT_save_probs.py small False 0 True` (this means fine-tuning GPT2-small and inferencing on dative sentences without pronuons, starting with batch 0, and use squared regularization loss).
3. Same as above, run the script of computing aggregation and generating intermediate representations for later analysis: `python scripts/FT_intermediate_representations.py`, producing `results/FT/IFE_FT_Stats.csv` etc. for plotting.


### 3. Corpus statistics
In this subsection, we ask whether it is able to find the dative distributions of the 22 ditransitive verbs in natural corpora that are used as part of the LLM training data. Specifically, we looked at [`EngWiki100M`](https://aclanthology.org/N18-1108/) and fragments of [`openwebtext`](https://huggingface.co/datasets/Skylion007/openwebtext). Download the two datasets and put them into the `datasets/` directory:
- For English Wikipedia, the downloaded files should be stored at `datasets/EngWiki100M/train.txt`, and the dataset also contains `valid.txt`, `test.txt`, and `vocab.txt`.
- For OpenWebText, we randomly picked two framents, stored as `datasets/openwebtext1/urlsf_subset00_1_data.xz` through `urlsf_subset00_1000_data.xz`, and `datasets/openwebtext2/urlsf_subset11_1_data.xz` through `urlsf_subset11_1000_data.xz`.

After preparing the dataset, run the following command: `python scripts/compute_corpus_verb_bias.py --dataset_name OWT1 --plot_bias_comparison True` (replace the dataset name with `OWT2` or `wiki`) to generate the verb biases in naturalistic corpus distributions, stored in `plots/corpus_verb_bias`.




