import sys
from utils import *

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from dataclasses import dataclass
from typing import Optional
import gc
import seaborn as sns
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer, utils

torch.set_grad_enabled(True)


SIZE = sys.argv[1] # only 'small' here
PRONOUN = False if sys.argv[2].lower() == 'false' else bool(sys.argv[2])
START_IDX = int(sys.argv[3])
SQUARED = False if sys.argv[4].lower() == 'false' else bool(sys.argv[4])
file_Pronoun = 'Pronoun' if PRONOUN else 'NoPronoun'
file_Square = 'Squared' if SQUARED else 'NotSquared'
BS = 2000
MAX_IDX = 40036 if PRONOUN else 24392

model_name = f'gpt2-{SIZE}'
model = HookedTransformer.from_pretrained(model_name, device=device)

print(torch.cuda.get_device_properties(0).name)
print(torch.cuda.get_device_properties(0).total_memory/1000000)
filler_path = f'{ROOT_DIR}/fine-datasets/FT_fillers/fillers_mid.txt'

with open(filler_path, 'r') as f:
    fillers = [line.strip() for line in f.readlines()]
print(f'The set of {len(fillers)} fillers.')

with torch.no_grad():
    raw_loss = model(fillers, return_type="loss").detach()

# Adapted from TransformerLens
@dataclass
class HookedTransformerTrainConfig:
    num_epochs: int
    batch_size: int
    lr: float = 1e-5
    lamb: float = 0.0
    seed: int = 0
    
    momentum: float = 0.0
    max_grad_norm: Optional[float] = None
    weight_decay: Optional[float] = None
    optimizer_name: str = "Adam"
    
    device: Optional[str] = None
    warmup_steps: int = 0
    save_every: Optional[int] = None
    save_dir: Optional[str] = None

    print_every: Optional[int] = None
    max_steps: Optional[int] = None

def train(
    model: HookedTransformer,
    config: HookedTransformerTrainConfig,
    dataset: Dataset,
    raw_loss: float
) -> HookedTransformer:
    torch.manual_seed(config.seed)
    model.train()

    if config.device is None:
        config.device = utils.get_device()

    if config.optimizer_name in ["Adam", "AdamW"]:
        # Weight decay in Adam is implemented badly, so use AdamW instead (see PyTorch AdamW docs)
        if config.weight_decay is not None:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=config.lr,
            )
    elif config.optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
            if config.weight_decay is not None
            else 0.0,
            momentum=config.momentum,
        )
    else:
        raise ValueError(f"Optimizer {config.optimizer_name} not supported")

    scheduler = None
    if config.warmup_steps > 0:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / config.warmup_steps),
        )

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model.to(config.device)
    for epoch in range(1, config.num_epochs + 1):
        samples = 0
        for step, batch in enumerate(dataloader):
            tokens = batch["tokens"].to(config.device)
            loss = model(tokens, return_type="loss")
            
            loss_reg = model(fillers, return_type="loss")
            if SQUARED:
                loss += config.lamb * ((loss_reg - raw_loss)**2)
            else:
                loss += config.lamb * (loss_reg - raw_loss)
            
            loss.backward()
            
            if config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            if config.warmup_steps > 0:
                assert scheduler is not None
                scheduler.step()
            optimizer.zero_grad()

            samples += tokens.shape[0]

            if config.print_every is not None and step % config.print_every == 0:
                print(f"Epoch {epoch} Samples {samples} Step {step} Loss {loss.item()}")

            if (
                config.save_every is not None
                and step % config.save_every == 0
                and config.save_dir is not None
            ):
                torch.save(model.state_dict(), f"{config.save_dir}/model_{step}.pt")

            if config.max_steps is not None and step >= config.max_steps:
                break
                
        torch.cuda.empty_cache()
        gc.collect()

    return model

class FineTuningDataset(Dataset):
    def __init__(self, data, max_length=15, padding=False):
        self.data = data
        self.max_length = max_length
        self.pad_idx = 50256
        self.padding = padding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.padding:
            return {'tokens': model.to_tokens(self.data[idx]).squeeze(0)}
        else:
            temp = model.to_tokens(self.data[idx]).tolist()[0]
            temp.extend([50256] * (self.max_length - len(temp)))
            toks = torch.tensor(temp)
            return {'tokens': toks}

def get_log_prob(sentence, output):
    probs = nn.functional.log_softmax(output.squeeze()[:-1, :], dim=-1)
    probs = probs[range(probs.size(0)), model.to_tokens(sentence).squeeze(0).tolist()[1:]]
    return torch.sum(probs).item(), list(probs.cpu().detach().numpy())

# prime is a single sentence
def finetuning(raw_loss, prime):
    raw_model = HookedTransformer.from_pretrained(model_name, device=device)
            
    train_config = HookedTransformerTrainConfig(
        num_epochs=10,
        batch_size=1,
        lr=1e-5,
        lamb = 0.8,
        optimizer_name="AdamW")
    
    dataset = FineTuningDataset([prime])
    adapted_model = train(raw_model, train_config, dataset, raw_loss)

    return adapted_model


# MAIN
df = pd.read_csv(f'{ROOT_DIR}/datasets/Corpus_{file_Pronoun}_Priming.csv')
grouped = df.groupby('prime_sentence')

cols = ['prime_sentence', 'target_sentence', 'prime_structure',
       'target_structure', 'prime_verb', 'target_verb', 'prime_det',
       'target_det', 'prime_prep', 'target_prep', 'log_prob', 'probs']
output_path = f'{ROOT_DIR}/fine-results/FT/GPT2-{SIZE}/{file_Pronoun}/probs_{SQUARED}-{START_IDX}.csv'
pd.DataFrame(columns=cols).to_csv(output_path, index=False)

for prime in tqdm(list(grouped.groups.keys())[START_IDX : min(START_IDX+BS, MAX_IDX)]):
    adapted_model = finetuning(raw_loss, prime.capitalize()[:-2]+'.')
    
    for _, row in grouped.get_group(prime).iterrows():
        target_sen = row['target_sentence'].capitalize()[:-2]+'.'
        output = adapted_model(target_sen)
        log_prob, probs = get_log_prob(target_sen, output)
        row['log_prob'] = log_prob
        row['probs'] = probs
        with open(output_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(row.values.flatten())
    
    torch.cuda.empty_cache()
    gc.collect()





