# there are in total 7*6*2=84 rounds of FT, each round is evaluated on 16 sentences
import sys
from utils import *
from priming_utils import *

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

SIZE = sys.argv[1]
FILLER_SIZE = sys.argv[2]
PADDING = False if sys.argv[3].lower() == 'false' else bool(sys.argv[3])
batch_size = 10
file_Padding = 'Padding' if PADDING else 'NoPadding'

# ######################## create filler set ########################
model_name = f'gpt2-{SIZE}'
model = HookedTransformer.from_pretrained(model_name, device=device)

filler_path = f'{ROOT_DIR}/fine-datasets/FT_fillers/fillers_{FILLER_SIZE}.txt'
with open(filler_path, 'r') as f:
    fillers = [line.strip() for line in f.readlines()]
print(f'The {FILLER_SIZE} filler set has {len(fillers)} fillers.')

long_tensor = torch.tensor([]).to(device)
for filler in fillers:
    long_tensor = torch.cat((long_tensor, model.to_tokens(filler+' ')[0, 1:]))
long_tensor = torch.cat((long_tensor, model.to_tokens("One person started clapping, and then more people started")[0, 1: 11 - long_tensor.shape[0]%10]))
#print(long_tensor.shape)

long_tensor = long_tensor.reshape(batch_size, -1).to(torch.int64)
bos = torch.full((batch_size, 1), 50256).to(device)
input_tensor = torch.cat((bos, long_tensor), dim=1)
FILLERS = fillers if PADDING else input_tensor

# compute raw loss for the filler sentences, stored in a list
with torch.no_grad():
    #raw_loss = model(FILLERS, return_type="loss").detach()
    raw_loss = []
    for filler in FILLERS:
        raw_loss.append(model(filler, return_type="loss").detach())


# ######################## define relevant functions ########################
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
            loss.backward()

            for idx, filler in enumerate(FILLERS):
                loss_reg = model(filler, return_type="loss")
                loss = config.lamb * ((loss_reg - raw_loss[idx])**2) # may want to divide by 30 if we think this as averaging
                loss.backward()
            #print(torch.cuda.memory_reserved(device=device)/1000000)
            
            
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


# ######################## Prepare the prime-target pairs ########################
Prime1 = 'The teacher delivered the chocolate for the uncle.'
Target_Set1 = [('A buddy got a cup for an army.', 'A buddy got an army a cup.'),
             ('An aunt left a beer for a band.', 'An aunt left a band a beer.'),
             ('A brother made a guitar for a secretary.', 'A brother made a secretary a guitar.'),
             ('A chief saved a pot for a band.', 'A chief saved a band a pot.'),
             ('A guy showed a beer to an attorney.', 'A guy showed an attorney a beer.'),
             ('A guy supplied a salad to a corporation.', 'A guy supplied a corporation a salad.'),
             ('A guy caught a jacket for a friend.', 'A guy caught a friend a jacket.'),
             ('A mayor promised a tea to a secretary.', 'A mayor promised a secretary a tea.')]

Prime2 = 'A chief delivered a lady a cheese.'
Target_Set2 = [('The mayor gave the juice to the boy.', 'The mayor gave the boy the juice.'),
             ('The child kept the gun for the secretary.', 'The child kept the secretary the gun.'),
             ('The cop made the card for the corporation.', 'The cop made the corporation the card.'),
             ('The child supplied the beer to the corporation.', 'The child supplied the corporation the beer.'),
             ('The brother supplied the beer to the physician.', 'The brother supplied the physician the beer.'),
             ('The writer designed the machine for the bishop.', 'The writer designed the bishop the machine.'),
             ('The writer bought the pie for the brother.', 'The writer bought the brother the pie.'),
             ('The author bought the card for the man.', 'The author bought the man the card.')]

# ######################## Loops ########################
rec = []
torch.cuda.empty_cache()
gc.collect()

for LAMB in [0.1, 0.5, 1, 5, 10, 20, 50]: 
    print(f'############################ Lamb = {LAMB} ############################')
    for LR in [5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4]:#1e-7, 5e-4
        print(f'########### Lr = {LR} ###########')        
        train_config = HookedTransformerTrainConfig(
            num_epochs=10,
            batch_size=1,
            lr=LR,
            lamb = LAMB,
            optimizer_name="AdamW")

        # round 1: FT on a PD prime
        raw_model = HookedTransformer.from_pretrained(model_name, device=device)
        dataset = FineTuningDataset([Prime1])
        adapted_model = train(raw_model, train_config, dataset, raw_loss)

        tempPD = []
        tempDO = []
        for target_pair in Target_Set1:
            targetPD = target_pair[0]
            targetDO = target_pair[1]
            
            # first get baseline
            output_PD = model(targetPD)
            output_DO = model(targetDO)
            baseline_PD,_ = get_log_prob(targetPD, output_PD)
            baseline_DO,_ = get_log_prob(targetDO, output_DO)

            # then compute the difference
            output_PD = adapted_model(targetPD)
            output_DO = adapted_model(targetDO)
            sen_prob_PD,_ = get_log_prob(targetPD, output_PD)
            sen_prob_DO,_ = get_log_prob(targetDO, output_DO)

            tempPD.append(sen_prob_PD - baseline_PD)
            tempDO.append(sen_prob_DO - baseline_DO)

            torch.cuda.empty_cache()
            gc.collect()

        rec.append({'lambda': train_config.lamb,
                    'lr': train_config.lr,
                    'prime_structure': 'PD',
                    'PD': torch.tensor(tempPD).mean().item(),
                    'DO': torch.tensor(tempDO).mean().item()})


        # round 2: FT on a DO prime
        raw_model = HookedTransformer.from_pretrained(model_name, device=device)
        dataset = FineTuningDataset([Prime2])
        adapted_model = train(raw_model, train_config, dataset, raw_loss)
        
        tempPD = []
        tempDO = []
        for target_pair in Target_Set2:
            targetPD = target_pair[0]
            targetDO = target_pair[1]
            
            # first get baseline
            output_PD = model(targetPD)
            output_DO = model(targetDO)
            baseline_PD,_ = get_log_prob(targetPD, output_PD)
            baseline_DO,_ = get_log_prob(targetDO, output_DO)

            # then compute the difference
            output_PD = adapted_model(targetPD)
            output_DO = adapted_model(targetDO)
            sen_prob_PD,_ = get_log_prob(targetPD, output_PD)
            sen_prob_DO,_ = get_log_prob(targetDO, output_DO)

            tempPD.append(sen_prob_PD - baseline_PD)
            tempDO.append(sen_prob_DO - baseline_DO)

            torch.cuda.empty_cache()
            gc.collect()

        rec.append({'lambda': train_config.lamb,
                    'lr': train_config.lr,
                    'prime_structure': 'DO',
                    'PD': torch.tensor(tempPD).mean().item(),
                    'DO': torch.tensor(tempDO).mean().item()})

        
    
df = pd.DataFrame(rec)
df.to_csv(f'{ROOT_DIR}/results/FT/Hyperparameter/Hypersearch_{SIZE}_{FILLER_SIZE}_{file_Padding}.csv', index=False)







