root_dir = '/vast/palmer/home.grace/zz436/neural_priming/'
import sys
sys.path.append(root_dir)
from priming_utils import *

from dataclasses import dataclass
from typing import Optional

import torch
import torch.optim as optim
import json
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from transformer_lens import HookedTransformer, utils

torch.set_grad_enabled(True)


LR = float(sys.argv[1])

################### Set up FineTuning ####################
@dataclass
class HookedTransformerTrainConfig:
    """
    Configuration class to store training hyperparameters for a training run of
    an HookedTransformer model.
    Args:
        num_epochs (int): Number of epochs to train for
        batch_size (int): Size of batches to use for training
        lr (float): Learning rate to use for training
        seed (int): Random seed to use for training
        momentum (float): Momentum to use for training
        max_grad_norm (float, *optional*): Maximum gradient norm to use for
        weight_decay (float, *optional*): Weight decay to use for training
        optimizer_name (str): The name of the optimizer to use
        device (str, *optional*): Device to use for training
        warmup_steps (int, *optional*): Number of warmup steps to use for training
        save_every (int, *optional*): After how many batches should a checkpoint be saved
        save_dir, (str, *optional*): Where to save checkpoints
        wandb (bool): Whether to use Weights and Biases for logging
        wandb_project (str, *optional*): Name of the Weights and Biases project to use
        print_every (int, *optional*): Print the loss every n steps
        max_steps (int, *optional*): Terminate the epoch after this many steps. Used for debugging.
    """

    num_epochs: int
    batch_size: int
    lr: float = 1e-3
    seed: int = 0
    momentum: float = 0.0
    max_grad_norm: Optional[float] = None
    weight_decay: Optional[float] = None
    optimizer_name: str = "Adam"
    device: Optional[str] = None
    warmup_steps: int = 0
    save_every: Optional[int] = None
    save_dir: Optional[str] = None
    wandb: bool = False
    wandb_project_name: Optional[str] = None
    print_every: Optional[int] = None
    max_steps: Optional[int] = None


def train(
    model: HookedTransformer,
    config: HookedTransformerTrainConfig,
    dataset: Dataset,
) -> HookedTransformer:
    """
    Trains an HookedTransformer model on an autoregressive language modeling task.
    Args:
        model: The model to train
        config: The training configuration
        dataset: The dataset to train on - this function assumes the dataset is set up for autoregressive language modeling.
    Returns:
        The trained model
    """
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
    logging = []

    for epoch in range(1, config.num_epochs + 1):
        samples = 0
        for step, batch in enumerate(dataloader):
            tokens = batch["tokens"].to(config.device)
            loss = model(tokens, return_type="loss")
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
            
        logging.append((epoch, loss.item()))

    return model, logging



################### Create Dataset Class ####################
SIZE = 'large'
model_name = f'gpt2-{SIZE}'
model = HookedTransformer.from_pretrained(model_name, device=device)

class FineTuningDataset(Dataset):
    def __init__(self, data, max_length=15):
        self.data = data
        self.max_length = max_length
        self.pad_idx = 50256

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'tokens': model.to_tokens(self.data[idx]).squeeze(0)}

class FineTuningDataset_Padding(Dataset):
    def __init__(self, data, max_length=15):
        self.data = data
        self.max_length = max_length
        self.pad_idx = 50256

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        temp = model.to_tokens(self.data[idx]).tolist()[0]
        temp.extend([50256] * (self.max_length - len(temp)))
        toks = torch.tensor(temp)
        return {'tokens': toks}


################### Hyperparameter Search ####################
save_path = f'{root_dir}fine-tuning/FT_hyperparameter_search_{SIZE}.json'

for EPOCHS in [1, 5, 10, 50, 200, 1000]:
    for BS in [1, 10]:
        print(f"{'#'*20} LR: {LR}, EPOCHS: {EPOCHS}, BS: {BS} {'#'*10}")
        raw_model = HookedTransformer.from_pretrained(model_name, device=device)

        train_config = HookedTransformerTrainConfig(
            num_epochs=EPOCHS,
            batch_size=BS,
            lr=LR,
            optimizer_name="AdamW")

        if BS == 10:
            data = [
            "The partner drew the hat for the attorney.",
            "The employee found the tea for the band.",
            "The guy got the guitar for the company.",
            "The king gave the tea to the candidate.",
            "The brother kept the wire for the business.",
            "The mayor left the beer for the administration.",
            "The professor made the key for the president.",
            "The leader purchased the juice for the business.",
            "The husband saved the plate for the king.",
            "The son sold the cup to the army."
            ]
        else: data = ['The professor made the key for the president.']


        dataset = FineTuningDataset(data)
        adapted_model, logging = train(raw_model, train_config, dataset)

        target = 'A lady brought a pie to a club.'
        output_adapted = adapted_model(target)
        probs = nn.functional.log_softmax(output_adapted.squeeze()[:-1, :], dim=-1)
        probs = probs[range(probs.size(0)), adapted_model.to_tokens(target).squeeze(0).tolist()[1:]]
        sen_prob = torch.sum(probs).item()
        
        rec = {'parameters':f'{LR}_{EPOCHS}_{BS}','lr': LR, 'epochs': EPOCHS, 'batch_size': BS, 'sen_prob': sen_prob, 'logging': logging}
        with open(save_path, "a") as outfile:
            json.dump(rec, outfile)
            outfile.write('\n')


