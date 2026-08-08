from __future__ import annotations
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_one_epoch(
        model:nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
)->Dict[str,float]:
    
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader,
                        desc='Training')

    for batch in progress_bar:
        images = batch['image'].to(device)
        heatmaps = batch['heatmaps'].to(device)
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions,
                         heatmaps)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        progress_bar.set_postfix(loss=loss.item())

    average_loss = running_loss / len(dataloader)

    return{
        "loss":average_loss
    }


def validate_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device
)->Dict[str,float]:

    model.eval()
    running_loss = 0.0
    progress_bar = tqdm(dataloader,
                        desc='Validation')

    with torch.no_grad():
        for batch in progress_bar:
            images = batch['image'].to(device)
            heatmaps = batch['image'].to(device)
            predictions = model(images)
            loss = criterion(predictions,heatmaps)
            running_loss+=loss.item()

            progress_bar.set_postfix(loss = loss.item())

    average_loss = running_loss / len(dataloader)

    return {
        'loss' : average_loss
    }