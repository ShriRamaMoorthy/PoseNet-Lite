from __future__ import annotations
from torch import nn
from torch.utils.data import DataLoader
import torch
from training.trainer import(train_one_epoch, validate_one_epoch)


class TrainingOrchesrtator:
    def __init__(self,
                 model:nn.Module,
                 train_loader:DataLoader,
                 val_loader:DataLoader,
                 criterion:nn.Module,
                 optimizer:torch.optim.Optimizer,
                 device:torch.device,
                 epochs:int):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs


    def train(self):
        for epoch in range(self.epochs):
            print()
            print('='*60)
            print(
                f'Epoch {epoch+1}/{self.epochs}'
            )
            print('='*60)

            train_metrics = train_one_epoch(
                model = self.model,
                dataloader = self.train_loader,
                criterion = self.criterion,
                optimizer = self.optimizer,
                device = self.device
            )

            validation_metrics = validate_one_epoch(
                model = self.model,
                dataloader = self.val_loader,
                criterion = self.criterion,
                device = self.device
            )

            print()

            print(f'Train Loss : {train_metrics['loss']:.4f}')
            print(f'Validation Loss : {validation_metrics['loss']:.4f}')