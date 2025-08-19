import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np


class AudioRegressionTrainer:
    """Trainer class for audio temperature regression"""

    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.0001)
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.get('scheduler_step_size', 15),
            gamma=config.get('scheduler_gamma', 0.1)
        )

        # Training history
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'train_mae': [],
            'val_mae': [],
            'best_val_mae': float('inf')
        }

        self.best_model_state = None

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0

        pbar = tqdm(train_loader, desc='Training')
        for data, targets in pbar:
            data = data.to(self.device)
            targets = targets.to(self.device).float().unsqueeze(1)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            mae = torch.abs(outputs - targets).mean().item()

            total_loss += loss.item()
            total_mae += mae

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'MAE': f'{mae:.4f}'
            })

        avg_loss = total_loss / len(train_loader)
        avg_mae = total_mae / len(train_loader)
        return avg_loss, avg_mae

    def validate_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0

        with torch.no_grad():
            for data, targets in tqdm(val_loader, desc='Validation'):
                data = data.to(self.device)
                targets = targets.to(self.device).float().unsqueeze(1)

                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                mae = torch.abs(outputs - targets).mean().item()

                total_loss += loss.item()
                total_mae += mae

        avg_loss = total_loss / len(val_loader)
        avg_mae = total_mae / len(val_loader)
        return avg_loss, avg_mae

    def train(self, train_loader, val_loader):
        num_epochs = self.config['num_epochs']
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            print('-' * 50)

            train_loss, train_mae = self.train_epoch(train_loader)
            val_loss, val_mae = self.validate_epoch(val_loader)

            # Update history
            self.history['train_losses'].append(train_loss)
            self.history['val_losses'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)

            # Save best model
            if val_mae < self.history['best_val_mae']:
                self.history['best_val_mae'] = val_mae
                self.best_model_state = self.model.state_dict().copy()

            self.scheduler.step()

            # Print epoch summary
            print(f'Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')
            print(f'Best Val MAE: {self.history["best_val_mae"]:.4f}')

        # Load best model
        self.model.load_state_dict(self.best_model_state)

        return self.model, self.history
