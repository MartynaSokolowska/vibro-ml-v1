import torch
import torch.nn as nn
from tqdm import tqdm

from training.base_trainer import BaseTrainer


class AudioRegressionTrainer(BaseTrainer):
    """Trainer class for audio temperature regression"""

    def __init__(self, model, config, device='cuda'):
        super().__init__(model, config, device)
        self.criterion = nn.MSELoss()

    def init_history(self):
        return {
            'train_losses': [],
            'val_losses': [],
            'train_mae': [],
            'val_mae': [],
            'best_val_mae': float('inf')
        }

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

    def update_history(self, train_metrics, val_metrics):
        train_loss, train_mae = train_metrics
        val_loss, val_mae = val_metrics

        self.history['train_losses'].append(train_loss)
        self.history['val_losses'].append(val_loss)
        self.history['train_mae'].append(train_mae)
        self.history['val_mae'].append(val_mae)

    def save_best_model(self, val_metrics):
        _, val_mae = val_metrics
        if val_mae < self.history['best_val_mae']:
            self.history['best_val_mae'] = val_mae
            self.best_model_state = self.model.state_dict().copy()

    def print_epoch_summary(self, train_metrics, val_metrics):
        train_loss, train_mae = train_metrics
        val_loss, val_mae = val_metrics
        print(f'Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')
        print(f'Best Val MAE: {self.history["best_val_mae"]:.4f}')
