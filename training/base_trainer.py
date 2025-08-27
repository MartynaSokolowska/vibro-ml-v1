import torch
import torch.optim as optim
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.0001)
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.get('scheduler_step_size', 15),
            gamma=config.get('scheduler_gamma', 0.1)
        )
        self.history = self.init_history()
        self.best_model_state = None

    @abstractmethod
    def init_history(self):
        pass

    @abstractmethod
    def train_epoch(self, train_loader):
        pass

    @abstractmethod
    def validate_epoch(self, val_loader):
        pass

    def train(self, train_loader, val_loader):
        num_epochs = self.config['num_epochs']
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            print('-' * 50)

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)

            self.update_history(train_metrics, val_metrics)
            self.save_best_model(val_metrics)

            self.scheduler.step()
            self.print_epoch_summary(train_metrics, val_metrics)

        self.model.load_state_dict(self.best_model_state)
        return self.model, self.history

    @abstractmethod
    def update_history(self, train_metrics, val_metrics):
        pass

    @abstractmethod
    def save_best_model(self, val_metrics):
        pass

    @abstractmethod
    def print_epoch_summary(self, train_metrics, val_metrics):
        pass