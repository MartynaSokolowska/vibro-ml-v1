import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score


class AudioClassificationTrainer:
    """Trainer class for audio temperature classification"""

    def __init__(self, model, config, device='cuda'):
        """
        Initialize trainer

        Args:
            model: PyTorch model to train
            config: Training configuration dictionary
            device: Device to use for training
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
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
            'train_accuracies': [],
            'val_accuracies': [],
            'train_f1_scores': [],
            'val_f1_scores': [],
            'best_val_acc': 0.0
        }

        self.best_model_state = None

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_predictions = []
        all_targets = []

        pbar = tqdm(train_loader, desc='Training')

        for data, targets in pbar:
            data, targets = data.to(self.device), targets.to(self.device)

            # print(data.shape)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * train_correct / train_total:.2f}%'
            })

        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        train_f1 = f1_score(all_targets, all_predictions, average='weighted')
        return avg_train_loss, train_acc, train_f1

    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, targets in tqdm(val_loader, desc='Validation'):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        val_f1 = f1_score(all_targets, all_predictions, average='weighted')

        return avg_val_loss, val_acc, val_f1

    def train(self, train_loader, val_loader):
        """Full training loop"""
        num_epochs = self.config['num_epochs']

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            print('-' * 50)

            train_loss, train_acc, train_f1 = self.train_epoch(train_loader)
            val_loss, val_acc, val_f1 = self.validate_epoch(val_loader)

            # Update history
            self.history['train_losses'].append(train_loss)
            self.history['val_losses'].append(val_loss)
            self.history['train_accuracies'].append(train_acc)
            self.history['val_accuracies'].append(val_acc)
            self.history['train_f1_scores'].append(train_f1)
            self.history['val_f1_scores'].append(val_f1)

            # Save best model
            if val_acc > self.history['best_val_acc']:
                self.history['best_val_acc'] = val_acc
                self.best_model_state = self.model.state_dict().copy()

            # Step scheduler
            self.scheduler.step()

            # Print epoch results
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}')
            print(f'Best Val Acc: {self.history["best_val_acc"]:.2f}%')

        # Load best model
        self.model.load_state_dict(self.best_model_state)

        return self.model, self.history
