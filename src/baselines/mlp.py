import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.baselines import extract_features_from_indices, get_feature_names
from src.data.data_manager import create_file_based_splits
from src.data.audio_dataset import AudioTemperatureDataset
from src.utils.config_manager import load_config


class FeatureDataset(Dataset):
    """Dataset wrapper for extracted features.

    Accepts precomputed numpy arrays X, y and optional groups.
    Returns feature tensors shaped (n_features,) for MLP input.
    """

    def __init__(self, X, y, groups=None, mode='regression'):
        """
        Args:
            X: Feature array (n_samples, n_features) - numpy array or torch tensor
            y: Labels array (n_samples,) - numpy array or torch tensor
            groups: Optional group labels for cross-validation
            mode: 'regression' or 'classification'
        """
        self.mode = mode
        self.groups = groups

        if isinstance(X, np.ndarray):
            X = X.astype(np.float32)
            self.X = torch.from_numpy(X)
        elif torch.is_tensor(X):
            self.X = X.to(torch.float32)
        else:
            raise TypeError("X must be numpy array or torch tensor")

        if mode == 'classification':
            if isinstance(y, np.ndarray):
                self.y = torch.from_numpy(y.astype(np.int64))
            elif torch.is_tensor(y):
                self.y = y.to(torch.int64)
            else:
                raise TypeError("y must be numpy array or torch tensor")
        else:
            if isinstance(y, np.ndarray):
                self.y = torch.from_numpy(y.astype(np.float32)).reshape(-1, 1)
            elif torch.is_tensor(y):
                self.y = y.to(torch.float32).view(-1, 1)
            else:
                raise TypeError("y must be numpy array or torch tensor")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleMLP(nn.Module):
    """MLP for handcrafted feature vectors.

    Architecture:
        Input(n_features)
        -> Linear(256) + BN + ReLU + Dropout(0.3)
        -> Linear(128) + BN + ReLU + Dropout(0.3)
        -> Linear(64)  + BN + ReLU + Dropout(0.2)
        -> head: Linear(1) [regression] or Linear(num_classes) [classification]
    """

    def __init__(self, n_features: int, mode: str = 'regression', num_classes: int = 8):
        super().__init__()
        self.mode = mode

        self.encoder = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        if mode == 'regression':
            self.head = nn.Linear(64, 1)
        else:
            self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, n_features)
        return self.head(self.encoder(x))


def build_feature_dataloaders(base_dataset,
                              train_indices,
                              val_indices,
                              test_indices=None,
                              batch_size: int = 32,
                              mode: str = 'regression',
                              num_workers: int = 0,
                              return_scaler: bool = False):
    """
    Args:
        base_dataset: Non-augmented AudioTemperatureDataset instance
        train_indices: Indices for training set
        val_indices: Indices for validation set
        test_indices: Optional indices for test set
        batch_size: Batch size for dataloaders
        mode: 'regression' or 'classification'
        num_workers: Number of workers for dataloaders
        return_scaler: Whether to return the fitted scaler

    Returns:
        (train_loader, val_loader, n_features) or
        (train_loader, val_loader, test_loader, n_features, scaler) if test_indices provided and return_scaler=True
        or variations depending on test_indices and return_scaler flags
    """
    X_train, y_train, groups_train = extract_features_from_indices(base_dataset, train_indices)
    X_val, y_val, groups_val = extract_features_from_indices(base_dataset, val_indices)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_ds = FeatureDataset(X_train, y_train, groups=groups_train, mode=mode)
    val_ds = FeatureDataset(X_val, y_val, groups=groups_val, mode=mode)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    n_features = X_train.shape[1]

    if test_indices is not None:
        X_test, y_test, groups_test = extract_features_from_indices(base_dataset, test_indices)
        X_test = scaler.transform(X_test)
        test_ds = FeatureDataset(X_test, y_test, groups=groups_test, mode=mode)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        if return_scaler:
            return train_loader, val_loader, test_loader, n_features, scaler
        return train_loader, val_loader, test_loader, n_features

    if return_scaler:
        return train_loader, val_loader, n_features, scaler
    return train_loader, val_loader, n_features


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """Run one epoch. Returns mean loss and (preds, targets) numpy arrays."""
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(X_batch)
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(y_batch.detach().cpu().numpy())

    mean_loss = total_loss / len(loader.dataset)
    preds_np = np.concatenate(all_preds).ravel()
    targets_np = np.concatenate(all_targets).ravel()
    return mean_loss, preds_np, targets_np


def train_mlp(model,
              train_loader,
              val_loader,
              test_loader=None,
              num_epochs: int = 100,
              lr: float = 1e-3,
              weight_decay: float = 1e-4,
              patience: int = 15,
              save_dir: str = 'results/baselines/mlp',
              device: str = None):
    """
    Train SimpleMLP with early stopping, then evaluate on test set and save plots.

    Args:
        model:          SimpleMLP instance (already on correct device or will be moved)
        train_loader:   Training DataLoader
        val_loader:     Validation DataLoader
        test_loader:    Optional test DataLoader for final evaluation
        num_epochs:     Maximum training epochs
        lr:             Adam learning rate
        weight_decay:   L2 regularisation
        patience:       Early-stopping patience (epochs without val improvement)
        save_dir:       Directory to save model checkpoint and plots
        device:         'cpu', 'cuda', or None (auto-detect)

    Returns:
        history dict with keys: train_loss, val_loss, val_mae, val_r2
        test_metrics dict (empty if no test_loader)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 3
    )

    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_r2': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0

    print(f"Training on {device} | max_epochs={num_epochs} | patience={patience}")
    print(f"{'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'Val MAE':>9}  {'Val R²':>8}")
    print("-" * 52)

    for epoch in range(1, num_epochs + 1):
        train_loss, _, _ = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_preds, val_targets = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        val_mae = mean_absolute_error(val_targets, val_preds)
        val_r2 = r2_score(val_targets, val_preds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path / 'best_model.pt')
        else:
            epochs_no_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  {val_mae:>9.4f}  {val_r2:>8.4f}")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (best epoch: {best_epoch})")
            break

    print(f"\nBest val MSE: {best_val_loss:.4f} at epoch {best_epoch}")

    test_metrics = {}
    if test_loader is not None:
        model.load_state_dict(torch.load(save_path / 'best_model.pt', map_location=device))
        _, test_preds, test_targets = _run_epoch(model, test_loader, criterion, optimizer, device, train=False)
        test_metrics = {
            'mse': mean_squared_error(test_targets, test_preds),
            'rmse': np.sqrt(mean_squared_error(test_targets, test_preds)),
            'mae': mean_absolute_error(test_targets, test_preds),
            'r2': r2_score(test_targets, test_preds),
        }
        print("\nTest set metrics:")
        for k, v in test_metrics.items():
            print(f"  {k.upper():>5}: {v:.4f}")

        plot_results(history, test_preds, test_targets, test_metrics, save_path)
    else:
        plot_results(history, save_path=save_path)

    return history, test_metrics


def _plot_loss_panel(ax, epochs, train_loss, val_loss, title):
    ax.plot(epochs, train_loss, label='Train MSE', linewidth=1.8)
    ax.plot(epochs, val_loss,   label='Val MSE',   linewidth=1.8, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_metrics_panel(ax, epochs, val_mae, val_r2, title):
    color_mae = '#1f77b4'
    color_r2  = '#d62728'
    ax.plot(epochs, val_mae, color=color_mae, linewidth=1.8, label='Val MAE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val MAE', color=color_mae)
    ax.tick_params(axis='y', labelcolor=color_mae)
    ax_twin = ax.twinx()
    ax_twin.plot(epochs, val_r2, color=color_r2, linewidth=1.8, linestyle='-.', label='Val R²')
    ax_twin.set_ylabel('Val R²', color=color_r2)
    ax_twin.tick_params(axis='y', labelcolor=color_r2)
    ax.set_title(title)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.grid(True, alpha=0.3)


def _plot_scatter_panel(ax, test_preds, test_targets, test_metrics):
    if test_preds is not None and test_targets is not None:
        ax.scatter(test_targets, test_preds, alpha=0.6, s=25, edgecolors='none')
        lims = [min(test_targets.min(), test_preds.min()),
                max(test_targets.max(), test_preds.max())]
        ax.plot(lims, lims, 'r--', linewidth=1.2, label='Perfect fit')
        if test_metrics:
            ax.set_title(
                f'Test: Predicted vs Actual\n'
                f'RMSE={test_metrics["rmse"]:.3f}  MAE={test_metrics["mae"]:.3f}  R²={test_metrics["r2"]:.3f}'
            )
        else:
            ax.set_title('Test: Predicted vs Actual')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No test set provided',
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='grey')
        ax.set_title('Test: Predicted vs Actual')
        ax.axis('off')


def plot_results(history, test_preds=None, test_targets=None, test_metrics=None, save_path='.',
                 zoom_start: int = 20):
    """
    Two-row figure (2 x 3):
      Row 1: full training history  — loss | val MAE+R² | predicted vs actual
      Row 2: epoch zoom_start+      — loss | val MAE+R² | predicted vs actual (same scatter)
    """
    save_path = Path(save_path)

    n_epochs = len(history['train_loss'])
    epochs_full = range(1, n_epochs + 1)

    zoom_idx = zoom_start - 1 if n_epochs > zoom_start else 0
    epochs_zoom = range(zoom_start, n_epochs + 1) if n_epochs > zoom_start else epochs_full

    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.38, hspace=0.45)

    _plot_loss_panel(
        fig.add_subplot(gs[0, 0]), epochs_full,
        history['train_loss'], history['val_loss'],
        'Loss – full training'
    )
    _plot_metrics_panel(
        fig.add_subplot(gs[0, 1]), epochs_full,
        history['val_mae'], history['val_r2'],
        'Val Metrics – full training'
    )
    _plot_scatter_panel(fig.add_subplot(gs[0, 2]), test_preds, test_targets, test_metrics)

    _plot_loss_panel(
        fig.add_subplot(gs[1, 0]), epochs_zoom,
        history['train_loss'][zoom_idx:], history['val_loss'][zoom_idx:],
        f'Loss – epoch {zoom_start}+'
    )
    _plot_metrics_panel(
        fig.add_subplot(gs[1, 1]), epochs_zoom,
        history['val_mae'][zoom_idx:], history['val_r2'][zoom_idx:],
        f'Val Metrics – epoch {zoom_start}+'
    )
    _plot_scatter_panel(fig.add_subplot(gs[1, 2]), test_preds, test_targets, test_metrics)

    plt.suptitle('SimpleMLP – Training Summary', fontsize=13, y=1.01)
    out_path = save_path / 'training_summary.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {out_path}")


def get_feature_count():
    return len(get_feature_names())


def create_feature_dataloaders_from_manager(config):
    base_dataset = AudioTemperatureDataset(config=config, augment=False)

    train_indices, val_indices, test_indices = create_file_based_splits(
        base_dataset,
        test_split=config['data']['test_split'],
        val_split=config['data']['val_split'],
        random_seed=config['data']['random_seed'],
        equalize_num_samples=config['data'].get('equalize_num_samples', True)
    )

    train_loader, val_loader, test_loader, n_features, scaler = build_feature_dataloaders(
        base_dataset,
        train_indices,
        val_indices,
        test_indices,
        batch_size=config['training']['batch_size'],
        mode='regression' if config['model']['type'] == 'regression' else 'classification',
        num_workers=config['data']['num_workers'],
        return_scaler=True
    )

    return train_loader, val_loader, test_loader, n_features, scaler


if __name__ == "__main__":
    config = load_config()
    train_loader, val_loader, test_loader, n_features, scaler = create_feature_dataloaders_from_manager(config)

    model = SimpleMLP(n_features=n_features, mode='regression')
    history, test_metrics = train_mlp(
        model, train_loader, val_loader, test_loader,
        num_epochs=1000,
        patience=20,
        save_dir='results/baselines/mlp',
    )
