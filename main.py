import yaml
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os

from data.audio_dataset import AudioTemperatureDataset
from training.model import VibroNet
from training.trainer import AudioTrainer
from utils.evaluation import evaluate_model
from utils.visualization import plot_training_history, plot_confusion_matrix


def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_file_based_splits(dataset, test_split=0.2, val_split=0.2, random_seed=42):
    """
    Create train/val/test splits ensuring all slices from same file stay together

    Args:
        dataset: AudioTemperatureDataset instance
        test_split: Proportion of files for test set
        val_split: Proportion of remaining files for validation set
        random_seed: Random seed for reproducibility

    Returns:
        train_indices, val_indices, test_indices: Lists of dataset indices
    """
    # Group slices by their source audio file
    file_to_slices = defaultdict(list)
    file_to_temp = {}

    for idx, slice_info in enumerate(dataset.slice_data):
        file_path = slice_info['audio_path']
        file_to_slices[file_path].append(idx)
        file_to_temp[file_path] = slice_info['temperature']

    # Get unique files and their temperatures
    files = list(file_to_slices.keys())
    file_temperatures = [file_to_temp[f] for f in files]

    print(f"Total unique audio files: {len(files)}")

    # Count files per temperature
    temp_counts = defaultdict(int)
    for temp in file_temperatures:
        temp_counts[temp] += 1

    print("Files per temperature:")
    for temp, count in sorted(temp_counts.items()):
        print(f"  {temp}°C: {count} files")

    # First split: separate test files
    train_val_files, test_files = train_test_split(
        files,
        test_size=test_split,
        random_state=random_seed,
        stratify=file_temperatures
    )

    # Get temperatures for remaining files
    train_val_temperatures = [file_to_temp[f] for f in train_val_files]

    # Second split: separate train and validation files
    train_files, val_files = train_test_split(
        train_val_files,
        test_size=val_split,  # This is now relative to train_val_files
        random_state=random_seed,
        stratify=train_val_temperatures
    )

    # Convert file lists to slice indices
    train_indices = []
    val_indices = []
    test_indices = []

    for file_path in train_files:
        train_indices.extend(file_to_slices[file_path])

    for file_path in val_files:
        val_indices.extend(file_to_slices[file_path])

    for file_path in test_files:
        test_indices.extend(file_to_slices[file_path])

    print(f"Split results:")
    print(f"  Train files: {len(train_files)} -> {len(train_indices)} slices")
    print(f"  Val files: {len(val_files)} -> {len(val_indices)} slices")
    print(f"  Test files: {len(test_files)} -> {len(test_indices)} slices")

    # Verify no file appears in multiple splits
    train_file_set = set(train_files)
    val_file_set = set(val_files)
    test_file_set = set(test_files)

    assert len(train_file_set & val_file_set) == 0, "Files overlap between train and val!"
    assert len(train_file_set & test_file_set) == 0, "Files overlap between train and test!"
    assert len(val_file_set & test_file_set) == 0, "Files overlap between val and test!"

    print("✓ No file overlap between splits - data leakage prevented!")

    return train_indices, val_indices, test_indices


def create_data_loaders(config):
    """Create train, validation, and test data loaders with file-based splitting"""
    # Create dataset to get the slice information
    dataset = AudioTemperatureDataset(
        data_root=config['data']['data_root'],
        annotation_root=config['data']['annotation_root'],
        slice_length=config['data']['slice_length'],
        sample_rate=config['data']['sample_rate'],
        augment=False  # We'll handle augmentation per split
    )

    print(f"Total samples: {len(dataset)}")

    # Create file-based splits
    train_indices, val_indices, test_indices = create_file_based_splits(
        dataset,
        test_split=config['data']['test_split'],
        val_split=config['data']['val_split'],
        random_seed=config['data']['random_seed']
    )

    # Create datasets with different augmentation settings
    train_dataset = AudioTemperatureDataset(
        data_root=config['data']['data_root'],
        annotation_root=config['data']['annotation_root'],
        slice_length=config['data']['slice_length'],
        sample_rate=config['data']['sample_rate'],
        augment=True  # Augmentation for training
    )

    val_test_dataset = AudioTemperatureDataset(
        data_root=config['data']['data_root'],
        annotation_root=config['data']['annotation_root'],
        slice_length=config['data']['slice_length'],
        sample_rate=config['data']['sample_rate'],
        augment=False  # No augmentation for validation/test
    )

    # Create data loaders
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )

    val_loader = DataLoader(
        Subset(val_test_dataset, val_indices),
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )

    test_loader = DataLoader(
        Subset(val_test_dataset, test_indices),
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )

    return train_loader, val_loader, test_loader, dataset


def main():
    """Main function"""
    config = load_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, dataset = create_data_loaders(config)

    model = VibroNet(
        num_classes=config['model']['num_classes']
    )

    trainer = AudioTrainer(
        model=model,
        config=config['training'],
        device=device
    )

    print("Starting training...")
    trained_model, history = trainer.train(train_loader, val_loader)

    plot_training_history(history)

    print("Evaluating on test set...")
    test_accuracy, classification_rep, predictions, targets = evaluate_model(
        trained_model, test_loader, device
    )

    class_names = [dataset.label_to_temp[i] for i in range(len(dataset.label_to_temp))]
    plot_confusion_matrix(targets, predictions, class_names)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_rep)

    model_save_path = config['training']['model_save_path']
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'temp_to_label': dataset.temp_to_label,
        'label_to_temp': dataset.label_to_temp,
        'slice_length': config['data']['slice_length'],
        'sample_rate': config['data']['sample_rate'],
        'config': config
    }, model_save_path)

    print(f"Model saved as '{model_save_path}'")


if __name__ == "__main__":
    main()