import random
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from collections import defaultdict
from data.audio_dataset import AudioTemperatureDataset
from utils.config_manager import load_config


def create_file_based_splits(dataset, test_split=0.2, val_split=0.2, random_seed=42, equalize_num_samples = True):
    """
    Create train/val/test splits ensuring all slices from one file stay together

    Args:
        dataset: AudioTemperatureDataset instance
        test_split: Proportion of files for test set
        val_split: Proportion of remaining files for validation set
        random_seed: Random seed for reproducibility

    Returns:
        train_indices, val_indices, test_indices: Lists of dataset indices
    """
    file_to_slices = defaultdict(list)
    file_to_temp = {}

    for idx, slice_info in enumerate(dataset.slice_data):
        file_path = slice_info['audio_path']
        file_to_slices[file_path].append(idx)
        file_to_temp[file_path] = slice_info['temperature_set']

    # Get unique files and their temperatures
    files = list(file_to_slices.keys())
    file_temperatures = [file_to_temp[f] for f in files]

    print(f"Total unique audio files: {len(files)}")

    # Count files per temperature
    temp_counts = defaultdict(int)
    for temp in file_temperatures:
        temp_counts[temp] += 1

    config = load_config()
    if config['model']['type'] == 'classification':
        print("Files per temperature:")
        for temp, count in sorted(temp_counts.items()):
            print(f"  {temp}°C: {count} files")
        

    # First split: separate test files
    train_val_files, test_files = train_test_split(
        files,
        test_size=test_split,
        random_state=random_seed,
        stratify=file_temperatures if config['model']['type'] == 'classification' else None
    )

    # Get temperatures for remaining files
    train_val_temperatures = [file_to_temp[f] for f in train_val_files]

    # Second split: separate train and validation files
    train_files, val_files = train_test_split(
        train_val_files,
        test_size=val_split,  # This is now relative to train_val_files
        random_state=random_seed,
        stratify=train_val_temperatures if config['model']['type'] == 'classification' else None
    )

    # Convert file lists to slice indices
    train_indices = []
    val_indices = []
    test_indices = []

    if equalize_num_samples:
        # NOTE: Right now it only cuts the number of samples to the smaller set for training 
        # (consider additional augmentation or overlaping AND equalizing all datasets)
        temp_to_files = defaultdict(list)
        for file_path in train_files:
            temp = file_to_temp[file_path]
            temp_to_files[temp].append(file_path)
        min_count = min(len(files) for files in temp_to_files.values())

        temp_to_train_files = defaultdict(list)
        for file_path in train_files:
            temp = file_to_temp[file_path]
            temp_to_train_files[temp].append(file_path)

        balanced_train_files = []
        for temp, files in temp_to_train_files.items():
            if len(files) >= min_count:
                balanced_train_files.extend(random.sample(files, min_count))
            else:
                print(f"⚠️ Warning: class {temp} has less than min_count files")

        train_indices = []
        for file_path in balanced_train_files:
            train_indices.extend(file_to_slices[file_path])

    else: 
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
        config=config,
        augment=False  # We'll handle augmentation per split
    )

    print(f"Total samples: {len(dataset)}")

    # Create file-based splits
    train_indices, val_indices, test_indices = create_file_based_splits(
        dataset,
        test_split=config['data']['test_split'],
        val_split=config['data']['val_split'],
        random_seed=config['data']['random_seed'],
        equalize_num_samples=config['data'].get('equalize_num_samples', True)
    )

    train_dataset = AudioTemperatureDataset(
        config=config,
        augment=True
    )

    val_test_dataset = AudioTemperatureDataset(
        config=config,
        augment=False
    )

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
