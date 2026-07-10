import torch
from torch.utils.data import DataLoader
import os

from src.data.audio_dataset import AudioTemperatureDataset
from src.training import VibroNet
from src.utils.config_manager import load_config
from src.utils.evaluation.evaluate_regression import evaluate_and_plot

"""

def main():
    config = load_config()
    model_path = config['training']['model_save_path']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading trained model...")
    model, checkpoint = load_trained_model(model_path, device)

    print("Creating DataLoader for new data...")
    checkpoint['config']['data']['data_root'] = config['data']['data_root']
    checkpoint['config']['data']['annotation_root'] = os.path.join(
        checkpoint['config']['data']['data_root'], 'annotations'
    )
    loader, dataset = create_new_data_loader(checkpoint['config'])
    print(f"Załadowano {len(dataset)} slice'ów z nowych danych")


    print("Evaluating on new data...")
    accuracy, report, predictions, targets = evaluate_model(model, loader, device)

    # Odzyskanie nazw klas
    label_to_temp = checkpoint['label_to_temp']
    class_names = [label_to_temp[i] for i in range(len(label_to_temp))]

    plot_confusion_matrix(targets, predictions, class_names)

    print(f"Accuracy on new data: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
"""

def load_trained_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    config = checkpoint.get('config', {})
    model_config = checkpoint.get('model_config', config.get('model', {}))

    model = VibroNet(
        mode=checkpoint.get('mode', model_config.get('type', 'regression')),
        num_classes=model_config.get('num_classes', 1),
        backbone=checkpoint.get('backbone', model_config.get('backbone')),
        config=config,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint


def create_new_data_loader(config):
    dataset = AudioTemperatureDataset(
        config=config,
        augment=False
    )

    loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )

    return loader, dataset


def main():
    config = load_config()
    model_path = config['training']['model_save_path']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading trained regression model...")
    model, checkpoint = load_trained_model(model_path, device)

    print("Creating DataLoader for new data...")
    checkpoint['config']['data']['data_root'] = config['data']['data_root']
    checkpoint['config']['data']['annotation_root'] = os.path.join(
        checkpoint['config']['data']['data_root'], 'annotations'
    )
    loader, dataset = create_new_data_loader(checkpoint['config'])
    print(f"Załadowano {len(dataset)} slice'ów z nowych danych")

    print("Evaluating regression model on new data...")
    evaluate_and_plot(model, loader, device=device)


if __name__ == "__main__":
    main()
