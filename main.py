import os

import torch

from data.data_manager import create_data_loaders
from training import VibroNet, AudioClassificationTrainer
from training.trainer_regression import AudioRegressionTrainer
from utils.config_manager import load_config
from utils.evaluation.evaluate_classification import evaluate_model
from utils.evaluation.evaluate_regression import evaluate_and_plot
from utils.visualization import plot_confusion_matrix


def main():
    config = load_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, dataset = create_data_loaders(config)

    mode = config['model'].get('type', 'classification')
    model = VibroNet(mode=mode, num_classes=config['model']['num_classes'])

    if mode == 'regression':
        trainer = AudioRegressionTrainer(model=model, config=config['training'], device=device)
    else:
        trainer = AudioClassificationTrainer(model=model, config=config['training'], device=device)

    print("Starting training...")
    trained_model, history = trainer.train(train_loader, val_loader)

    if mode == 'regression':
        evaluate_and_plot(trained_model, val_loader, device=device)
        return

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
