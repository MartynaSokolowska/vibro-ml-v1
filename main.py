import torch
import os

from data.data_manager import create_data_loaders
from training import VibroNetRegressor
from training.VibroNetClassifier import VibroNetClassifier
from training.trainer import AudioClassificationTrainer
from training.trainerRegression import AudioRegressionTrainer
from utils.config_manager import load_config
from utils.evaluation import evaluate_model
from utils.visualization import plot_training_history, plot_confusion_matrix
 

import matplotlib.pyplot as plt

def evaluate_and_plot(model, data_loader, device):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device).float().unsqueeze(1)
            output = model(data)
            predictions.extend(output.cpu().numpy().flatten())
            targets.extend(target.cpu().numpy().flatten())

    # MAE, MSE, RMSE
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np

    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)

    print(f"\n📊 Evaluation Metrics:")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Przykładowe predykcje
    print("\n🔍 Sample predictions:")
    for i in range(min(10, len(predictions))):
        print(f"Predicted: {predictions[i]:.2f}, Target: {targets[i]:.2f}")

    # Wykres dopasowania
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, predictions, alpha=0.6)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', label="Ideal")
    plt.xlabel("True Temperature")
    plt.ylabel("Predicted Temperature")
    plt.title("Model Fit: True vs Predicted")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    """Main function"""
    config = load_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, dataset = create_data_loaders(config)

    model = VibroNetRegressor(
        # num_classes=config['model']['num_classes']
    )

    trainer = AudioRegressionTrainer(
        model=model,
        config=config['training'],
        device=device
    )

    print("Starting training...")
    trained_model, history = trainer.train(train_loader, val_loader)

    # plot_training_history(history)
    evaluate_and_plot(trained_model, val_loader, device='cuda')
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
