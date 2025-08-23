import torch
from matplotlib import pyplot as plt


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