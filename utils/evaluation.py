import torch
from sklearn.metrics import accuracy_score, classification_report
from training.model import VibroNet


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model on test set"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_predictions)

    # Convert labels back to temperatures for report
    temp_mapping = {0: 20, 1: 25, 2: 30, 3: 35, 4: 40, 5: 45, 6: 50, 7: 55}  # TODO: should be from config
    temp_targets = [temp_mapping[t] for t in all_targets]
    temp_predictions = [temp_mapping[p] for p in all_predictions]

    report = classification_report(temp_targets, temp_predictions)

    return accuracy, report, all_predictions, all_targets


def load_trained_model(model_path, device='cuda'):
    """Load a trained model for inference"""
    checkpoint = torch.load(model_path, map_location=device)

    model = VibroNet(num_classes=8)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def predict_temperature(model, spectrogram, checkpoint=None, device='cuda'):
    """Predict temperature for a single spectrogram"""
    model.eval()

    # Add batch dimension if needed
    if spectrogram.dim() == 3:
        spectrogram = spectrogram.unsqueeze(0)

    spectrogram = spectrogram.to(device)

    with torch.no_grad():
        outputs = model(spectrogram)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

    # Convert to temperature
    label_to_temp = {0: 20, 1: 25, 2: 30, 3: 35, 4: 40, 5: 45, 6: 50, 7: 55}
    if checkpoint and 'label_to_temp' in checkpoint:
        label_to_temp = checkpoint['label_to_temp']

    predicted_temp = label_to_temp[predicted.item()]
    confidence = probabilities[0][predicted].item()

    return predicted_temp, confidence, probabilities[0].cpu().numpy()