import torch
from sklearn.metrics import accuracy_score, classification_report
from training.VibroNetClassifier import VibroNetClassifier
from utils.config_manager import load_config


def evaluate_model(model, test_loader, device='cuda'):
    config = load_config()
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
    temp_mapping = {idx: temp for idx, temp in enumerate(config["model"]["classes"])}
    temp_targets = [temp_mapping[t] for t in all_targets]
    temp_predictions = [temp_mapping[p] for p in all_predictions]

    report = classification_report(temp_targets, temp_predictions)

    return accuracy, report, all_predictions, all_targets


def load_trained_model(model_path, device='cuda'):
    """Load a trained model for inference"""
    checkpoint = torch.load(model_path, map_location=device)

    model = VibroNetClassifier(num_classes=8)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def predict_temperature(model, spectrogram, checkpoint=None, device='cuda'):
    """Predict temperature for a single spectrogram"""
    config = load_config()
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
    label_to_temp = {idx: temp for idx, temp in enumerate(config["model"]["classes"])}
    if checkpoint and 'label_to_temp' in checkpoint:
        label_to_temp = checkpoint['label_to_temp']

    predicted_temp = label_to_temp[predicted.item()]
    confidence = probabilities[0][predicted].item()

    return predicted_temp, confidence, probabilities[0].cpu().numpy()