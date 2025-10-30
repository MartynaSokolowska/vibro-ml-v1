import os
import sys

import torch

from data.data_manager import create_data_loaders
from training import VibroNet
from utils.config_manager import load_config
from utils.saliency_maps.saliency_maps import batch_saliency_analysis, generate_saliency_for_sample


def load_trained_model(model_path, device):
    """ Load a trained VibroNet model from a checkpoint"""

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    mode = checkpoint.get('mode', 'classification')
    config = checkpoint.get('config', {})

    if mode == 'classification':
        num_classes = len(checkpoint.get('label_to_temp', {}))
    else:
        num_classes = config.get('model', {}).get('num_classes', 6)

    model = VibroNet(mode=mode, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model: {model_path}")
    print(f"Mode: {mode}")
    if mode == 'classification':
        print(f"Classes: {checkpoint.get('label_to_temp', {})}")

    return model, checkpoint


def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    model_path = config['training']['model_save_path']

    if not os.path.exists(model_path):
        print(f"ERROR: Model does not exist {model_path}")
        return

    model, checkpoint = load_trained_model(model_path, device)
    mode = checkpoint.get('mode', 'classification')

    print("\nLoading data...")
    train_loader, val_loader, test_loader, dataset = create_data_loaders(config)

    dataloader = test_loader

    print("\n" + "="*60)
    print("GENERATING SALIENCY MAPS")
    print("="*60)

    save_dir_vanilla = "saliency_results/vanilla"
    batch_saliency_analysis(
        model=model,
        dataloader=dataloader,
        mode=mode,
        num_samples=5,
        save_dir=save_dir_vanilla
    )

    inputs, targets = next(iter(dataloader))
    input_tensor = inputs[0:1]
    true_value = targets[0].item()
    generate_saliency_for_sample(
        model=model,
        input_tensor=input_tensor,
        true_value=true_value,
        mode=mode,
        save_path="saliency_results/single_sample_vanilla.png"
    )


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    os.chdir(project_root)
    sys.path.insert(0, project_root)
    main()
