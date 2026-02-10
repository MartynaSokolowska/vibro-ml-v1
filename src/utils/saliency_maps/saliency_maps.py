import os

import matplotlib.pyplot as plt
import torch
import random


class VanillaSaliency:
    """ Vanilla saliency map implementation """

    def __init__(self, model):
        self.model = model

    def generate_saliency(self, input_tensor, target_class=None):
        """
        Generate saliency map

        Args:
            input_tensor: (batch_size, channels, height, width)
            target_class: Target class index for classification tasks

        Returns:
            saliency: Saliency map (height, width)
            output: Model output
        """
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        self.model.eval()

        output = self.model(input_tensor)

        if target_class is None:
            if output.dim() == 1 or output.size(1) == 1:
                target = output.squeeze()
            else:
                pred_idx = int(output.argmax(dim=1).item())
                target = output[0, pred_idx]
        else:
            if isinstance(target_class, torch.Tensor):
                target_idx = int(target_class.item())
            else:
                target_idx = int(target_class)
            target = output[0, target_idx]

        self.model.zero_grad()
        target.backward()

        saliency = input_tensor.grad.abs()
        saliency = saliency.squeeze(0).max(dim=0)[0]
        saliency = saliency - saliency.min()
        saliency = saliency / (saliency.max() + 1e-8)

        return saliency.detach().cpu().numpy(), output.detach().cpu()


def visualize_saliency_map(
    original_spec,
    saliency_map,
    output_value,
    true_value,
    method_name="Saliency",
    mode="regression",
    save_path=None
):
    """ Visualize saliency map alongside original spectrogram """

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im1 = axes[0].imshow(original_spec, origin='lower', aspect='auto', cmap='magma')
    axes[0].set_title('Original spectrogram')
    axes[0].set_xlabel('Timeframe')
    axes[0].set_ylabel('Mel Frequency')
    plt.colorbar(im1, ax=axes[0], format='%+2.0f dB')

    im2 = axes[1].imshow(saliency_map, origin='lower', aspect='auto', cmap='hot')
    axes[1].set_title(f'{method_name} Map')
    axes[1].set_xlabel('Timeframe')
    axes[1].set_ylabel('Mel Frequency')
    plt.colorbar(im2, ax=axes[1])

    axes[2].imshow(original_spec, origin='lower', aspect='auto', cmap='magma', alpha=0.7)
    axes[2].imshow(saliency_map, origin='lower', aspect='auto', cmap='hot', alpha=0.4)
    axes[2].set_title(f'{method_name} Overlay')
    axes[2].set_xlabel('Timeframe')
    axes[2].set_ylabel('Mel Frequency')

    if mode == "regression":
        fig.suptitle(f'Predicted: {output_value:.2f}°C | True: {true_value:.2f}°C',
                     fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f'Predicted: {output_value} | True: {true_value}',
                     fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved saliency map to: {save_path}")

    plt.show()


def generate_saliency_for_sample(
    model,
    input_tensor,
    true_value,
    mode="regression",
    target_class=None,
    save_path=None
):
    """ Generate and visualize saliency map for a single sample """

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    vanilla = VanillaSaliency(model)
    saliency_map, output = vanilla.generate_saliency(input_tensor.clone(), target_class)
    method_name = "Vanilla Saliency"

    original_spec = input_tensor.squeeze(0).cpu().numpy()
    if original_spec.ndim == 3:
        original_spec = original_spec[0]

    if mode == "regression":
        output_value = output.item()
    else:
        output_value = output.argmax(dim=1).item()

    visualize_saliency_map(
        original_spec,
        saliency_map,
        output_value,
        true_value,
        method_name,
        mode,
        save_path
    )

    return saliency_map, output


def batch_saliency_analysis(
        model,
        dataloader,
        mode="regression",
        num_samples=5,
        save_dir=None
):
    """ Generate saliency maps for randomly sampled examples from the dataloader """
    import random

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    model.eval()

    all_inputs = []
    all_targets = []

    for inputs, targets in dataloader:
        all_inputs.append(inputs)
        all_targets.append(targets)

    all_inputs = torch.cat(all_inputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    total_samples = all_inputs.size(0)

    sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))

    print(f"Randomly selected {len(sample_indices)} samples from {total_samples} total samples")
    print(f"Sample indices: {sample_indices}")

    for idx, sample_idx in enumerate(sample_indices):
        input_tensor = all_inputs[sample_idx:sample_idx + 1]
        true_value = all_targets[sample_idx].item()

        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f"saliency_vanilla_{idx}_sample{sample_idx}.png")

        print(f"\nGenerating saliency map {idx + 1}/{len(sample_indices)} (sample index: {sample_idx})...")
        generate_saliency_for_sample(
            model,
            input_tensor,
            true_value,
            mode,
            save_path=save_path
        )

    print(f"\nFinished generating {len(sample_indices)} saliency maps.")
