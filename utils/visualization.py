import matplotlib.pyplot as plt
import random
import torch
import torchaudio
import torchaudio.transforms as T

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot losses
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(history['train_accuracies'], label='Train Accuracy')
    ax2.plot(history['val_accuracies'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(targets, predictions, class_names):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(targets, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Temperature')
    plt.ylabel('Actual Temperature')
    plt.show()


def show_random_spectrogram(dataset, n_examples=1):
    """
    Wyświetl przykładowe spektrogramy z datasetu

    Args:
        dataset: instancja AudioTemperatureDataset
        n_examples: ile spektrogramów pokazać
    """
    for _ in range(n_examples):
        idx = random.randint(0, len(dataset) - 1)
        spec, label = dataset[idx]

        spec_single = spec[0].numpy()

        plt.figure(figsize=(10, 4))
        plt.imshow(spec_single, origin="lower", aspect="auto", cmap="magma")
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Przykładowy spektrogram\nEtykieta: {label}")
        plt.xlabel("Czas (ramki)")
        plt.ylabel("Częstotliwość (mel)")
        plt.show()


def show_file_spectrogram(audio_path, sample_rate=48000):
    waveform, sr = torchaudio.load(audio_path)

    if sr != sample_rate:
        resampler = T.Resample(sr, sample_rate)
        waveform = resampler(waveform)
        sr = sample_rate

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        f_min=20,
        f_max=8000
    )
    db_transform = T.AmplitudeToDB(top_db=80)

    mel_spec = mel_transform(waveform)
    mel_spec_db = db_transform(mel_spec)

    plt.figure(figsize=(12, 5))
    plt.imshow(mel_spec_db.squeeze(0).numpy(), origin="lower", aspect="auto", cmap="magma")
    plt.colorbar(label="dB")
    plt.title(f"Mel-spektrogram: {audio_path}")
    plt.xlabel("Czas (ramki)")
    plt.ylabel("Częstotliwość (mel)")
    plt.show()
