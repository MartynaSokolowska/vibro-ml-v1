import numpy as np
import torchaudio
import torch
import json, os
import torchaudio.transforms as T
import re


def load_file(audio_path, sample_rate, normalize=False):
    """
    Load a WAV file and return a mono numpy array resampled to `sample_rate`.

    Parameters:
        audio_path : str
            Path to the WAV file
        sample_rate : int
            Target sampling rate in Hz

    Returns:
        signal : np.ndarray or None
            1D numpy array with audio samples or None if loading fails
    """
    try:
        waveform, sr = torchaudio.load(audio_path)
    except FileNotFoundError:
        print(f"File not found: {audio_path}")
        return None, None
    except RuntimeError as e:
        print(f"Cannot read {audio_path}: {e}")
        return None, None

    if sr != sample_rate:
        resampler = T.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    signal = waveform.squeeze().numpy()
    signal = signal - np.mean(signal)

    if normalize:
        signal = signal / np.max(np.abs(signal))

    return signal


def load_segment(audio_path, annotation_path, sample_rate, normalize=False):
    signal = load_file(audio_path, sample_rate, normalize=normalize)
    if signal is None:
        return None

    with open(annotation_path, 'r') as f:
        ann = json.load(f)

    start_sample = ann["audio_annotations"]["1"]["sample"]
    end_sample = ann["audio_annotations"]["2"]["sample"]

    return signal[start_sample:end_sample]


def get_date(filename):
    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
    if match:
        date_only = match.group()
        return date_only
    return None


def load_dataset(data_root, annotations_root, sample_rate, normalize=False):
    dataset = []

    for fname in os.listdir(data_root):
            if not fname.endswith(".wav"):
                continue
            if fname.endswith(".processed.wav"):
                continue

            audio_path = os.path.join(data_root, fname)
            ann_path = os.path.join(annotations_root, fname.replace(".wav", ".json"))

            if not os.path.exists(ann_path):
                print(f"Missing annotation for {fname}")
                continue

            segment = load_segment(audio_path, ann_path, sample_rate, normalize)
            if segment is None:
                continue

            raw_temp_str = fname[:4].replace(',', '.')
            try:
                true_temp = float(raw_temp_str)
            except ValueError:
                print(f"Warning: Cannot parse temperature from filename '{fname}'")
                continue

            dataset.append({
                "temperature_set": true_temp,
                "temperature": true_temp,
                "filename": fname,
                "date": get_date(fname),
                "signal": segment
            })
    """
    for temp_folder in sorted(os.listdir(data_root)):
        temp_path = os.path.join(data_root, temp_folder)

        if not os.path.isdir(temp_path):
            continue
        if temp_folder == "annotations":
            continue

        #temperature_nominal = float(temp_folder)

        for fname in os.listdir(temp_path):
            if not fname.endswith(".wav"):
                continue
            if fname.endswith(".processed.wav"):
                continue

            audio_path = os.path.join(temp_path, fname)
            ann_path = os.path.join(annotations_root, fname.replace(".wav", ".json"))

            if not os.path.exists(ann_path):
                print(f"Missing annotation for {fname}")
                continue

            segment = load_segment(audio_path, ann_path, sample_rate, normalize)
            if segment is None:
                continue

            raw_temp_str = fname[:4].replace(',', '.')
            try:
                true_temp = float(raw_temp_str)
            except ValueError:
                print(f"Warning: Cannot parse temperature from filename '{fname}'")
                continue

            dataset.append({
                "temperature_set": true_temp,
                "temperature": true_temp,
                "filename": fname,
                "date": get_date(fname),
                "signal": segment
            })
    """
    return dataset
