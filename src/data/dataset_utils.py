import json
from datetime import datetime

import numpy as np
import torch
import torchaudio
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


def extract_datetime_from_filename(filename):
    try:
        parts = filename.split('_')
        date_str = parts[-2]  # '2025-07-15'
        time_str = parts[-1].split('.')[0:3]  # ['14', '02', '34']
        dt_str = date_str + ' ' + ':'.join(time_str)
        return datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"Could not parse datetime from {filename}: {e}")
        return None

def load_annotation(annotation_path):
    """Load annotation file and extract start and end time"""
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)

        audio_annotations = annotation.get('audio_annotations', {})

        # Get start and end time (annotation "1" and "2")
        if "1" in audio_annotations and "2" in audio_annotations:
            start_time = audio_annotations["1"]["time"]
            end_time = audio_annotations["2"]["time"]
            return start_time, end_time
    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"Warning: Could not load annotation from {annotation_path}: {e}")

    return None, None

def interpolate_temperatures(all_files):
    """ Interpolates temperatures for files with the same temperature """
    all_files = sorted(all_files, key=lambda x: extract_datetime_from_filename(x['file_name']))

    n = len(all_files)
    i = 0
    while i < n:
        current = all_files[i]
        group = [current]
        j = i + 1

        while j < n and all_files[j]['temperature'] == current['temperature']:
            group.append(all_files[j])
            j += 1
        if j < n and all_files[j]['temperature'] < current['temperature']:
            next_temp = all_files[j]['temperature']
            current_temp = current['temperature']

            temp_step = (current_temp - next_temp) / len(group)

            print(
                f"\n[INTERPOLATION] Group from {current['temperature']}°C → {next_temp}°C over {len(group)} files")
            for idx, g in enumerate(group):
                interpolated_temp = round(current['temperature'] - temp_step * idx, 3)
                g['temperature'] = interpolated_temp
                print(f"{interpolated_temp},", end=' ')
        else:
            print(f"\n[NO MATCH] {current['file_name']} → no next file with different temperature")
        i = j

def detect_pulses(audio_path, annotation_path, sample_rate, base_threshold=0,
                  smooth_win=5, local_win=50, min_distance_s=0.1):
    """
    Detects pulses in a signal using energy envelope and adaptive thresholding.
    
    Steps:
      1. Remove mean (DC offset)
      2. Take absolute value
      3. Compute RMS-like energy
      4. Smooth with moving average
      5. Adaptive thresholding (local + std-based)
      6. Enforce minimum spacing between detected pulses
    """
    start_time, end_time = load_annotation(annotation_path)
    if start_time is None or end_time is None:
        return []

    # Można zmienić frame i hop size albo sparametryzować
    frame_size = int(0.05 * sample_rate)
    hop_size = int(0.01 * sample_rate)

    def compute_energy_envelope(waveform):
        if waveform.shape[0] > 1:
            x = waveform.mean(dim=0)
        else:
            x = waveform.squeeze(0)

        x = x - x.mean()
        x = x.abs()

        if x.numel() < frame_size:
            energy = torch.sqrt(torch.mean(x ** 2)).unsqueeze(0)
        else:
            x_frames = x.unfold(0, frame_size, hop_size)
            energy = torch.sqrt(torch.mean(x_frames ** 2, dim=1))

        kernel = torch.ones(smooth_win) / smooth_win
        energy_smooth = torch.nn.functional.conv1d(
            energy.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding='same'
        ).squeeze()

        energy_smooth = energy_smooth / energy_smooth.max()
        return energy_smooth

    info = torchaudio.info(audio_path)
    sr = info.sample_rate
    start_sample = int(start_time * sr)
    num_samples = int((end_time - start_time) * sr)
    waveform, sr = torchaudio.load(audio_path, frame_offset=start_sample, num_frames=num_samples)

    energy = compute_energy_envelope(waveform)
    energy_np = energy.numpy()
    energy_np = np.asarray(energy_np).reshape(-1)

    local_mean = uniform_filter1d(energy_np, size=local_win)
    adaptive_threshold = local_mean + base_threshold * energy_np.std()

    min_distance_frames = int(min_distance_s * sample_rate / hop_size)

    peaks, _ = find_peaks(energy_np, height=adaptive_threshold, distance=min_distance_frames)
    times = start_time + peaks * hop_size / sr

    # --- Wizualizacja ---
    """
    plt.figure(figsize=(12, 5))
    plt.plot(energy_np, label="Smoothed Energy", alpha=0.8)
    plt.plot(local_mean, label="Local Mean", alpha=0.5)
    plt.plot(adaptive_threshold, label="Adaptive Threshold", linestyle="--", alpha=0.7)
    plt.plot(peaks, energy_np[peaks], "rx", label="Detected Pulses")
    plt.xlabel("Frame index")
    plt.ylabel("Normalized energy")
    plt.legend()
    plt.title("Adaptive Pulse Detection")
    plt.tight_layout()
    plt.show()
    """

    return times


def get_temperature_sets(all_files, classes):
    classes_sorted = sorted(classes, reverse=True)
    for file in all_files:
        for temp_cls in classes_sorted:
            if file['temperature'] >= temp_cls:
                file['temperature_set'] = temp_cls
                break
    

