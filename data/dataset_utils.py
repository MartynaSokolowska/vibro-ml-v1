import datetime
import json

import torch
import torchaudio
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

def detect_pulses(audio_path, annotation_path, sample_rate, threshold=0.5):
    """ Detects pulses in foam """
    start_time, end_time = load_annotation(annotation_path)
    if start_time is None or end_time is None:
        return []

    # Można zmienić frame i hop size albo sparametryzować
    frame_size = int(0.05 * sample_rate)
    hop_size = int(0.01 * sample_rate)

    def compute_rms(waveform):
        if waveform.shape[0] > 1:
            x = waveform.mean(dim=0)
        else:
            x = waveform.squeeze(0)

        if x.numel() < frame_size:
            rms = torch.sqrt(torch.mean(x ** 2)).unsqueeze(0)
            return rms

        x_frames = x.unfold(0, frame_size, hop_size)
        rms = torch.sqrt(torch.mean(x_frames ** 2, dim=1))
        return rms

    info = torchaudio.info(audio_path)
    sr = info.sample_rate
    start_sample = int(start_time * sr)
    num_samples = int((end_time - start_time) * sr)
    waveform, sr = torchaudio.load(audio_path, frame_offset=start_sample, num_frames=num_samples)

    rms = compute_rms(waveform)
    rms = rms / rms.max()
    rms_np = rms.numpy()

    peaks, _ = find_peaks(rms_np, height=threshold * rms_np.mean(), distance=5)
    times = start_time + peaks * hop_size / sr

    ''' For visualisation
    plt.figure(figsize=(12, 4))
    plt.plot(rms.numpy(), label="RMS energy")
    plt.plot(peaks, rms.numpy()[peaks], "rx", label="Foud pulses")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Normalized energy")
    plt.show()
    '''

    return times