import os
import json
import numpy as np
import soundfile as sf

from src.utils.config_manager import load_config

config = load_config()

AUDIO_DIR = config["data"]["data_root"]
ANN_DIR = config["data"]["annotation_root"]

SILENCE_THRESHOLD = 1e-6

bad_segments = []

for ann_file in os.listdir(ANN_DIR):
    if not ann_file.endswith(".json"):
        continue

    ann_path = os.path.join(ANN_DIR, ann_file)

    with open(ann_path, "r") as f:
        ann = json.load(f)

    wav_name = ann["audio_file"]
    wav_path = os.path.join(AUDIO_DIR, wav_name)

    if not os.path.exists(wav_path):
        print(f"Brak WAV: {wav_name}")
        continue

    try:
        signal, sr = sf.read(wav_path)
    except Exception as e:
        print(f"Nie da się wczytać WAV: {wav_name} | {e}")
        continue

    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    audio_len = len(signal)

    annotations = list(ann["audio_annotations"].values())

    for i in range(len(annotations) - 1):
        start_time = annotations[i]["time"]
        end_time = annotations[i + 1]["time"]

        start = int(start_time * sr)
        end = int(end_time * sr)

        reason = None

        if start >= end:
            reason = "start >= end"

        elif start >= audio_len:
            reason = "start poza sygnałem"

        elif end > audio_len:
            reason = "end poza sygnałem"

        else:
            segment = signal[start:end]

            if len(segment) == 0:
                reason = "pusty segment"

            elif np.isnan(segment).any() or np.isinf(segment).any():
                reason = "NaN / Inf"

            elif np.all(segment == 0):
                reason = "same zera"

            else:
                rms = np.sqrt(np.mean(segment ** 2))
                if rms < SILENCE_THRESHOLD:
                    reason = f"praktyczna cisza (rms={rms:.2e})"

        if reason:
            bad_segments.append({
                "wav": wav_name,
                "segment_id": i,
                "start_time": start_time,
                "end_time": end_time,
                "reason": reason
            })

print(f"\nZnaleziono {len(bad_segments)} problematycznych fragmentów")
print(bad_segments)