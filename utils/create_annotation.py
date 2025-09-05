import os
import json
import yaml
from pathlib import Path

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def generate_annotations(folder_root, sample_rate):
    folder_root = Path(folder_root)
    annotations_dir = folder_root / "annotations"
    annotations_dir.mkdir(exist_ok=True)

    for wav_file in folder_root.rglob("*.processed.wav"):
        base_name = wav_file.stem.replace(".processed", "")
        json_file = annotations_dir / f"{base_name}.json"

        # Approximate times
        time_stamps = [3.0, 4.7]

        audio_annotations = {}
        for i, t in enumerate(time_stamps, 1):
            sample = int(round(t * sample_rate))
            audio_annotations[str(i)] = {
                "time": t,
                "sample": sample
            }

        annotation_data = {
            "audio_file": f"{base_name}.wav",
            "audio_annotations": audio_annotations
        }

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(annotation_data, f, indent=4)

# === USAGE ===

config = load_config("config\\config.yaml")
sample_rate = config["data"]["sample_rate"]
data_root = "C:\\Users\\sokol\\OneDrive\\Pulpit\\SEM_8\\magisterka\\new_data_1\\data" # config["data"]["data_root"]

generate_annotations(data_root, sample_rate)
