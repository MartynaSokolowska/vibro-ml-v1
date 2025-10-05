import os

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from audiomentations import Compose, Gain, PitchShift
from torch.utils.data import Dataset

from data.dataset_utils import detect_pulses, interpolate_temperatures
from preprocessing.preprocessing_pipeline import AudioPipeline
from utils.config_manager import load_config


class AudioTemperatureDataset(Dataset):
    """Dataset for audio temperature classification"""

    def __init__(self, config, transform=None, augment=True):
        """
        Initialize dataset

        Args:
            config: loaded configuration dictionary
            transform: Optional transform to be applied to spectrograms
            augment: Whether to apply data augmentation
        """
        self.data_root = config["data"]["data_root"]
        self.annotation_root = config["data"]["annotation_root"]
        self.transform = transform
        self.slice_length = config["data"]["slice_length"]
        self.overlap = config["data"]["overlap"]
        self.sample_rate = config["data"]["sample_rate"]
        self.augment = augment
        self.should_interpolate = config["data"].get("should_interpolate", False)

        self.mode = config["model"]["type"]
        if self.mode not in ["classification", "regression"]:
            raise ValueError(f"Incorrect model type: {self.mode}. Allowed: 'classification', 'regression'.")

        if self.mode == "classification":
            self.temp_to_label = {temp: idx for idx, temp in enumerate(config["model"]["classes"])}
            self.label_to_temp = {v: k for k, v in self.temp_to_label.items()}

        self.slice_data = self._create_slice_index()

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=20,
            f_max=8000
        )

        self.db_transform = T.AmplitudeToDB(top_db=80)

        if self.augment:
            self.audio_augment = Compose([
                Gain(min_gain_db=-2, max_gain_db=2, p=0.3),
                PitchShift(min_semitones=-1, max_semitones=1, p=0.2)
            ])

    def _create_slice_index(self):
        """Create index of all slices from all audio files"""
        slice_data = []
        all_files = []

        for root, dirs, files in os.walk(self.data_root):
            for audio_file in files:
                if audio_file.lower().endswith('wav') and not audio_file.lower().endswith('.processed.wav'):
                    audio_path = os.path.join(root, audio_file)
                    annotation_file = audio_file.replace('.wav', '.json')
                    annotation_path = os.path.join(self.annotation_root, annotation_file)

                    if os.path.exists(annotation_path):

                        raw_temp_str = audio_file[:4].replace(',', '.')
                        try:
                            true_temp = float(raw_temp_str)
                        except ValueError:
                            print(f"Warning: Cannot parse temperature from filename '{audio_file}'")
                            continue

                        all_files.append({
                            'audio_path': audio_path,
                            'annotation_path': annotation_path,
                            'temperature_set': true_temp,
                            'temperature': true_temp,
                            'file_name': audio_file
                        })

        if self.should_interpolate:
            interpolate_temperatures(all_files)

        for file_data in all_files:
            pulses = detect_pulses(file_data['audio_path'], file_data['annotation_path'], self.sample_rate)
            num_slices = len(pulses)
            for slice_idx in range(num_slices):
                slice_data.append({
                    'audio_path': file_data['audio_path'],
                    'annotation_path': file_data['annotation_path'],
                    'temperature_set': file_data['temperature_set'],
                    'temperature': file_data['temperature'],
                    'pulse_time': pulses[slice_idx],
                })

        return slice_data

    def _extract_slice(self, audio_path, pulse_time):
        """Extract specific slice from the annotated segment"""
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            waveform = torch.zeros(1, int(self.slice_length * self.sample_rate))
            sr = self.sample_rate

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        slice_samples = int(self.slice_length * self.sample_rate)

        '''
        # Based on constant length
        # Extract the annotated segment first
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)

        start_sample = max(0, start_sample)
        end_sample = min(waveform.shape[1], end_sample)

        annotated_segment = waveform[:, start_sample:end_sample]

        # Now extract the specific slice
        overlap_offset = int(slice_samples * (1 - self.overlap))

        slice_start = slice_idx * overlap_offset
        slice_end = slice_start + slice_samples

        # Extract slice
        if slice_end <= annotated_segment.shape[1]:
            slice_audio = annotated_segment[:, slice_start:slice_end]
        else:
            # Handle last slice that might be shorter
            slice_audio = annotated_segment[:, slice_start:]
            # Pad to correct length
            if slice_audio.shape[1] < slice_samples:
                padding = slice_samples - slice_audio.shape[1]
                slice_audio = torch.nn.functional.pad(slice_audio, (0, padding))
        '''
        # Based on pulses
        peak_sample = int(pulse_time * self.sample_rate)
        start_sample = peak_sample - slice_samples // 2
        end_sample = start_sample + slice_samples

        start_sample = max(0, start_sample)
        end_sample = min(waveform.shape[1], end_sample)

        slice_audio = waveform[:, start_sample:end_sample]

        # Padding
        if slice_audio.shape[1] < slice_samples:
            padding = slice_samples - slice_audio.shape[1]
            slice_audio = torch.nn.functional.pad(slice_audio, (0, padding))

        return slice_audio

    def _apply_audio_augmentation(self, audio_slice):
        """Apply audio augmentation to the audio slice"""
        if hasattr(self, 'audio_augment'):
            audio_np = audio_slice.squeeze().numpy()
            augmented = self.audio_augment(samples=audio_np, sample_rate=self.sample_rate)
            return torch.from_numpy(augmented).unsqueeze(0)
        return audio_slice

    def _audio_to_spectrogram(self, audio_slice):
        """Convert audio slice to mel spectrogram"""
        spec = self.mel_spectrogram(audio_slice)
        spec = self.db_transform(spec)
        return spec.squeeze(0)

    def _apply_spectrogram_augmentation(self, spectrogram):
        # TODO: Ogarnąć te wymiarowości, czy to błąd konkretnego pliku?
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)
        elif spectrogram.dim() == 1:
            spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected spectrogram shape: {spectrogram.shape}")
        
        if torch.rand(1).item() < 0.3:
            spectrogram = T.TimeMasking(time_mask_param=25)(spectrogram)
        if torch.rand(1).item() < 0.3:
            spectrogram = T.FrequencyMasking(freq_mask_param=10)(spectrogram)
        return spectrogram.squeeze(0)


    def __len__(self):
        return len(self.slice_data)

    def __getitem__(self, idx):
        slice_info = self.slice_data[idx]

        # Extract the specific slice
        audio_slice = self._extract_slice(
            slice_info['audio_path'],
            slice_info['pulse_time']
        )

        pipeline = AudioPipeline(audio_slice.squeeze(0).numpy(), self.sample_rate)
        config = load_config()
        processed_slice = pipeline.run_from_config(config["preprocessing"])
        audio_slice = torch.from_numpy(processed_slice.astype(np.float32)).unsqueeze(0)

        if self.augment and torch.rand(1).item() < 0.5:
            audio_slice = self._apply_audio_augmentation(audio_slice)

        spectrogram = self._audio_to_spectrogram(audio_slice)

        if self.augment:
            spectrogram = self._apply_spectrogram_augmentation(spectrogram)

        # Convert to 3-channel image for ResNet (RGB)
        spectrogram = torch.stack([spectrogram, spectrogram, spectrogram])

        if self.transform:
            spectrogram = self.transform(spectrogram)

        if self.mode == 'classification':
            label = self.temp_to_label[slice_info['temperature_set']]
        else:
            label = float(slice_info['temperature'])

        return spectrogram, label
