import os
import json
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from audiomentations import Compose, Gain, PitchShift
from torchvision.transforms.functional import crop
from random import choice


class AudioTemperatureDataset(Dataset):
    """Dataset for audio temperature classification"""

    def __init__(self, data_root, annotation_root, transform=None,
                 slice_length=1.0, overlap=0.5, sample_rate=48000, augment=True):
        """
        Initialize dataset

        Args:
            data_root: Root directory containing temperature folders
            annotation_root: Directory containing JSON annotation files
            transform: Optional transform to be applied to spectrograms
            slice_length: Duration in seconds for each slice
            overlap: Overlap ratio between slices (0.0 = no overlap, 0.5 = 50% overlap)
            sample_rate: Expected sample rate of audio files
            augment: Whether to apply data augmentation
        """
        self.data_root = data_root
        self.annotation_root = annotation_root
        self.transform = transform
        self.slice_length = slice_length
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.augment = augment

        # Temperature mapping
        self.temp_to_label = {20: 0, 25: 1, 30: 2, 35: 3, 40: 4, 45: 5, 50: 6, 55 : 7}
        self.label_to_temp = {v: k for k, v in self.temp_to_label.items()}

        self.slice_data = self._create_slice_index()

        # Spectrogram transforms
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )

        self.db_transform = T.AmplitudeToDB(top_db=80)

        # Audio augmentation
        if self.augment:
            self.audio_augment = Compose([
                Gain(min_gain_db=-5, max_gain_db=5, p=0.5),
                PitchShift(min_semitones=-3, max_semitones=3, p=0.5)
            ])

    def _create_slice_index(self):
        """Create index of all slices from all audio files"""
        slice_data = []

        # Get all audio files
        for root, dirs, files in os.walk(self.data_root):
            folder_name = os.path.basename(root)
            if folder_name.isdigit():
                temp = int(folder_name)
                if temp in self.temp_to_label:
                    for audio_file in files:
                        if audio_file.lower().endswith('.processed.wav'):
                            audio_path = os.path.join(root, audio_file)
                            annotation_file = audio_file.replace('.processed.wav', '.json')
                            annotation_path = os.path.join(self.annotation_root, annotation_file)

                            if os.path.exists(annotation_path):
                                # Get the number of slices for this file
                                num_slices = self._count_slices(audio_path, annotation_path)

                                # Add each slice to the index
                                for slice_idx in range(num_slices):
                                    slice_data.append({
                                        'audio_path': audio_path,
                                        'annotation_path': annotation_path,
                                        'temperature': temp,
                                        'slice_idx': slice_idx
                                    })

        return slice_data

    def _load_annotation(self, annotation_path):
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

    def _count_slices(self, audio_path, annotation_path):
        """Count how many slices can be extracted from this audio file"""
        try:
            # Load annotation
            start_time, end_time = self._load_annotation(annotation_path)
            if start_time is None or end_time is None:
                return 0

            # Calculate segment duration
            segment_duration = end_time - start_time

            # Calculate slice parameters
            slice_samples = int(self.slice_length * self.sample_rate)
            overlap_offset = int(slice_samples * (1 - self.overlap))
            segment_samples = int(segment_duration * self.sample_rate)

            # Count possible slices
            slice_count = 0
            while segment_samples > slice_samples + slice_count * overlap_offset:
                slice_count += 1

            return max(1, slice_count)  # At least 1 slice even if segment is short

        except Exception as e:
            print(f"Error counting slices for {audio_path}: {e}")
            return 1

    def _extract_slice(self, audio_path, start_time, end_time, slice_idx):
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

        # Extract the annotated segment first
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)

        start_sample = max(0, start_sample)
        end_sample = min(waveform.shape[1], end_sample)

        annotated_segment = waveform[:, start_sample:end_sample]

        # Now extract the specific slice
        slice_samples = int(self.slice_length * self.sample_rate)
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

        return slice_audio

    def _apply_audio_augmentation(self, audio_slice):
        """Apply audio augmentation to the audio slice"""
        if hasattr(self, 'audio_augment'):
            # Convert to numpy for audiomentations
            audio_np = audio_slice.squeeze().numpy()
            # Apply augmentation
            augmented = self.audio_augment(samples=audio_np, sample_rate=self.sample_rate)
            # Convert back to tensor
            return torch.from_numpy(augmented).unsqueeze(0)
        return audio_slice

    def _audio_to_spectrogram(self, audio_slice):
        """Convert audio slice to mel spectrogram"""
        # Generate mel spectrogram
        spec = self.mel_spectrogram(audio_slice)
        # Convert to dB scale
        spec = self.db_transform(spec)
        # Remove channel dimension and return
        return spec.squeeze(0)

    def _apply_spectrogram_augmentation(self, spectrogram):
        """Apply spectrogram augmentation"""
        # Time stretch augmentation
        target_shape = spectrogram.shape
        stretch = T.TimeStretch(n_freq=128)
        stretch_factor = choice([0.5, 0.7, 1.0, 1.2, 1.5])

        # Add batch dimension for TimeStretch
        spec_with_batch = spectrogram.unsqueeze(0)
        stretched = stretch(spec_with_batch, stretch_factor)

        # Crop/pad to original shape and remove batch dimension
        stretched = crop(stretched.squeeze(0), 0, 0, *target_shape)

        return stretched.to(torch.float32)

    def __len__(self):
        return len(self.slice_data)

    def __getitem__(self, idx):
        slice_info = self.slice_data[idx]

        # Load annotation
        start_time, end_time = self._load_annotation(slice_info['annotation_path'])

        # Extract the specific slice
        audio_slice = self._extract_slice(
            slice_info['audio_path'],
            start_time,
            end_time,
            slice_info['slice_idx']
        )

        # Apply audio augmentation during training
        if self.augment and torch.rand(1).item() < 0.5:
            audio_slice = self._apply_audio_augmentation(audio_slice)

        # Convert to spectrogram
        spectrogram = self._audio_to_spectrogram(audio_slice)

        # Apply spectrogram augmentation during training
        if self.augment and torch.rand(1).item() < 0.3:
            spectrogram = self._apply_spectrogram_augmentation(spectrogram)

        # Convert to 3-channel image for ResNet (RGB)
        spectrogram = torch.stack([spectrogram, spectrogram, spectrogram])

        # Apply transforms
        if self.transform:
            spectrogram = self.transform(spectrogram)

        label = self.temp_to_label[slice_info['temperature']]

        return spectrogram, label