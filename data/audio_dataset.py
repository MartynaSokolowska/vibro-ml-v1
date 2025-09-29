import os
import json
import torch
import numpy as np
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from audiomentations import Compose, Gain, PitchShift
from torchvision.transforms.functional import crop
from random import choice
from datetime import datetime, timedelta
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from preprocessing.preprocessing_pipeline import AudioPipeline
from utils.config_manager import load_config


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
        config = load_config()
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

        # Audio augmentation
        if self.augment:
            self.audio_augment = Compose([
                Gain(min_gain_db=-2, max_gain_db=2, p=0.3),
                PitchShift(min_semitones=-1, max_semitones=1, p=0.2)
            ])

    def _create_slice_index(self):
        """Create index of all slices from all audio files"""
        slice_data = []
        all_files = []

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

                                # Get the real temperature
                                raw_temp_str = audio_file[:4].replace(',', '.')
                                try:
                                    true_temp = float(raw_temp_str)
                                except ValueError:
                                    print(f"Warning: Cannot parse temperature from filename '{audio_file}'")
                                    continue

                                try:
                                    file_time = datetime.fromtimestamp(os.path.getmtime(audio_path))
                                except Exception as e:
                                    print(f"Error reading timestamp for {audio_path}: {e}")
                                    continue

                                all_files.append({
                                    'audio_path': audio_path,
                                    'annotation_path': annotation_path,
                                    'temperature_set': temp,
                                    'temperature': true_temp,
                                    'pulse_time': 0,  # will change later
                                    'file_name': audio_file,
                                })

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

        # Sort by time
        all_files = sorted(all_files, key=lambda x: extract_datetime_from_filename(x['file_name']))

        n = len(all_files)
        i = 0
        while i < n:
            current = all_files[i]
            group = [current]
            j = i + 1

            # Szukamy kolejnych plików z tą samą temperaturą
            while j < n and all_files[j]['temperature'] == current['temperature']:
                    group.append(all_files[j])
                    j += 1
            if j<n and all_files[j]['temperature'] < current['temperature']:
                # znaleziono plik z inną temperaturą 
                # obliczamy krok
                    next_temp = all_files[j]['temperature']
                    current_temp = current['temperature']

                    temp_step = (current_temp - next_temp) / len(group)

                    print(f"\n[INTERPOLATION] Group from {current['temperature']}°C → {next_temp}°C over {len(group)} files") # TODO: no interpolation for classification? or make it configurable
                    for idx, g in enumerate(group):
                        interpolated_temp = round(current['temperature'] - temp_step * idx, 3)
                        print(interpolated_temp, ",", end='')

                        pulses = self._detect_pulses(g['audio_path'], g['annotation_path'])
                        num_slices = len(pulses)
                        for slice_idx in range(num_slices):
                            slice_data.append({
                                'audio_path': g['audio_path'],
                                'annotation_path': g['annotation_path'],
                                'temperature_set': g['temperature_set'],
                                'temperature': interpolated_temp,
                                'pulse_time': pulses[slice_idx],  
                            })            
            else:
                print(f"\n[NO MATCH] {current['file_name']} → no next file with different temperature")
                pulses = self._detect_pulses(current['audio_path'], current['annotation_path'])
                num_slices = len(pulses)
                for slice_idx in range(num_slices):
                    slice_data.append({
                        'audio_path': current['audio_path'],
                        'annotation_path': current['annotation_path'],
                        'temperature_set': current['temperature_set'],
                        'temperature': current['temperature'],
                        'pulse_time': pulses[slice_idx],
                    })
            i = j

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
    
    def _detect_pulses(self, audio_path, annotation_path, threshold=0.5):
        """ Detects pulses in foam """
        start_time, end_time = self._load_annotation(annotation_path)
        if start_time is None or end_time is None:
            return []
        
        # Można zmienić frame i hop size albo sparametryzować
        frame_size= int(0.05 * self.sample_rate)
        hop_size= int(0.01 * self.sample_rate)

        def compute_rms(waveform):
            if waveform.shape[0] > 1:
                x = waveform.mean(dim=0) 
            else:
                x = waveform.squeeze(0)   
                    
            if x.numel() < frame_size:
                rms = torch.sqrt(torch.mean(x**2)).unsqueeze(0)
                return rms

            x_frames = x.unfold(0, frame_size, hop_size)
            rms = torch.sqrt(torch.mean(x_frames**2, dim=1))
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

    def _count_slices(self, audio_path, annotation_path):
        # NOTE: Not used rn
        """Count how many slices can be extracted from this audio file"""
        try:
            # Load annotation
            start_time, end_time = self._load_annotation(annotation_path)
            if start_time is None or end_time is None:
                return 0
            
            ''' Constant count
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
            '''

            # Cut by pulses
            times = self.detect_pulses(audio_path, start_time, end_time)
            return len(times)

        except Exception as e:
            print(f"Error counting slices for {audio_path}: {e}")
            return 1

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

        start_time, end_time = self._load_annotation(slice_info['annotation_path'])

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

        config = load_config()
        if config['model']['type'] == 'classification':
            label = self.temp_to_label[slice_info['temperature_set']]
        else:
            label = float(slice_info['temperature'])

        return spectrogram, label
