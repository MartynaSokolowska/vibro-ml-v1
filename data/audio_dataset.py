"""Module containing data-related functions and classes.
Change to be compatible with Windows"""
# pylint: disable=import-error

import logging
import os

from time import time
from random import shuffle, choice
from collections import defaultdict, Counter

import torch
import torchaudio.transforms as T

from torch.utils.data import Subset
from torchaudio import load, functional
from torchvision.datasets import DatasetFolder
from torchvision.transforms.functional import crop
from audiomentations import Compose, Gain, PitchShift


# pylint: disable=too-many-instance-attributes
class AudioDataset(DatasetFolder):
    """Datset used for loading audio files."""

    # pylint: disable=too-many-arguments
    def __init__(self, config):
        # get required dataset attributes
        try:
            config["target_folder"]
        except KeyError:
            logging.exception(
                'Dataset folder attribute (target_folder) must be defined in "data" config section!'
            )
        try:
            self.sample_rate = config["sr"]
        except KeyError:
            logging.exception(
                'Sample rate attribute (sr) must be defined in "data" config section!'
            )
        # find classes
        self.classes, self.class_to_idx = self.find_classes(config["target_folder"])
        # assign files in dataset folder to class
        self.file_to_class = DatasetFolder.make_dataset(
            config["target_folder"], self.class_to_idx, extensions=config["extensions"]
        )
        # shuffled_idxs = list(self.class_to_idx.values())
        # shuffle(shuffled_idxs)
        # self.class_to_idx = {cls: idx for cls, idx in zip(list(self.class_to_idx.keys()), shuffled_idxs)}
        # create helper attribute
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        # get optional dataset attributes
        self.stereo = config.get("stereo", False)
        if "all" in config["augment"]:
            self.augment = list(self.class_to_idx.values())
        else:
            self.augment = [
                self.class_to_idx[class_name]
                for class_name in config["augment"]
                if class_name in self.class_to_idx
            ]
        self.spectogram_conversion_times = []

    def __str__(self):
        counter = Counter([elem[1] for elem in self.file_to_class])
        selfstring = f"Dataset containing {len(self)} data points, including:\n"
        for class_idx, cnt in counter.items():
            selfstring += f"\t- class {self.idx_to_class[class_idx]} (id {class_idx}, augment: {class_idx in self.augment}): {cnt} items\n"
        return selfstring

    def __len__(self):
        return len(self.file_to_class)

    # pylint: disable=invalid-name
    def __getitem__(self, index):
        # get path and load audio
        audio_path, label = self.file_to_class[index]
        signal, sr = load(audio_path)
        # augment discrete signal if class should be augmented
        if label in self.augment:
            signal = self._augment_signal(signal)
        # if stereo, select appropriate channel
        if self.stereo:
            signal = signal[0]
        # adjust sample rate if necessary
        if sr != self.sample_rate:
            print(sr, self.sample_rate)
            signal = functional.resample(signal, sr, self.sample_rate)
        # convert to spectrogram
        begin = time()
        spec = self._convert_to_spectogram(signal)
        self.spectogram_conversion_times.append(time() - begin)
        # augment spectrogram if class hould be augmented:
        if label in self.augment:
            spec = self._augment_spectogram(spec)
        # add 3 dimensions for ResNet
        spec = torch.stack([spec, spec, spec])
        # convert label to expected temperature
        label = int(self.idx_to_class[label])
        return spec, label

    def _augment_signal(self, signal):
        # apply gain variation
        augment = Compose([
            Gain(min_gain_db=-5, max_gain_db=5, p=1.0),
            PitchShift(min_semitones=-3, max_semitones=3, p=1.0)
        ])

        # Torchaudio tensor ➡ numpy ➡ augment ➡ back to tensor
        signal = signal.numpy()
        if signal.ndim == 2:
            signal = signal[0]  # Remove channel dimension if mono

        signal = augment(samples=signal, sample_rate=self.sample_rate)
        signal = torch.tensor(signal, dtype=torch.float32)
        print(f"Signal shape before MelSpectrogram: {signal.shape}")
        return signal

    def _augment_spectogram(self, spec):
        # apply time stretch
        target_shape = spec.shape
        stretch = T.TimeStretch(n_freq=256)
        value = choice([0.5, 0.7, 1, 1.2, 1.5])
        spec = stretch(spec, value)
        spec = crop(spec, 0, 0, *target_shape)
        return spec.to(torch.float32)

    def _convert_to_spectogram(self, signal):
        print(f"Signal type: {type(signal)}, shape: {signal.shape}, device: {signal.device}")

        spec = T.MelSpectrogram(
            self.sample_rate, n_fft=2048, hop_length=512, n_mels=256
        )(signal)
        spec = T.AmplitudeToDB(top_db=80)(spec)
        # spec = adjust_contrast(spec.unsqueeze(0), contrast_factor=2)
        # spec = adjust_sharpness(spec, sharpness_factor=4).squeeze()
        return spec


# pylint: disable=too-many-locals
def file_split(dataset, ratios):
    """Split dataset into n subsets so that each subset has its assigned ratio of samples.
    The functions takes into account a specific use case: the raw audio files are split
    into 1 second parts. In order to prevent data leakage, all samples resulting from one
    raw audio file split should be in the same subset.

    Args:
        dataset (DatasetFolder): The dataset to be split.
        ratios (iterable(float)): Ratio of training samples to testing samples.

    Returns:
        tuple(Subset): Two subsets resulting in the dataset split.
    """
    assert sum(ratios) == 1, "Ratios do not sum to 1."
    # variable for holding the number of files per recording split
    data_count = {dataset.class_to_idx[c]: defaultdict(int) for c in dataset.classes}
    # variable for holding the number of examples per class
    class_count = {dataset.class_to_idx[c]: 0 for c in dataset.classes}
    # calculate the number of files per recording and examples per class
    for file, idx in dataset.file_to_class:
        rec_name = os.path.basename(os.path.dirname(file))
        #rec_name = file.split("/")[-2]
        data_count[idx][rec_name] += 1
        class_count[idx] += 1
    # calculate the number of instances per class that should be in the training subset
    split_classes_count = []
    for ratio in ratios:
        split_classes_count.append(
            {k: int(count * ratio) for k, count in class_count.items()}
        )
    # variable for holding decision on subset membership of recording
    data_split = data_count.copy()
    # split the dataset
    for class_idx, files in data_count.items():
        dirs = list(files.keys())
        # ensure randomness of split
        shuffle(dirs)
        # pylint: disable=invalid-name
        for d in dirs:
            for split_idx, cls_count in enumerate(split_classes_count):
                # put example into train dataset
                if cls_count[class_idx] > 0:
                    cls_count[class_idx] -= data_count[class_idx][d]
                    data_split[class_idx][d] = split_idx
                    break
    # extract indices of dataset samples that belong in each subset
    indices = [[] for _ in split_classes_count]
    for idx, (file, class_idx) in enumerate(dataset.file_to_class):
        rec_name = os.path.basename(os.path.dirname(file))
        try:
            indices[data_split[class_idx][rec_name]].append(idx)
        except IndexError:
            pass
    # create subsets
    subsets = [Subset(dataset, idxs) for idxs in indices]
    return subsets
