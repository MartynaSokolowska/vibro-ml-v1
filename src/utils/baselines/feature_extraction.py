"""Feature extraction functions for spectrograms"""

import numpy as np
import torch
from scipy import stats
from scipy.signal import find_peaks
from scipy.stats import gmean


def extract_simple_features(data_loader):
    """Extract simple statistical features from spectrograms"""
    features_list = []
    labels_list = []
    for spectrograms, labels in data_loader:
        for spec, label in zip(spectrograms, labels):
            if isinstance(spec, torch.Tensor):
                spec_np = spec[0].cpu().numpy() if spec.ndim == 3 else spec.cpu().numpy()
            else:
                spec_np = spec

            features = extract_simple_features_from_spec(spec_np)
            features_list.append(list(features.values()))
            labels_list.append(label.cpu().numpy() if isinstance(label, torch.Tensor) else label)
    return np.array(features_list), np.array(labels_list)


def extract_simple_features_from_indices(base_dataset, indices):
    """
    Deterministic feature extraction:
    - uses base `dataset` returned by create_data_loaders(...) (augment=False)
    - preserves exact subset order
    - returns groups aligned 1:1 with X and y
    """
    features_list = []
    labels_list = []
    groups_list = []

    for idx in indices:
        spec, label = base_dataset[idx]

        if isinstance(spec, torch.Tensor):
            spec_np = spec[0].cpu().numpy() if spec.ndim == 3 else spec.cpu().numpy()
        else:
            spec_np = spec

        features = extract_simple_features_from_spec(spec_np)
        features_list.append(list(features.values()))
        labels_list.append(label.item() if isinstance(label, torch.Tensor) else float(label))
        groups_list.append(base_dataset.slice_data[idx]['audio_path'])  # group by source file

    return np.array(features_list), np.array(labels_list), np.array(groups_list)


def extract_features(data_loader):
    """Extract both simple and advanced features"""
    features_list = []
    labels_list = []

    for spectrograms, labels in data_loader:
        for spec, label in zip(spectrograms, labels):

            if isinstance(spec, torch.Tensor):
                spec_np = spec[0].cpu().numpy() if spec.ndim == 3 else spec.cpu().numpy()
            else:
                spec_np = spec

            all_features = {}

            all_features.update(extract_simple_features_from_spec(spec_np))
            all_features.update(extract_bandwise_energy(spec_np, num_bands=8))
            all_features.update(extract_spectral_features(spec_np))
            all_features.update(extract_framewise_stats(spec_np))
            all_features.update(extract_delta_features(spec_np))
            all_features.update(extract_peak_features(spec_np))
            all_features.update(extract_logmel_features(spec_np))

            features_list.append(list(all_features.values()))
            labels_list.append(label.cpu().numpy() if isinstance(label, torch.Tensor) else label)

    return np.array(features_list), np.array(labels_list)


def extract_features_from_indices(base_dataset, indices):
    """
    Extract all features (simple + advanced) deterministically from indices:
    - uses base `dataset` returned by create_data_loaders(...) (augment=False)
    - preserves exact subset order
    - returns groups aligned 1:1 with X and y for GroupKFold CV

    Returns:
        X: Feature array of shape (n_samples, n_features)
        y: Labels array of shape (n_samples,)
        groups: Audio file paths for grouping samples by source file
    """
    features_list = []
    labels_list = []
    groups_list = []

    for idx in indices:
        spec, label = base_dataset[idx]

        if isinstance(spec, torch.Tensor):
            spec_np = spec[0].cpu().numpy() if spec.ndim == 3 else spec.cpu().numpy()
        else:
            spec_np = spec

        all_features = {}
        all_features.update(extract_simple_features_from_spec(spec_np))
        all_features.update(extract_bandwise_energy(spec_np, num_bands=8))
        all_features.update(extract_spectral_features(spec_np))
        all_features.update(extract_framewise_stats(spec_np))
        all_features.update(extract_delta_features(spec_np))
        all_features.update(extract_peak_features(spec_np))
        all_features.update(extract_logmel_features(spec_np))

        features_list.append(list(all_features.values()))
        labels_list.append(label.item() if isinstance(label, torch.Tensor) else float(label))
        groups_list.append(base_dataset.slice_data[idx]['audio_path'])

    return np.array(features_list), np.array(labels_list), np.array(groups_list)


def extract_simple_features_from_spec(spec):
    """Extract simple statistical features from a single spectrogram
    
    Args:
        spec: numpy array or torch tensor of shape (H, W) or (C, H, W)
    
    Returns:
        Dictionary of simple statistical features
    """
    
    features = {
        'mean': np.mean(spec),
        'std': np.std(spec),
        'max': np.max(spec),
        'min': np.min(spec),
        'median': np.median(spec),
        'q25': np.percentile(spec, 25),
        'q75': np.percentile(spec, 75),
        'energy': np.sum(spec ** 2),
        'rms': np.sqrt(np.mean(spec ** 2)),
        'skewness': stats.skew(spec.flatten()),
        'kurtosis': stats.kurtosis(spec.flatten())
    }
    
    return features


def extract_bandwise_energy(spec, num_bands=8):
    """Extract energy in fixed frequency bands"""
    n_freqs = spec.shape[0]
    band_size = n_freqs // num_bands
    features = {}

    for band_idx in range(num_bands):
        start_idx = band_idx * band_size
        end_idx = start_idx + band_size if band_idx < num_bands - 1 else n_freqs
        band_energy = np.sum(spec[start_idx:end_idx, :] ** 2)
        features[f'band_{band_idx}_energy'] = band_energy

    return features


def extract_spectral_features(spec):
    """Extract spectral centroid, bandwidth, roll-off, flatness"""
    features = {}

    # Sum over time to get frequency magnitude profile
    freq_mag = np.maximum(np.mean(spec, axis=1), 1e-10)
    freqs = np.arange(len(freq_mag))

    # Spectral centroid
    centroid = np.sum(freqs * freq_mag) / (np.sum(freq_mag) + 1e-10)
    features['spectral_centroid'] = centroid

    # Spectral bandwidth
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * freq_mag) / (np.sum(freq_mag) + 1e-10))
    features['spectral_bandwidth'] = bandwidth

    # Spectral roll-off (85% energy threshold)
    energy_cumsum = np.cumsum(freq_mag ** 2)
    energy_threshold = 0.85 * energy_cumsum[-1]
    roll_off_idx = np.where(energy_cumsum >= energy_threshold)[0]
    features['spectral_rolloff'] = roll_off_idx[0] if len(roll_off_idx) > 0 else len(freqs)

    # Spectral flatness (Wiener entropy)
    geom_mean = gmean(freq_mag)
    arith_mean = np.mean(freq_mag)
    features['spectral_flatness'] = geom_mean / (arith_mean + 1e-10)

    return features


def extract_framewise_stats(spec):
    """Extract frame-wise mean/std over time, then aggregate"""
    features = {}

    # Frame-wise statistics
    frame_mean = np.mean(spec, axis=0)  # Mean for each time frame
    frame_std = np.std(spec, axis=0)    # Std for each time frame

    # Aggregate over time
    features['framewise_mean_mean'] = np.mean(frame_mean)
    features['framewise_mean_std'] = np.std(frame_mean)
    features['framewise_std_mean'] = np.mean(frame_std)
    features['framewise_std_std'] = np.std(frame_std)

    # Additional stats
    features['framewise_mean_max'] = np.max(frame_mean)
    features['framewise_mean_min'] = np.min(frame_mean)
    features['framewise_std_max'] = np.max(frame_std)

    return features


def extract_delta_features(spec):
    """Extract delta features (temporal change in features)"""
    features = {}

    # Delta along time axis (axis=1)
    time_delta = np.diff(spec, axis=1)
    freq_delta = np.diff(spec, axis=0)

    features['time_delta_mean'] = np.mean(np.abs(time_delta))
    features['time_delta_std'] = np.std(time_delta)
    features['time_delta_energy'] = np.sum(time_delta ** 2)

    features['freq_delta_mean'] = np.mean(np.abs(freq_delta))
    features['freq_delta_std'] = np.std(freq_delta)
    features['freq_delta_energy'] = np.sum(freq_delta ** 2)

    # Second-order delta (acceleration)
    if spec.shape[1] > 2:
        time_delta2 = np.diff(time_delta, axis=1)
        features['time_delta2_mean'] = np.mean(np.abs(time_delta2))
        features['time_delta2_std'] = np.std(time_delta2)

    return features


def extract_peak_features(spec):
    """Extract peak counts and peak positions"""
    features = {}

    # Peak detection over frequency axis (for each time frame)
    frame_peaks = []
    frame_peak_positions = []

    for t in range(spec.shape[1]):
        peaks, _ = find_peaks(spec[:, t], height=np.std(spec[:, t]))
        frame_peaks.append(len(peaks))
        if len(peaks) > 0:
            frame_peak_positions.extend(peaks)

    features['peak_count_mean'] = np.mean(frame_peaks)
    features['peak_count_std'] = np.std(frame_peaks) if len(frame_peaks) > 1 else 0
    features['peak_count_max'] = np.max(frame_peaks) if frame_peaks else 0

    if frame_peak_positions:
        features['peak_position_mean'] = np.mean(frame_peak_positions)
        features['peak_position_std'] = np.std(frame_peak_positions)
    else:
        features['peak_position_mean'] = 0
        features['peak_position_std'] = 0

    return features


def extract_logmel_features(spec, n_mels=128):
    """Extract log-mel spectrogram summary statistics"""
    features = {}

    # Convert to log scale
    spec_db = 10 * np.log10(np.maximum(spec, 1e-10))

    # Summary stats of log-mel
    features['logmel_mean'] = np.mean(spec_db)
    features['logmel_std'] = np.std(spec_db)
    features['logmel_max'] = np.max(spec_db)
    features['logmel_min'] = np.min(spec_db)
    features['logmel_median'] = np.median(spec_db)

    # Percentiles
    features['logmel_q25'] = np.percentile(spec_db, 25)
    features['logmel_q75'] = np.percentile(spec_db, 75)

    # Energy in log-mel space
    features['logmel_energy'] = np.sum(spec_db ** 2)

    return features
