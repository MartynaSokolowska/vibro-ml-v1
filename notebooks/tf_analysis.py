import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import stft
import pywt
from scipy.stats import linregress
from scipy.stats import entropy


# NOTE: Could be changed to use GPU

def compute_fft(signal, fs):
    N = len(signal)
    X = fft(signal)
    freqs = fftfreq(N, 1/fs)
    return freqs[:N//2], np.abs(X[:N//2])

def compute_stft(signal, fs, nperseg=1024, noverlap=768, window='hann'):
    f, t, Zxx = stft(signal,
                     fs=fs,
                     window=window,
                     nperseg=nperseg,
                     noverlap=noverlap)
    return f, t, np.abs(Zxx)

def compute_cwt(signal, fs, scales=None, wavelet='morl'):
    if scales is None:
        scales = np.arange(1, 256)
    coeffs, freqs = pywt.cwt(signal,
                             scales,
                             wavelet,
                             sampling_period=1/fs)
    return freqs, np.abs(coeffs)

# ----------------------------------------------------
# 
# ----------------------------------------------------

def extract_time_features(signal):
    features = {}
    
    features["rms"] = np.sqrt(np.mean(signal**2))
    features["std"] = np.std(signal)
    features["ptp"] = np.ptp(signal)
    features["energy"] = np.sum(signal**2)
    features["zcr"] = np.mean(np.abs(np.diff(np.sign(signal)))) / 2
    features["crest_factor"] = np.max(np.abs(signal)) / features["rms"]

    return features

def spectral_slope(freqs, mag, fmin=100, fmax=8000):
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    log_freqs = np.log(freqs[idx] + 1e-12)
    log_mag = np.log(mag[idx] + 1e-12)
    slope, _, _, _, _ = linregress(log_freqs, log_mag)
    return slope

def spectral_rolloff(freqs, mag, roll_percent=0.85):
    energy = mag ** 2
    cumulative_energy = np.cumsum(energy)
    threshold = roll_percent * cumulative_energy[-1]
    idx = np.where(cumulative_energy >= threshold)[0][0]
    return freqs[idx]

def extract_fft_features(signal, sample_rate, band_width=100, show_bands=True):
    features = {}

    freqs, mag = compute_fft(signal, sample_rate)
    features["max_mag"] = np.max(mag)
    features["dom_freq"] = freqs[np.argmax(mag)]
    centroid = np.sum(freqs * mag) / (np.sum(mag) + 1e-12)
    features["spectral_centroid"] = centroid
    features["spectral_bandwidth"] = np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / (np.sum(mag) + 1e-12))
    features["spectral_slope"] = spectral_slope(freqs, mag, 0, 2500)
    features["spectral_rolloff_85%"] = spectral_rolloff(freqs, mag)
    if show_bands:
        for start in range(0, 35000, band_width):
            end = start + band_width
            features[f"{start}-{end}Hz_energy"] = np.sum(mag[start:end]**2)
    features["fft_energy_low"] = np.sum(mag[:2000]**2)
    features["fft_energy_mid"] = np.sum(mag[2000:5000]**2)
    features["fft_energy_high"] = np.sum(mag[5000:]**2)

    return features

def band_energy(f, Z, fmin, fmax):
    idx = np.logical_and(f >= fmin, f <= fmax)
    return np.mean(Z[idx, :]**2)

def extract_stft_features(signal, sample_rate, band_width=100, show_bands=True):
    features = {}

    f, t, Z = compute_stft(signal, sample_rate)
    features["stft_energy"] = np.mean(Z**2)
    features["stft_var"] = np.var(Z**2)
    features["stft_time_var"] = np.mean(np.var(Z**2, axis=0))
    centroid = np.sum(f[:, None] * Z, axis=0) / np.sum(Z, axis=0)
    features["stft_centroid_mean"] = np.mean(centroid)
    features["stft_centroid_std"] = np.std(centroid)
    if show_bands:
        for start in range(0, 24000, band_width):
            end = start + band_width
            features[f"{start}-{end}Hz_energy"] = band_energy(f, Z, start, end)
    features["energy_low"] = band_energy(f, Z, 0, 500)
    features["energy_mid"] = band_energy(f, Z, 500, 2000)
    features["energy_high"] = band_energy(f, Z, 2000, 25000)
    
    return features

def extract_cwt_features(signal, sample_rate, band_width=100, show_bands=True):
    features = {}

    freqs, C = compute_cwt(signal, sample_rate)
    prob = C / C.sum(axis=1, keepdims=True) 
    ent = entropy(prob, axis=1) 
    features["cwt_entropy"] = np.mean(ent)
    features["cwt_time_var"] = np.mean(np.var(C, axis=1))
    features["cwt_p95"] = np.percentile(C, 95)
    if show_bands:
        for start in range(0, 24000, band_width):
            end = start + band_width
            mask = (freqs >= start) & (freqs < end)
            if np.any(mask):
                band_energy = np.sum(C[mask,:]**2)
                features[f"{start}-{end}Hz_energy"] = band_energy
    features["energy_low"] = np.sum(C[freqs<500,:]**2)
    features["energy_mid"] = np.sum(C[(freqs>=500) & (freqs<5000),:]**2)
    features["energy_high"] = np.sum(C[freqs>=5000,:]**2)
    
    return features