"""Signal denoising functions"""
# NOTE: For consideration ...Deep Learning Denoising...
import pywt
import numpy as np
import librosa
from scipy.signal import wiener, medfilt, savgol_filter
from PyEMD import EMD


def wavelet_denoise(x, wavelet="db4", level=2, threshold_factor=0.04):
    coeffs = pywt.wavedec(x, wavelet, level=level, mode="per")
    cA, cD = coeffs[0], coeffs[1:]
    
    threshold = threshold_factor * np.max(cD[0])
    cD_thresh = [pywt.threshold(c, threshold, mode='soft') for c in cD]
    
    coeffs_thresh = [cA] + cD_thresh
    return pywt.waverec(coeffs_thresh, wavelet, mode="per")


def spectral_subtraction(x, fs, noise_clip=None, n_fft=256, hop_length=64):
    S = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(S)

    if noise_clip is None:
        noise_mag = np.mean(magnitude[:, :5], axis=1, keepdims=True)
    else:
        noise_mag = np.mean(np.abs(librosa.stft(noise_clip, n_fft=n_fft, hop_length=hop_length)), axis=1, keepdims=True)

    magnitude_denoised = np.maximum(magnitude - noise_mag, 0)
    S_denoised = magnitude_denoised * np.exp(1j * np.angle(S))
    return librosa.istft(S_denoised, hop_length=hop_length)



def wiener_filter(x, mysize=29, noise=None):
    return wiener(x, mysize=mysize, noise=noise)


def median_filter(x, kernel_size=3):
    return medfilt(x, kernel_size=kernel_size)


def savgol_denoise(x, window_length=51, polyorder=3):
    return savgol_filter(x, window_length=window_length, polyorder=polyorder)


def emd_denoise_threshold(x, max_imf=5, energy_threshold=0.05):
    emd = EMD()
    imfs = emd(x, max_imf=max_imf)

    energies = [np.sum(imf**2) for imf in imfs]
    total_energy = np.sum(energies)

    imfs_keep = [imf for imf, e in zip(imfs, energies) if e / total_energy > energy_threshold]

    return np.sum(imfs_keep, axis=0)


def emd_denoise(x, max_imf=5, drop_imf=1):
    emd = EMD()
    imfs = emd(x, max_imf=max_imf)
    return np.sum(imfs[drop_imf:], axis=0)


import numpy as np
import matplotlib.pyplot as plt
import torchaudio


class DenoiseTester:
    def __init__(self, path, sample_rate=48000):
        self.path = path
        self.sample_rate = sample_rate

        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True) 
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resampler(waveform)
        self.waveform = waveform.squeeze().numpy()
        self.sr = sample_rate

    def get_fragment(self, start_sec=None, end_sec=None):
        if start_sec is None or end_sec is None:
            return self.waveform
        start = int(start_sec * self.sr)
        end = int(end_sec * self.sr)
        return self.waveform[start:end]

    def apply(self, method="wavelet", **kwargs):
        x = self.waveform
        if method == "wavelet":
            return wavelet_denoise(x, **kwargs)
        elif method == "spectral_subtraction":
            return spectral_subtraction(x, fs=self.sr, **kwargs)
        elif method == "wiener":
            return wiener_filter(x, **kwargs)
        elif method == "median":
            return median_filter(x, **kwargs)
        elif method == "savgol":
            return savgol_denoise(x, **kwargs)
        
        elif method == "emd":
            return emd_denoise(x, **kwargs)
        elif method == "emd_threshold":
           return emd_denoise_threshold(x, **kwargs)
        
        else:
            raise ValueError(f"Unknown method: {method}")

    def plot(self, method="wavelet", start_sec=None, end_sec=None, **kwargs):
        fragment = self.get_fragment(start_sec, end_sec)
        denoised = self.apply(method=method, **kwargs)
        if start_sec is not None and end_sec is not None:
            denoised = denoised[int(start_sec*self.sr):int(end_sec*self.sr)]

        difference = fragment - denoised
        t = np.arange(len(fragment)) / self.sr

        fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        axs[0].plot(t, fragment, label="Original")
        axs[0].set_ylabel("Amplitude")
        axs[0].set_title("Original Signal")
        axs[0].legend()

        axs[1].plot(t, denoised, label=f"Denoised ({method})", color='green')
        axs[1].set_ylabel("Amplitude")
        axs[1].set_title(f"Denoised Signal ({method})")
        axs[1].legend()

        axs[2].plot(t, difference, label="Difference", color='red')
        axs[2].set_xlabel("Time [s]")
        axs[2].set_ylabel("Amplitude")
        axs[2].set_title("Difference (Original - Denoised)")
        axs[2].legend()

        plt.tight_layout()
        plt.show()

"""
path = "C:\\Users\\sokol\\OneDrive\\Pulpit\\SEM_8\\magisterka\\new_data_1\\data\\35\\35.5_Foam_Speed-10_24g_quincke_stethescope_2025-07-08_20.07.52.wav"
tester = DenoiseTester(path, sample_rate=48000)
tester.plot(method="savgol", start_sec=3, end_sec=4.7)
"""