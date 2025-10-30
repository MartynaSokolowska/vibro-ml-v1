import numpy as np
import matplotlib.pyplot as plt
import librosa
from preprocessing.filters import *
from preprocessing.denoising import *


class AudioPipeline:
    def __init__(self, signal, sr):
        self.signal, self.fs = signal, sr
        self.processed = self.signal.copy()

    def apply_filter(self, filter_type="bandpass", **kwargs):
        if filter_type == "bandpass":
            self.processed = bandpass_filter(self.signal, fs=self.fs, **kwargs)
        elif filter_type == "notch":
            self.processed = notch_filter(self.signal, fs=self.fs, **kwargs)
        elif filter_type == "highpass":
            self.processed = highpass_filter(self.signal, fs=self.fs, **kwargs)
        elif filter_type == "lowpass":
            self.processed = lowpass_filter(self.signal, fs=self.fs, **kwargs)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        return self.processed

    def apply_denoising(self, method="wavelet", **kwargs):
        if self.processed is None:
            x = self.signal
        else:
            x = self.processed

        if method == "wavelet":
            self.processed = wavelet_denoise(x, **kwargs)
        elif method == "spectral":
            self.processed = spectral_subtraction(x, fs=self.fs, **kwargs)
        elif method == "wiener":
            self.processed = wiener_filter(x, **kwargs)
        elif method == "median":
            self.processed = median_filter(x, **kwargs)
        elif method == "savgol":
            self.processed = savgol_denoise(x, **kwargs)
        else:
            raise ValueError(f"Unknown denoising method: {method}")
        return self.processed

    def run_from_config(self, config):
        for f in config.get("filters", []):
            self.processed = self.apply_filter(filter_type=f["type"], **f.get("kwargs", {}))

        for d in config.get("denoising", []):
            self.processed = self.apply_denoising(method=d["method"], **d.get("kwargs", {}))

        norm_mode = config.get("normalize", None)

        if norm_mode == "peak":
            self.processed = (self.processed - np.mean(self.processed)) / (np.max(np.abs(self.processed)) + 1e-8)
        elif norm_mode == "zscore":
            self.processed = (self.processed - np.mean(self.processed)) / (np.std(self.processed) + 1e-8)

        return self.processed

    def plot_results(self):
        if self.processed is None:
            raise ValueError("No processing applied yet.")

        t = np.arange(len(self.signal)) / self.fs
        difference = self.signal - self.processed

        plt.figure(figsize=(14, 8))
        plt.plot(t, self.signal, label="Original", alpha=0.7)
        plt.plot(t, self.processed, label="Processed", alpha=0.7)
        plt.plot(t, difference, label="Difference", alpha=0.7)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title("Pipeline: Original vs Processed vs Difference")
        plt.legend()
        plt.tight_layout()
        plt.show()


"""
if __name__ == "__main__":
    pipeline = AudioPipeline("C:\\Users\\sokol\\OneDrive\\Pulpit\\SEM_8\\magisterka\\new_data_1\\data\\35\\35.5_Foam_Speed-10_24g_quincke_stethescope_2025-07-08_20.07.52.wav")
    config = load_config()
    processed = pipeline.run_from_config(config["preprocessing"])
    pipeline.plot_results()
"""