"""Signal filtering functions"""
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(x, fs=48000, lowcut=50, highcut=8000, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, x)


def notch_filter(x, fs, freq=50.0, quality=30.0):
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = iirnotch(w0, quality)
    return filtfilt(b, a, x)


def highpass_filter(x, fs, lowcut=50, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='high')
    return filtfilt(b, a, x)


def lowpass_filter(x, fs, highcut=8000, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
    return filtfilt(b, a, x)


import torchaudio
import matplotlib.pyplot as plt
import numpy as np


def test_filters(path, filter_type="bandpass"):
    waveform, fs = torchaudio.load(path) 

    start_sec = 3.0  
    end_sec = 4.7  

    start_sample = int(start_sec * 48000)
    end_sample = int(end_sec * 48000)

    waveform = waveform[:, start_sample:end_sample] 

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    audio_np = waveform.squeeze().numpy()

    if filter_type == "bandpass":
        audio_filtered = bandpass_filter(audio_np, fs)
    elif filter_type == "highpass":
        audio_filtered = highpass_filter(audio_np, fs)
    elif filter_type == "lowpass":
        audio_filtered = lowpass_filter(audio_np, fs)
    elif filter_type == "notch":
        audio_filtered = notch_filter(audio_np, fs)
    else:
        raise ValueError("Unknown filter_type")

    difference = audio_np - audio_filtered

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(audio_np, color='blue')
    plt.title("Original Signal")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 2)
    plt.plot(audio_filtered, color='red')
    plt.title(f"Filtered Signal ({filter_type})")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 3)
    plt.plot(difference, color='green')
    plt.title("Difference (Original - Filtered)")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude Difference")

    plt.tight_layout()
    plt.show()

    return audio_filtered

test_filters("C:\\Users\\sokol\\OneDrive\\Pulpit\\SEM_8\\magisterka\\new_data_1\\data\\35\\35.5_Foam_Speed-10_24g_quincke_stethescope_2025-07-08_20.03.35.processed.wav")
