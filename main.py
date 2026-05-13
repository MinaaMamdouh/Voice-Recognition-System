import sounddevice as sd
from scipy.io.wavfile import write

# Record duration
duration = 5

# Sample rate
fs = 16000

print("Start Recording...")

# Record audio
recording = sd.rec(
    int(duration * fs),
    samplerate=fs,
    channels=1
)


sd.wait()

print("Recording Finished")

# Save audio
recording = (recording * 32767).astype('int16')
write("output.wav", fs, recording)

print("Audio Saved Successfully")

# =========================
# Load Audio
# =========================

import librosa
import matplotlib.pyplot as plt

signal, sr_rate = librosa.load("output.wav")

print("Sample Rate:", sr_rate)
print("Signal Shape:", signal.shape)

# =========================
# Plot Audio Signal
# =========================

plt.figure(figsize=(12,4))

plt.plot(signal)

plt.title("Audio Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.show()


# =========================
# Gaussian Filter
# =========================

from scipy.ndimage import gaussian_filter1d

smoothed_signal = gaussian_filter1d(signal,
                                    sigma=2)

plt.figure(figsize=(12,4))

plt.plot(signal,
         label='Original Signal')

plt.plot(smoothed_signal,
         label='Smoothed Signal')

plt.legend()

plt.title("Gaussian Filter")

plt.show()


# =========================
# Average Filter
# =========================

from scipy.ndimage import convolve1d
import numpy as np

window_size = 5

window = np.ones(window_size) / window_size

average_filtered_signal = convolve1d(signal,
                                     window,
                                     mode='nearest')

plt.figure(figsize=(12,4))

plt.plot(signal,
         label='Original Signal')

plt.plot(average_filtered_signal,
         label='Average Filter')

plt.legend()

plt.title("Average Filter")

plt.show()


# =========================
# Silence Removal
# =========================

trimmed_signal, index = librosa.effects.trim(
    smoothed_signal,
    top_db=20
)

plt.figure(figsize=(12,4))

plt.plot(trimmed_signal)

plt.title("Signal After Silence Removal")

plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.show()


# =========================
# Zero Crossing Rate (ZCR)
# =========================

zcr = librosa.feature.zero_crossing_rate(
    trimmed_signal
)

plt.figure(figsize=(12,4))

plt.plot(zcr[0])

plt.title("Zero Crossing Rate")

plt.xlabel("Frames")
plt.ylabel("ZCR")

plt.show()


# =========================
# Short Time Energy (STE)
# =========================

frame_length = 1024
hop_length = 512

energy = []

for i in range(0, len(trimmed_signal), hop_length):

    frame = trimmed_signal[i:i + frame_length]

    ste = np.sum(frame ** 2)

    energy.append(ste)

plt.figure(figsize=(12,4))

plt.plot(energy)

plt.title("Short Time Energy")

plt.xlabel("Frames")
plt.ylabel("Energy")

plt.show()


# =========================
# Level Crossing Rate (LCR)
# =========================

mean_level = np.mean(trimmed_signal)

lcr = np.sum(
    np.diff(trimmed_signal > mean_level)
)

print("LCR Value:", lcr)


# =========================
# MFCC Feature Extraction
# =========================

mfccs = librosa.feature.mfcc(
    y=trimmed_signal,
    sr=sr_rate,
    n_mfcc=13
)

print("MFCC Shape:", mfccs.shape)

plt.figure(figsize=(12,6))

librosa.display.specshow(
    mfccs,
    x_axis='time'
)

plt.colorbar()

plt.title("MFCC")

plt.show()

# =========================
# Spectrogram
# =========================

stft = librosa.stft(trimmed_signal)

spectrogram = librosa.amplitude_to_db(
    np.abs(stft)
)

plt.figure(figsize=(12,6))

librosa.display.specshow(
    spectrogram,
    sr=sr_rate,
    x_axis='time',
    y_axis='hz'
)

plt.colorbar()

plt.title("Spectrogram")

plt.show()


import speech_recognition as sr

recognizer = sr.Recognizer()

with sr.AudioFile("output.wav") as source:

    audio = recognizer.record(source)

try:


    text = recognizer.recognize_google(audio)

    print("Recognized Text:")
    print(text)

except:

    print("Could not recognize speech")