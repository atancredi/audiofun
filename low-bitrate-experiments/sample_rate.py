from scipy.signal import resample

import numpy as np
from scipy.io import wavfile


def downsample(audio, factor):
    return audio[::factor]

# Read original WAV
rate, data = wavfile.read('input.wav')

# Ensure it's mono or take one channel
if data.ndim > 1:
    data = data[:, 0]

# Normalize to [-1, 1]
data_norm = data / np.max(np.abs(data))


# Reduce sample rate from original to e.g., 8000 Hz
new_rate = 8000
num_samples = int(len(data_norm) * new_rate / rate)
resampled = resample(data_norm, num_samples)

# Upsample back to original rate (to save as WAV)
data_resampled = resample(resampled, len(data_norm))
wavfile.write('output_resampled.wav', rate, data_resampled)
