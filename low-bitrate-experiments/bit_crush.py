import numpy as np
from scipy.io import wavfile

def bitcrusher(audio, bit_depth):
    max_val = 2**(bit_depth - 1)
    return np.round(audio * max_val) / max_val

# Read original WAV
rate, data = wavfile.read('input.wav')

# Ensure it's mono or take one channel
if data.ndim > 1:
    data = data[:, 0]

# Normalize to [-1, 1]
data_norm = data / np.max(np.abs(data))

# Apply to normalized signal
data_crushed = bitcrusher(data_norm, 4)  # e.g., 4-bit sound
