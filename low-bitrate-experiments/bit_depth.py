import numpy as np
from scipy.io import wavfile

# Read original WAV
rate, data = wavfile.read('input.wav')

# Ensure it's mono or take one channel
if data.ndim > 1:
    data = data[:, 0]

# Normalize to [-1, 1]
data_norm = data / np.max(np.abs(data))

# Reduce bit depth to 8-bit
bit_depth = 8
max_int = 2**(bit_depth - 1) - 1
data_8bit = (data_norm * max_int).astype(np.int8)

# Save as 8-bit WAV (we still have to trick scipy into doing this)
wavfile.write('output_8bit.wav', rate, data_8bit)
