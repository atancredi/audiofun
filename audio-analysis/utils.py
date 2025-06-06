import numpy as np
from scipy.io import wavfile

# Audio utility functions
def get_audio_channel(path, channel=0):
    rate, data = wavfile.read(path)

    # Ensure it's mono or take one channel
    if data.ndim > 1:
        data = data[:, channel]

    # Normalize if needed
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max

    return rate, data


def save_audio(filename, rate, audio, normalize=True):
    audio_final = audio
    if normalize:
        audio_final = ((audio / np.max(np.abs(audio))) * 32767).astype(np.int16)
    wavfile.write(filename, rate, audio_final)


def trim_silence(audio_channel, threshold=1e-4):
    """
    Removes silence from the start and end of a single-channel (mono) audio array.

    Parameters:
        audio (np.ndarray): Input audio array (mono).
        threshold (float): Amplitude threshold below which is considered silence.

    Returns:
        np.ndarray: Trimmed audio array.
    """

    # Absolute value to find non-silent regions
    mask = np.abs(audio_channel) > threshold

    if not np.any(mask):
        # Return empty if completely silent
        return np.array([], dtype=audio_channel.dtype)

    start = np.argmax(mask)
    end = len(mask) - np.argmax(mask[::-1])

    return audio_channel[start:end]
