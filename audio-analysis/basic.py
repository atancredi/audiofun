import numpy as np
import matplotlib.pyplot as plt

# Basic audio analysis
def plot_waveform(audio, rate, title="Waveform", seconds=None, output_path=None):
    """
    Plots amplitude vs time for a 1D audio array.

    Parameters:
        audio (np.ndarray): Audio signal (mono).
        rate (int): Sample rate (Hz).
        title (str): Plot title.
        seconds (float or None): Duration (in seconds) to display. If None, show full.
    """
    if seconds:
        samples = int(seconds * rate)
        audio = audio[:samples]

    times = np.arange(len(audio)) / rate

    plt.figure(figsize=(12, 4))
    plt.plot(times, audio, linewidth=0.5)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    if output_path == None:
        plt.show()
    else:
        plt.savefig(output_path)

