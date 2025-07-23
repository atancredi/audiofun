import numpy as np
from scipy.io import wavfile
from scipy import stats
from scipy.signal import fftconvolve
from scipy.signal import butter, lfilter
from scipy.signal import resample

class AudioFun:

    audio: np.ndarray
    original_audio: np.ndarray
    sample_rate: int

    def __init__(self, audio, sample_rate = 44100):
        self.audio = audio
        self.original_audio = audio
        self.sample_rate = sample_rate
    
    
    @staticmethod
    def read_file(filename):
        return wavfile.read(filename)


    @staticmethod
    def from_file(filename):
        sr, audio = wavfile.read(filename)
        return AudioFun(audio, sr)


    # UTILITY METHODS
    def get_audio_channel(self, channel=0):
        """
        Get a single audio channel from the current audio.

        Parameters:
            channel (int): Index of the channel
        """
        # Ensure it's mono or take one channel
        if self.audio.ndim > 1:
            self.audio = self.audio[:, channel]

        return self


    def set_sample_rate(self, sr: int):
        """
        Set the sample rate of the current audio (watch out!).

        Parameters:
            sr (int): Sample rate to set.
        """
        self.sample_rate = sr
        return self


    def save_audio(self, filename: str, clip=False):
        """
        Save audio to file

        Parameters:
            filename (str): Name of the file to save.
            clip (boolean): Apply clipping (default: False)
        """
        audio = self.audio
        if clip:
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767).astype(np.int16)
        wavfile.write(filename, self.sample_rate, audio)
        return self


    def normalize_to_peak_db(self, peak_db=-3.0):
        """
        Scales audio so that its peak reaches the target dB level.

        Parameters:
            peak_db (float): Target peak in decibels (default: -3 dBFS).
        """
        current_peak = np.max(np.abs(self.audio))
        if current_peak > 0: # Avoid division by zero
            target_peak_linear = 10 ** (peak_db / 20)
            scale = target_peak_linear / current_peak
            self.audio = self.audio * scale
        
        return self


    def normalize(self, max_interval=32767):
        """
        Normalizes audio in a custom interval

        Parameters:
            max_interval (int): Max absolute interval (default=32767)
        """
        self.audio = ((self.audio / np.max(np.abs(self.audio))) * max_interval).astype(np.int16)
        return self


    # EFFECTS
    def apply_gain_db(self, gain_db: float):
        """
        Applies gain to an audio signal in decibels.

        Parameters:
            gain_db (float): Gain in decibels. Positive = boost, Negative = cut.
        """
        gain_factor = 10 ** (gain_db / 20)
        self.audio = self.audio * gain_factor
        return self


    def bitcrush(self, bit_depth=4):
        """
        Bitcrusher effect

        Parameters:
            bit_depth (int)
        """
        max_val = 2 ** (bit_depth - 1)
        self.audio = np.round(self.audio * max_val) / max_val
        return self


    def wow_flutter(self, depth=0.002, speed=0.5):
        """
        Wow flutter effect

        Parameters:
            depth (float): (default: 0.002)
            speed (float): (default: 0.5)
        """
        t = np.linspace(0, len(self.audio) / self.sample_rate, len(self.audio))
        mod = 1 + depth * np.sin(2 * np.pi * speed * t)
        indices = np.clip((np.arange(len(self.audio)) * mod).astype(int), 0, len(self.audio) - 1)
        self.audio = self.audio[indices]
        return self


    def make_loop(self, loop_len_sec, n=5, starting_sample_index=0):
        """
        Loop audio

        Parameters:
            loop_len_sec (float): Lenght of the loop (in seconds)
            n (int): Number of loops 
            starting_sample_index (int): Index of the sample where to start the loop (default: 0)
        """
        
        loop_samples = int(loop_len_sec * self.sample_rate)
        loop = np.tile(self.audio[starting_sample_index:loop_samples], n)  # n loops
        self.audio = loop
        return self


    def apply_convolution(self, signal: np.ndarray):
        """
        Apply convolution with a signal

        Parameters:
            signal (np.ndarray): Signal to convolve the audio with
        """
        self.audio = fftconvolve(self.audio, signal, mode="full")[: len(self.audio)]
        return self


    def apply_batch_convolution(self, impulse_response, batch_size_ms):
        """
        Apply 'batch' convolution with a signal

        Parameters:
            signal (np.ndarray): Signal to convolve the audio with
            batch_size_ms (int): Size of the batch to use for convolution with the signal (in milliseconds)
            
        """
        reconstructed_audio = []
        batch_size_samples = round(self.sample_rate * batch_size_ms * 10 ** (-3))
        for i in range(round(len(self.audio)/batch_size_samples)):
            audio_batch = self.audio[batch_size_samples * i : batch_size_samples * (i+1)]
            convolved_audio_batch = fftconvolve(audio_batch, impulse_response, mode="full")[: len(audio_batch)]
            reconstructed_audio.extend(convolved_audio_batch)
        self.audio = reconstructed_audio
        return self


    def saturate(self, amount=1.5):
        """
        Saturation effect 

        Parameters:
            amount (float)
            
        """
        self.audio = np.tanh(self.audio * amount)
        return self


    # SAMPLE RATE
    def downsample_raw(self, factor):
        """
        Downsample (raw method) by an integer amount

        Parameters:
            factor (int)
        """
        self.audio = self.audio[::factor]
        return self
    

    def downsample(self, new_rate):
        """
        Downsample to a sample rate
        !! Remember to resample it back to the original sample rate before saving it to wav.

        Parameters:
            new_rate (int)
        """
        # Normalize to [-1, 1]
        data_norm = self.audio / np.max(np.abs(self.audio))

        num_samples = int(len(data_norm) * new_rate / self.sample_rate)
        self.audio = resample(data_norm, num_samples)
        return self


    # FILTERS
    # TODO it's not clear if in lp and hp filters the cutoffs are pass or cut tresholds. in bandpass too.
    def lowpass(self, cutoff=3000):
        """
        Low-Pass filter 

        Parameters:
            cutoff (int): Cut-off frequency
            
        """
        nyq = 0.5 * self.sample_rate
        b, a = butter(4, cutoff / nyq, btype="low")
        self.audio = lfilter(b, a, self.audio)
        return self
    

    def highpass(self, cutoff):
        """
        High-Pass filter 

        Parameters:
            cutoff (int): Cut-off frequency
            
        """
        nyq = 0.5 * self.sample_rate
        b, a = butter(4, cutoff / nyq, btype="high")
        self.audio = lfilter(b, a, self.audioaudio)
        return self


    def bandpass_filter(self, lowcut=2000, highcut=10000, order=4):
        """
        Band-Pass filter 

        Parameters:
            lowcut (int): Low cut-off frequency
            highcut (int): High cut-off frequency
            order (int): Order of the Butterworth filter
            
        """
        nyq = 0.5 * self.sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        self.audio = lfilter(b, a, self.audio)
        return self

