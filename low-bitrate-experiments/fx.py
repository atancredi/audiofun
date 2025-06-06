import numpy as np
from scipy.io import wavfile
from scipy import stats
from scipy.signal import fftconvolve
from scipy.signal import butter, lfilter


def normalize(audio, max_interval=32767):
    return ((audio / np.max(np.abs(audio))) * max_interval).astype(np.int16)

def get_audio_channel(path, channel=0):
    rate, data = wavfile.read(path)

    # Ensure it's mono or take one channel
    if data.ndim > 1:
        data = data[:, channel]

    return rate, data


def save_audio(filename, rate, audio):
    audio_final = audio
    wavfile.write(filename, rate, audio_final)


def get_noise(length_in_seconds, sample_rate=44100, amplitude=11):
    data = (
        stats.truncnorm(-1, 1, scale=min(2**16, 2**amplitude))
        .rvs(sample_rate * length_in_seconds)
        .astype(np.int16)
    )
    data_norm = data / np.max(np.abs(data))
    return sample_rate, data_norm


def bitcrush(audio, bit_depth=4):
    max_val = 2 ** (bit_depth - 1)
    return np.round(audio * max_val) / max_val


def downsample(audio, factor):
    return audio[::factor]


def wow_flutter(audio, rate, depth=0.002, speed=0.5):
    t = np.linspace(0, len(audio) / rate, len(audio))
    mod = 1 + depth * np.sin(2 * np.pi * speed * t)
    indices = np.clip((np.arange(len(audio)) * mod).astype(int), 0, len(audio) - 1)
    return audio[indices]


def make_loop(audio, loop_len_sec, rate, n=5):
    loop_samples = int(loop_len_sec * rate)
    loop = np.tile(audio[:loop_samples], n)  # n loops
    return loop


def apply_convolution(audio, impulse_response):
    return fftconvolve(audio, impulse_response, mode="full")[: len(audio)]


def apply_batch_convolution(audio, impulse_response, sample_rate, batch_size_ms):
    reconstructed_audio = []
    batch_size_samples = round(sample_rate * batch_size_ms * 10 ** (-3))
    print(batch_size_samples)
    print("SSS", min(list(audio)), max(list(audio)))
    for i in range(round(len(audio)/batch_size_samples)):
        audio_batch = audio[batch_size_samples * i : batch_size_samples * (i+1)]
        convolved_audio_batch = fftconvolve(audio_batch, impulse_response, mode="full")[: len(audio_batch)]
        reconstructed_audio.extend(convolved_audio_batch)
    print(len(audio), len(reconstructed_audio))
    return reconstructed_audio

def get_impulse_response(ir_path, audio_lenght, ir_right_padding_ms=0):
    sample_rate, ir = wavfile.read(ir_path)

    print("ir lenght", len(ir), "audio lenght", audio_lenght)

    # Normalize IR to float32 range [-1, 1] if it's integer-encoded
    if np.issubdtype(ir.dtype, np.integer):
        max_val = np.iinfo(ir.dtype).max
        ir = ir.astype(np.float32) / max_val
    else:
        ir = ir.astype(np.float32)

    # Convert stereo IR to mono by averaging channels
    if ir.ndim > 1:
        ir = ir.mean(axis=1)

    # add padding to ir
    ir = np.pad(
        ir,
        (0, int(ir_right_padding_ms * (10 ** (-3)) * sample_rate)),
        constant_values=(0),
    )
    print("padded lenght", len(ir))

    if len(ir) > audio_lenght:
        ir = ir[:audio_lenght]
    elif len(ir) < audio_lenght:
        repeats = int(np.ceil(audio_lenght / len(ir)))
        tiled = np.tile(ir, repeats)
        i = 0
        while len(tiled) > audio_lenght:
            i += 1
            tiled = np.tile(ir, repeats - i)
        ir = tiled[:audio_lenght]

    return sample_rate, ir


def saturate(audio, amount=1.5):
    return np.tanh(audio * amount)


def lowpass(audio, rate, cutoff=3000):
    b, a = butter(4, cutoff / (0.5 * rate), btype="low")
    return lfilter(b, a, audio)


def bandpass_filter(audio, rate, lowcut=2000, highcut=10000, order=4):
    nyq = 0.5 * rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, audio)
