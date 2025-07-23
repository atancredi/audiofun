import numpy as np
from scipy.io import wavfile
from scipy import stats

from typing import Tuple


def get_noise(
    length_in_seconds, sample_rate=44100, amplitude=11
) -> Tuple[int, np.ndarray]:
    data = (
        stats.truncnorm(-1, 1, scale=min(2**16, 2**amplitude))
        .rvs(sample_rate * length_in_seconds)
        .astype(np.int16)
    )
    data_norm: np.ndarray = data / np.max(np.abs(data))
    return sample_rate, data_norm


def get_impulse_response(ir_path, audio_lenght, ir_right_padding_ms=0):
    sample_rate, ir = wavfile.read(ir_path)

    # print("ir lenght", len(ir), "audio lenght", audio_lenght)

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
    # print("padded lenght", len(ir))

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
