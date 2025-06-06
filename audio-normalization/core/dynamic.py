import math
from os import devnull
from os.path import splitext
from subprocess import run

from pydub import AudioSegment, effects


### PyDub FUNCTIONS ##############################
def get_audio(input_path: str):
    """
    Open audio file and get the AudioSegment object.
    """
    ext = splitext(input_path)[1][1:].lower()
    return AudioSegment.from_file(input_path, format=ext), ext


def save_audio(audio: AudioSegment, output_path: str, ext: str):
    """
    Save audio in the AudioSegment to 'output_path'.
    """
    audio.export(output_path, format=ext)


def get_audio_peak_db(audio: AudioSegment):
    """
    Get true peak (dB) for an AudioSegment.
    """
    peak_amplitude = audio.max
    if peak_amplitude == 0:
        return -float("inf")  # silence
    peak_ratio = peak_amplitude / audio.max_possible_amplitude
    peak_db = 20 * math.log10(peak_ratio)
    return round(peak_db, 2)


def normalize_peak(audio: AudioSegment, headroom=0.1):
    """
    Normalize peak (dB) to 'headroom' value for an AudioSegment.
    """
    original_peak = get_audio_peak_db(audio)
    normalized_audio = effects.normalize(audio, headroom=headroom)
    normalized_peak = get_audio_peak_db(normalized_audio)
    return normalized_audio, original_peak, normalized_peak
