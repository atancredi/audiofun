from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from scipy.io import wavfile
from os import makedirs, listdir
from os.path import exists, split
from shutil import copy
from json import dump, JSONEncoder
import numpy as np

from clips import Clips

from audio_augmenters import (
    aggressive_augmenter,
    custom_augmenter_test,
    GeneralAugmentation,
)


class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)


def augment_clips(
    augmenter: GeneralAugmentation, clips_input_dir, output_dir, repeat: int, **kwargs
):
    clips = Clips(
        input_directory=clips_input_dir,
        file_pattern="*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,
    )
    clip_generator = clips.audio_generator(repeat=repeat, **kwargs)

    augmented_generator = augmenter.augment_generator(clip_generator)

    augmented_file_count = 0
    all_clips_with_parameters = []
    for augmented_clip_data, path_data in tqdm(
        augmented_generator, leave=False, desc="Augmenting clips"
    ):
        augmented_clip = augmented_clip_data[0]
        applied_parameters = augmented_clip_data[1]
        augmented_file_path = f"{output_dir}/augmented_{augmented_file_count}.wav"
        wavfile.write(augmented_file_path, 16000, augmented_clip)
        augmented_file_count += 1
        all_clips_with_parameters.append(
            {
                "path": augmented_file_path,
                "source_path": path_data,
                "parameters": applied_parameters,
            }
        )
    dump(
        all_clips_with_parameters,
        open(f"{output_dir}/applied_parameters_per_clip.json", "w+"),
        cls=NumpyEncoder,
    )


p = ArgumentParser()
p.add_argument("clips_folder")
p.add_argument("out_folder")
p.add_argument("--repeat", required=False, default=1, type=int)
p.add_argument("--include-originals", required=False, default=False, type=bool)

from random import uniform

def apply_amplitude_modulation(audio_data, samplerate):
    duration = len(audio_data) / samplerate
    # Create a time array corresponding to each sample in the audio data
    t = np.linspace(0, duration, len(audio_data), endpoint=False)

    # Calculate the modulation signal based on the provided function
    mod_f  = uniform(0.5, 1)
    print("applying modulation freq of", mod_f)
    modulation_signal = sine_amplitude_mod(t, mod_f)
    # modulation_signal = linear_amplitude_mod(t) #NOSONAR

    # define modulation signal application ranges
    apply_threshold = int(len(audio_data) / 2)
    print("Duration: ", duration, "n samples", len(audio_data))
    print("threshold: ", apply_threshold)
    truncated_modulation_signal = [1.0] * (len(audio_data) - apply_threshold)
    truncated_modulation_signal.extend(modulation_signal[apply_threshold:])

    # Apply the modulation by multiplying the audio data with the modulation signal
    modulated_audio = audio_data * truncated_modulation_signal

    print("n of clipping samples", len([x for x in modulated_audio if x > 1.0]))
    # Clip the audio to prevent values exceeding the valid range [-1.0, 1.0]
    modulated_audio = np.clip(modulated_audio, -1.0, 1.0)

    print("MAX AMPLITUDE, ", max(modulated_audio))
    print()
    return modulated_audio


def sine_amplitude_mod(t, mod_freq_hz = 1.0):
    # Ensure the modulation factor is always positive to avoid phase inversion
    return 1.0 + 1 * np.sin(2 * np.pi * mod_freq_hz * t)


def linear_amplitude_mod(t):
    return 1.0 + t * 1.15


if __name__ == "__main__":

    args = p.parse_args()

    # define folder of clips to augment and where to output
    clips_folder = Path(args.clips_folder)
    out_folder = Path(args.out_folder)
    rep = args.repeat
    include_originals = args.include_originals

    print("clips folder:", clips_folder)
    print("out folder:", out_folder)

    if not exists(out_folder):
        makedirs(out_folder)

    # if not wav convert them

    # augment audio files in the folder
    # augmenter = aggressive_augmenter()
    augmenter = custom_augmenter_test(apply_amplitude_modulation)
    augment_clips(augmenter, clips_folder, out_folder, rep)

    # include originals
    if include_originals:
        for p in listdir(clips_folder):
            _ = split(p)
            if _[1].endswith(".wav"):
                copy(clips_folder / p, out_folder / _[1])
