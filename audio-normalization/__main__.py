from typing import Optional
from logging import Logger
from tqdm import tqdm
from os import walk
from os.path import join, relpath, dirname, splitext

from fire import Fire

from core.dynamic import normalize_peak, get_audio, save_audio
from core.metadata import copy_metadata
from core.helper import ensure_dir
from core.my_logger import get_logger


class AudioNormalization(object):

    logger: Logger

    def __init__(self):
        self.logger = get_logger()
        self.supported_extensions = [".ogg", ".mp3", ".wav", ".aif", ".aiff"]

    def normalize(
        self,
        input_path: str,
        output_path: str,
        target: float = 1.0,
    ):
        self.logger.info(f"Starting normalization of {input_path} to {target} dB Peak")
        audio, ext = get_audio(input_path)
        normalized_audio, original_peak, normalized_peak = normalize_peak(
            audio, headroom=target
        )
        self.logger.info(
            f"Normalized audio peak from {original_peak} to {normalized_peak}"
        )
        save_audio(normalized_audio, output_path, ext)
        copy_metadata(input_path, output_path)

    def normalize_folder(
        self,
        input_folder: str,
        output_folder: Optional[str] = None,
        target: float = 1.0,
    ):
        self.logger.info(
            f"Starting normalization of all audio files in {input_folder} to {target} dB Peak - saving to {output_folder}"
        )

        files = [
            item
            for sublist in [
                [
                    (x[0], y)
                    for y in x[2]
                    if splitext(y.lower())[1] in self.supported_extensions
                ]
                for x in walk(input_folder)
            ]
            for item in sublist
        ]
        for f in tqdm(files):
            root = f[0]
            file = f[1]

            input_path = join(root, file)
            rel_path = relpath(root, input_folder)

            audio, ext = get_audio(input_path)
            normalized_audio, original_peak, normalized_peak = normalize_peak(
                audio, headroom=target
            )
            self.logger.info(
                f"Normalized audio peak from {original_peak} to {normalized_peak}"
            )
            fname, extension = splitext(file)

            output_path = join(output_folder, rel_path, fname + "_norm" + extension)
            ensure_dir(dirname(output_path))
            save_audio(normalized_audio, output_path, ext)
            copy_metadata(input_path, output_path)


if __name__ == "__main__":
    Fire(AudioNormalization)
