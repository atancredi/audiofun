# coding=utf-8
# Copyright 2024 Kevin Ahrendt.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import audio_metadata
import datasets
import math
import os
import random
import wave
from tqdm import tqdm

import numpy as np

from pathlib import Path

from audio_utils import remove_silence_webrtc


class Clips:
    """Class for loading audio clips from the specified directory. The clips can first be filtered by their duration using the `min_clip_duration_s` and `max_clip_duration_s` parameters. Clips are retrieved as numpy float arrays via the `get_random_clip` method or via the `audio_generator` or `random_audio_generator` generators. Before retrieval, the audio clip can trim non-voice activiity. Before retrieval, the audio clip can be repeated until it is longer than a specified minimum duration.

    Args:
        input_directory (str): Path to audio clip files.
        file_pattern (str): File glob pattern for selecting audio clip files.
        min_clip_duration_s (float | None, optional): The minimum clip duration (in seconds). Set to None to disable filtering by minimum clip duration. Defaults to None.
        max_clip_duration_s (float | None, optional): The maximum clip duration (in seconds). Set to None to disable filtering by maximum clip duration. Defaults to None.
        repeat_clip_min_duration_s (float | None, optional): If a clip is shorter than this duration, then it is repeated until it is longer than this duration. Set to None to disable repeating the clip. Defaults to None.
        remove_silence (bool, optional): Use webrtcvad to trim non-voice activity in the clip. Defaults to False.
        random_split_seed (int | None, optional): The random seed used to split the clips into different sets. Set to None to disable splitting the clips. Defaults to None.
        split_count (int | float, optional): The percentage/count of clips to be included in the testing and validation sets. Defaults to 0.1.
        trimmed_clip_duration_s: (float | None, optional): The duration of the clips to trim the end of long clips. Set to None to disable trimming. Defaults to None.
        trim_zerios: (bool, optional): If true, any leading and trailling zeros are removed. Defaults to false.
    """

    def __init__(
        self,
        input_directory: str,
        file_pattern: str,
        min_clip_duration_s: float | None = None,
        max_clip_duration_s: float | None = None,
        repeat_clip_min_duration_s: float | None = None,
        remove_silence: bool = False,
        random_split_seed: int | None = None,
        split_count: int | float = 0.1,
        trimmed_clip_duration_s: float | None = None,
        trim_zeros: bool = False,
    ):
        self.trim_zeros = trim_zeros
        self.trimmed_clip_duration_s = trimmed_clip_duration_s

        if min_clip_duration_s is not None:
            self.min_clip_duration_s = min_clip_duration_s
        else:
            self.min_clip_duration_s = 0.0

        if max_clip_duration_s is not None:
            self.max_clip_duration_s = max_clip_duration_s
        else:
            self.max_clip_duration_s = math.inf

        if repeat_clip_min_duration_s is not None:
            self.repeat_clip_min_duration_s = repeat_clip_min_duration_s
        else:
            self.repeat_clip_min_duration_s = 0.0

        self.remove_silence = remove_silence

        self.remove_silence_function = remove_silence_webrtc

        paths_to_clips = [str(i) for i in Path(input_directory).glob(file_pattern)]

        if (self.min_clip_duration_s == 0) and (math.isinf(self.max_clip_duration_s)):
            # No durations specified, so do not filter by length
            filtered_paths = paths_to_clips
        else:
            # Filter audio clips by length
            if file_pattern.endswith("wav"):
                # If it is a wave file, assume all wave files have the same parameters and filter by file size.
                # Based on openWakeWord's estimate_clip_duration and filter_audio_paths in data.py, accessed March 2, 2024.
                with wave.open(paths_to_clips[0], "rb") as input_wav:
                    channels = input_wav.getnchannels()
                    sample_width = input_wav.getsampwidth()
                    sample_rate = input_wav.getframerate()
                    frames = input_wav.getnframes()

                sizes = []
                sizes.extend([os.path.getsize(i) for i in paths_to_clips])

                # Correct for the wav file header bytes. Assumes all files in the directory have same parameters.
                header_correction = (
                    os.path.getsize(paths_to_clips[0])
                    - frames * sample_width * channels
                )

                durations = []
                for size in sizes:
                    durations.append(
                        (size - header_correction)
                        / (sample_rate * sample_width * channels)
                    )

                filtered_paths = [
                    path_to_clip
                    for path_to_clip, duration in zip(paths_to_clips, durations)
                    if (self.min_clip_duration_s < duration)
                    and (duration < self.max_clip_duration_s)
                ]
            else:
                # If not a wave file, use the audio_metadata package to analyze audio file headers for the duration.
                # This is slower!
                filtered_paths = []

                if (self.min_clip_duration_s > 0) or (
                    not math.isinf(self.max_clip_duration_s)
                ):
                    for audio_file in paths_to_clips:
                        metadata = audio_metadata.load(audio_file)
                        duration = metadata["streaminfo"]["duration"]
                        if (self.min_clip_duration_s < duration) and (
                            duration < self.max_clip_duration_s
                        ):
                            filtered_paths.append(audio_file)

        # Load all filtered clips
        audio_dataset = datasets.Dataset.from_dict(
            {"audio": [str(i) for i in filtered_paths]}
        ).cast_column("audio", datasets.Audio())

        # Convert all clips to 16 kHz sampling rate when accessed
        audio_dataset = audio_dataset.cast_column(
            "audio", datasets.Audio(sampling_rate=16000)
        )

        if random_split_seed is not None:
            train_testvalid = audio_dataset.train_test_split(
                test_size=2 * split_count, seed=random_split_seed
            )
            test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
            split_dataset = datasets.DatasetDict(
                {
                    "train": train_testvalid["train"],
                    "test": test_valid["test"],
                    "validation": test_valid["train"],
                }
            )
            self.split_clips = split_dataset

        self.clips = audio_dataset
        print(f"LOADED CLIPS {self.clips.num_columns}x{self.clips.num_rows}")


    def audio_generator(self, split: str | None = None, repeat: int = 1):
        """A Python generator that retrieves all loaded audio clips.

        Args:
            split (str | None, optional): Specifies which set the clips are retrieved from. If None, all clips are retrieved. Otherwise, it can be set to `train`, `test`, or `validation`. Defaults to None.
            repeat (int, optional): The number of times each audio clip will be yielded. Defaults to 1.

        Yields:
            numpy.ndarray: Array with the audio clip's samples.
        """
        if split is None:
            clip_list = self.clips
        else:
            clip_list = self.split_clips[split]
        for _ in range(repeat):
            for clip in clip_list:
                clip_audio = clip["audio"]["array"]

                # ADDED 04/07/2025
                # get and return clip path and repetition number
                clip_path = clip["audio"]["path"]

                if self.remove_silence:
                    clip_audio = self.remove_silence_function(clip_audio)

                if self.trim_zeros:
                    clip_audio = np.trim_zeros(clip_audio)

                if self.trimmed_clip_duration_s:
                    total_samples = int(self.trimmed_clip_duration_s * 16000)
                    clip_audio = clip_audio[:total_samples]

                clip_audio = self.repeat_clip(clip_audio)
                
                # ADDED 04/07/2025
                # get and return clip path and repetition number
                yield clip_audio, (clip_path, _)


    # TODO sono sicuro che non posso dare due volte la stessa clip? secondo me puo' succedere, e non va bene
    def get_random_clip(self):
        """Retrieves a random audio clip.

        Returns:
            numpy.ndarray: Array with the audio clip's samples.
        """
        rand_audio_entry = random.choice(self.clips)
        clip_audio = rand_audio_entry["audio"]["array"]

        if self.remove_silence:
            clip_audio = self.remove_silence_function(clip_audio)

        if self.trim_zeros:
            clip_audio = np.trim_zeros(clip_audio)

        if self.trimmed_clip_duration_s:
            total_samples = int(self.trimmed_clip_duration_s * 16000)
            clip_audio = clip_audio[:total_samples]

        clip_audio = self.repeat_clip(clip_audio)

        # ADDED 04/07/2025
        # get and return clip path and repetition number
        clip_path = rand_audio_entry["audio"]["path"]
        return clip_audio, (clip_path)
        # return clip_audio


    def random_audio_generator(self, max_clips: int = math.inf):
        """A Python generator that retrieves random audio clips.

        Args:
            max_clips (int, optional): The total number of clips the generator will yield before the StopIteration. Defaults to math.inf.

        Yields:
            numpy.ndarray: Array with the random audio clip's samples.
        """
        # while max_clips > 0:
        for _ in tqdm(range(self.clips.num_rows), leave=False, desc="Generating random audio"):
            max_clips -= 1
            if max_clips == 0:
                return

            # n_retrieved_clips += 1
            # if n_retrieved_clips == self.clips.num_rows:
                # raise StopIteration()

            yield self.get_random_clip()

    def repeat_clip(self, audio_samples: np.array):
        """Repeats the audio clip until its duration exceeds the minimum specified in the class.

        Args:
            audio_samples numpy.ndarray: Original audio clip's samples.

        Returns:
            numpy.ndarray: Array with duration exceeding self.repeat_clip_min_duration_s.
        """
        original_clip = audio_samples
        desired_samples = int(self.repeat_clip_min_duration_s * 16000)
        while audio_samples.shape[0] < desired_samples:
            audio_samples = np.append(audio_samples, original_clip)
        return audio_samples


class Clip:
    """Class for loading audio clips from the specified directory. The clips can first be filtered by their duration using the `min_clip_duration_s` and `max_clip_duration_s` parameters. Clips are retrieved as numpy float arrays via the `get_random_clip` method or via the `audio_generator` or `random_audio_generator` generators. Before retrieval, the audio clip can trim non-voice activiity. Before retrieval, the audio clip can be repeated until it is longer than a specified minimum duration.

    Args:
        input_directory (str): Path to audio clip files.
        file_pattern (str): File glob pattern for selecting audio clip files.
        repeat_clip_min_duration_s (float | None, optional): If a clip is shorter than this duration, then it is repeated until it is longer than this duration. Set to None to disable repeating the clip. Defaults to None.
        remove_silence (bool, optional): Use webrtcvad to trim non-voice activity in the clip. Defaults to False.
        random_split_seed (int | None, optional): The random seed used to split the clips into different sets. Set to None to disable splitting the clips. Defaults to None.
        split_count (int | float, optional): The percentage/count of clips to be included in the testing and validation sets. Defaults to 0.1.
        trimmed_clip_duration_s: (float | None, optional): The duration of the clips to trim the end of long clips. Set to None to disable trimming. Defaults to None.
        trim_zerios: (bool, optional): If true, any leading and trailling zeros are removed. Defaults to false.
    """

    def setup(self,

        repeat_clip_min_duration_s: float | None = None,
        
        remove_silence: bool = False,
        # random_split_seed: int | None = None,
        # split_count: int | float = 0.1,
        trimmed_clip_duration_s: float | None = None,
        trim_zeros: bool = False
    ):
    
        self.trim_zeros = trim_zeros
        self.trimmed_clip_duration_s = trimmed_clip_duration_s

        if repeat_clip_min_duration_s is not None:
            self.repeat_clip_min_duration_s = repeat_clip_min_duration_s
        else:
            self.repeat_clip_min_duration_s = 0.0

        self.remove_silence = remove_silence

        self.remove_silence_function = remove_silence_webrtc


    def __init__(
        self,
        clip_path: str,

        repeat_clip_min_duration_s: float | None = None,
        
        remove_silence: bool = False,
        random_split_seed: int | None = None,
        split_count: int | float = 0.1,
        trimmed_clip_duration_s: float | None = None,
        trim_zeros: bool = False,
    ):
        
        # self.setup(repeat_clip_min_duration_s, remove_silence, random_split_seed, split_count, trimmed_clip_duration_s, trim_zeros)
        self.setup(repeat_clip_min_duration_s, remove_silence, trimmed_clip_duration_s, trim_zeros)

        # paths_to_clips = [str(i) for i in Path(input_directory).glob(file_pattern)]
        paths_to_clips = [clip_path]

        # Load all filtered clips
        audio_dataset = datasets.Dataset.from_dict(
            {"audio": [str(i) for i in paths_to_clips]}
        ).cast_column("audio", datasets.Audio())

        # Convert all clips to 16 kHz sampling rate when accessed
        audio_dataset = audio_dataset.cast_column(
            "audio", datasets.Audio(sampling_rate=16000)
        )

        self.clips = audio_dataset
        print(f"LOADED CLIPS {self.clips.num_columns}x{self.clips.num_rows}")


    # XXX yield sequentially truncated part of clip
    def audio_generator(self, split: str | None = None, repeat: int = 1):
        """A Python generator that retrieves all loaded audio clips.

        Args:
            split (str | None, optional): Specifies which set the clips are retrieved from. If None, all clips are retrieved. Otherwise, it can be set to `train`, `test`, or `validation`. Defaults to None.
            repeat (int, optional): The number of times each audio clip will be yielded. Defaults to 1.

        Yields:
            numpy.ndarray: Array with the audio clip's samples.
        """
        if split is None:
            clip_list = self.clips
        else:
            clip_list = self.split_clips[split]
        for _ in range(repeat):
            for clip in clip_list:
                clip_audio = clip["audio"]["array"]

                # ADDED 04/07/2025
                # get and return clip path and repetition number
                clip_path = clip["audio"]["path"]

                if self.remove_silence:
                    clip_audio = self.remove_silence_function(clip_audio)

                if self.trim_zeros:
                    clip_audio = np.trim_zeros(clip_audio)

                if self.trimmed_clip_duration_s:
                    total_samples = int(self.trimmed_clip_duration_s * 16000)
                    clip_audio = clip_audio[:total_samples]

                clip_audio = self.repeat_clip(clip_audio)
                
                # ADDED 04/07/2025
                # get and return clip path and repetition number
                yield clip_audio, (clip_path, _)


    # XXX yield randomly truncated part of clip
    def random_audio_generator(self, max_clips: int = math.inf):
        """A Python generator that retrieves random audio clips.

        Args:
            max_clips (int, optional): The total number of clips the generator will yield before the StopIteration. Defaults to math.inf.

        Yields:
            numpy.ndarray: Array with the random audio clip's samples.
        """

        # get first (and only clip)
        clip = self.clips[0]

        # 16kHz array of samples
        clip_audio = clip["audio"]["array"]
        
        n_sec = 3.2
        s_r = 16000 # ok because it's previously converted and casted
        randomly_selcted_samples = [(clip_audio[i:int((n_sec*s_r)) * (i+1)], (i, (n_sec*s_r) * (i+1))) for i in range(int(len(clip_audio)/(n_sec*s_r)))]

        for v in tqdm(randomly_selcted_samples, leave=False, desc="Getting randomly splitted audio"):
            random_clip = v[0]
            rng = v[1]
            # get and return clip path and repetition number
            yield random_clip, (clip["audio"]["path"], rng)

