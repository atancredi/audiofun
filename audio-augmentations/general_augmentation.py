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

import audiomentations
import warnings

import numpy as np

from typing import List


class GeneralAugmentation:
    """A class that handles applying augmentations to audio clips.

    Args:
        augmentation_duration_s (float): The duration of the augmented clip in seconds.
        augmentation_probabilities (dict, optional): Dictionary that specifies each augmentation's probability of being applied. Defaults to { "SevenBandParametricEQ": 0.0, "TanhDistortion": 0.0, "PitchShift": 0.0, "BandStopFilter": 0.0, "AddColorNoise": 0.25, "AddBackgroundNoise": 0.75, "Gain": 1.0, "GainTransition": 0.25, "RIR": 0.5, }.
        impulse_paths (List[str], optional): List of directory paths that contain room impulse responses that the audio clip is reverberated with. If the list is empty, then reverberation is not applied. Defaults to [].
        background_paths (List[str], optional): List of directory paths that contain audio clips to be mixed into the audio clip. If the list is empty, then the background augmentation is not applied. Defaults to [].
        background_min_snr_db (int, optional): The minimum signal to noise ratio for mixing in background audio. Defaults to -10.
        background_max_snr_db (int, optional): The maximum signal to noise ratio for mixing in background audio. Defaults to 10.
        min_gain_db (float, optional): The minimum gain for the gain augmentation. Defaults to -45.0.
        max_gain_db (float, optional): The mmaximum gain for the gain augmentation. Defaults to 0.0.
        min_gain_transition_db (float, optional): The minimum gain for the gain transition augmentation. Defaults to -10.0.
        max_gain_transition_db (float, optional): The mmaximum gain for the gain transition augmentation. Defaults to 10.0.
        min_jitter_s (float, optional): The minimum duration in seconds that the original clip is positioned before the end of the augmented audio. Defaults to 0.0.
        max_jitter_s (float, optional): The maximum duration in seconds that the original clip is positioned before the end of the augmented audio. Defaults to 0.0.
        truncate_randomly: (bool, option): If true, the clip is truncated to the specified duration randomly. Otherwise, the start of the clip is truncated.
    """

    def __init__(
        self,
        augment: audiomentations.Compose,
        augmentation_duration_s: float | None = None,
        min_jitter_s: float = 0.0,
        max_jitter_s: float = 0.0,
        truncate_randomly: bool = False,
    ):
        self.truncate_randomly = truncate_randomly
        ############################################
        # Configure audio duration and positioning #
        ############################################

        self.min_jitter_samples = int(min_jitter_s * 16000)
        self.max_jitter_samples = int(max_jitter_s * 16000)

        if augmentation_duration_s is not None:
            self.augmented_samples = int(augmentation_duration_s * 16000)
        else:
            self.augmented_samples = None

        assert (
            self.min_jitter_samples <= self.max_jitter_samples
        ), "Minimum jitter must be less than or equal to maximum jitter."

        #######################
        # Setup augmentations #
        #######################

       
        # Based on openWakeWord's augmentations, accessed on February 23, 2024.
        self.augment = augment

    
    def add_jitter(self, input_audio: np.ndarray):
        """Pads the clip on the right by a random duration between the class's min_jitter_s and max_jitter_s paramters.

        Args:
            input_audio (numpy.ndarray): Array containing the audio clip's samples.

        Returns:
            numpy.ndarray: Array of audio samples with silence added to the end.
        """
        if self.min_jitter_samples < self.max_jitter_samples:
            jitter_samples = np.random.randint(
                self.min_jitter_samples, self.max_jitter_samples
            )
        else:
            jitter_samples = self.min_jitter_samples

        # Pad audio on the right by jitter samples
        return np.pad(input_audio, (0, jitter_samples))

    def create_fixed_size_clip(self, input_audio: np.ndarray):
        """Ensures the input audio clip has a fixced length. If the duration is too long, the start of the clip is removed. If it is too short, the start of the clip is padded with silence.

        Args:
            input_audio (numpy.ndarray): Array containing the audio clip's samples.

        Returns:
            numpy.ndarray: Array of audio samples with `augmented_duration_s` length.
        """
        if self.augmented_samples is None:
            return input_audio

        if self.augmented_samples < input_audio.shape[0]:
            # Truncate the too long audio by removing the start of the clip
            if self.truncate_randomly:
                random_start = np.random.randint(
                    0, input_audio.shape[0] - self.augmented_samples
                )
                input_audio = input_audio[
                    random_start : random_start + self.augmented_samples
                ]
            else:
                input_audio = input_audio[-self.augmented_samples :]
        else:
            # Pad with zeros at start of too short audio clip
            left_padding_samples = self.augmented_samples - input_audio.shape[0]

            input_audio = np.pad(input_audio, (left_padding_samples, 0))

        return input_audio

    def augment_clip(self, input_audio: np.ndarray):
        """Augments the input audio after adding jitter and creating a fixed size clip.

        Args:
            input_audio (numpy.ndarray): Array containing the audio clip's samples.

        Returns:
            numpy.ndarray: The augmented audio of fixed duration.
        """
        input_audio = self.add_jitter(input_audio)
        input_audio = self.create_fixed_size_clip(input_audio)

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )  # Suppresses warning about background clip being too quiet... TODO: find better approach!
            output_audio = self.augment(input_audio, sample_rate=16000)

            # ADDED ON 4/07/2025
            # farm applied parameters
            applied_parameters = []
            for transform in self.augment.transforms:
                params = {}
                if isinstance(transform, audiomentations.OneOf)\
                    or isinstance(transform, audiomentations.SomeOf):
                    params = [(tr.__class__.__name__, tr.parameters) for tr in transform.transforms]                        
                elif 'parameters' in transform.__dir__():
                    params = transform.parameters

                applied_parameters.append((transform.__class__.__name__, params))
        
        return output_audio, applied_parameters

    def augment_generator(self, audio_generator):
        """A Python generator that augments clips retrived from the input audio generator.

        Args:
            audio_generator (generator): A Python generator that yields audio clips.

        Yields:
            numpy.ndarray: The augmented audio clip's samples.
        """
        # ADDED 04/07/2025
        # get and return clip path and repetition number
        for audio, path_data in audio_generator:
            yield self.augment_clip(audio), path_data
