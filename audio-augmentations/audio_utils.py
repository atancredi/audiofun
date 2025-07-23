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

import numpy as np
import webrtcvad


def remove_silence_webrtc(
    audio_data: np.ndarray,
    frame_duration: float = 0.030,
    sample_rate: int = 16000,
    min_start: int = 2000,
) -> np.ndarray:
    """Uses webrtc voice activity detection to remove silence from the clips

    Args:
        audio_data (numpy.ndarray): The input clip's audio samples.
        frame_duration (float): The frame_duration for webrtcvad. Defaults to 0.03.
        sample_rate (int): The audio's sample rate. Defaults to 16000.
        min_start: (int): The number of audio samples from the start of the clip to always include. Defaults to 2000.

    Returns:
        numpy.ndarray: Array with the trimmed audio clip's samples.
    """
    vad = webrtcvad.Vad(0)

    # webrtcvad expects int16 arrays as input, so convert if audio_data is a float
    float_type = audio_data.dtype in (np.float32, np.float64)
    if float_type:
        audio_data = (audio_data * 32767).astype(np.int16)

    filtered_audio = audio_data[0:min_start].tolist()

    step_size = int(sample_rate * frame_duration)

    for i in range(min_start, audio_data.shape[0] - step_size, step_size):
        vad_detected = vad.is_speech(
            audio_data[i : i + step_size].tobytes(), sample_rate
        )
        if vad_detected:
            # If voice activity is detected, add it to filtered_audio
            filtered_audio.extend(audio_data[i : i + step_size].tolist())

    # If the original audio data was a float array, convert back
    if float_type:
        trimmed_audio = np.array(filtered_audio)
        return np.array(trimmed_audio / 32767).astype(np.float32)

    return np.array(filtered_audio).astype(np.int16)
