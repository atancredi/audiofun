import numpy as np
from numpy.typing import NDArray
from typing import Callable

from audiomentations.core.audio_loading_utils import load_sound_file
from audiomentations.core.transforms_interface import BaseWaveformTransform


class AddCustomFunction(BaseWaveformTransform):
    """
    """

    def __init__(
        self,
        function: Callable[[NDArray[np.float32]], NDArray[np.float32]],
        p: float = 0.5,
    ):
        """
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.function = function

    @staticmethod
    def _load_sound(file_path, sample_rate):
        return load_sound_file(file_path, sample_rate)

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            pass

    def apply(self, samples: NDArray[np.float32], sample_rate: int):
        samples = self.function(samples, sample_rate)
        return samples

    def __getstate__(self):
        state = self.__dict__.copy()
        return state
