from audiomentations import (
    Compose,
    OneOf,
    AddBackgroundNoise,
    ApplyImpulseResponse,
    PitchShift,
    TimeStretch,
    BandPassFilter,
    Gain,
    Shift,
    AddColorNoise,
    Normalize
)
from .custom_augmentations import AddCustomFunction

def aggressive():
    return Compose(
    [
        # change voice nature
        OneOf(
            [
                PitchShift(min_semitones=-1, max_semitones=2, p=1.0),
                TimeStretch(min_rate=0.9, max_rate=1.1, p=1.0),
            ],
            p=0.9,
        ),
        # simulate environment
        AddBackgroundNoise(
            sounds_path=[
                "_augmentation_data/fma_16k",
                "_augmentation_data/audioset_16k",
            ],
            min_snr_in_db=0,
            max_snr_in_db=15,
            p=0.9,
        ),
        ApplyImpulseResponse(ir_path=["_augmentation_data/mit_rirs"], p=0.7),
        AddColorNoise(
            p=0.5,
            min_snr_db=10,
            max_snr_db=30,
        ),
        # simulate devices
        BandPassFilter(  # simulate old radio/microphones
            min_center_freq=300,
            max_center_freq=4000,
            min_bandwidth_fraction=0.7,
            max_bandwidth_fraction=1.2,
            p=0.4,
        ),
        # gain, stereo shift and normalize
        Gain(min_gain_in_db=-3, max_gain_in_db=3, p=0.8),
        Shift(min_shift=-0.2, max_shift=0.2, p=0.5),
        Normalize(apply_to="only_too_loud_sounds", p=1.0),
    ],
    p=1.0,
)


def aggressive_no_noise():
    return Compose(
    [
        # change voice nature
        OneOf(
            [
                PitchShift(min_semitones=-1, max_semitones=2, p=1.0),
                TimeStretch(min_rate=0.9, max_rate=1.1, p=1.0),
                # TanhDistortion(min_distortion=tanh_distortion_range[0], max_distortion=tanh_distortion_range[1], p=1.0),
            ],
            p=0.9,
        ),
        # simulate devices
        BandPassFilter(  # simulate old radio/microphones
            min_center_freq=300,
            max_center_freq=4000,
            min_bandwidth_fraction=0.7,
            max_bandwidth_fraction=1.2,
            p=0.4,
        ),
        # gain, stereo shift and normalize
        Gain(min_gain_in_db=-3, max_gain_in_db=3, p=0.8),
        Shift(min_shift=-0.2, max_shift=0.2, p=0.5),
        Normalize(apply_to="only_too_loud_sounds", p=1.0),
    ],
    p=1.0,
)


def custom(function):
    return Compose(
    [
        AddCustomFunction(function=function, p=1),
        Normalize(apply_to="only_too_loud_sounds", p=1.0),
    ]
)

