from augmentation import Augmentation
from general_augmentation import GeneralAugmentation

from .composed_effects import aggressive, aggressive_no_noise, custom


def default_augmenter(
        background_paths = [
            "_augmentation_data/fma_16k",
            "_augmentation_data/audioset_16k",
        ],
        impulse_paths=["_augmentation_data/mit_rirs"]
):
    return Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.1,
            "TanhDistortion": 0.1,
            "PitchShift": 0.1,
            "BandStopFilter": 0.1,
            "AddColorNoise": 0.1,
            "AddBackgroundNoise": 0.75,
            "Gain": 1.0,
            "RIR": 0.5,
        },
        impulse_paths=impulse_paths,
        background_paths=background_paths,
        background_min_snr_db=-5,
        background_max_snr_db=10,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )


def aggressive_augmenter():
    augmenter = aggressive()
    return GeneralAugmentation(
        augment=augmenter,
        augmentation_duration_s=3.2,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )


def aggressive_no_noise_augmenter():
    augmenter = aggressive_no_noise()
    return GeneralAugmentation(
        augment=augmenter,
        augmentation_duration_s=3.2,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )


def custom_augmenter_test(function):
    augmenter = custom(function)
    return GeneralAugmentation(
        augment=augmenter,
        augmentation_duration_s=3.2,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )
