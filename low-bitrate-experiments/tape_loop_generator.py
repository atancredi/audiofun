import numpy as np

import sys
sys.path.append("../")

from audiofun import AudioFun, noise

if __name__ == "__main__":
    
    # get audio
    sample_rate = 44100
    sample_rate, audio  = noise.get_noise(10, sample_rate=sample_rate)

    # get impulse response
    # 
    # BUONO per 'transienti' periodici
    # sperimentare con altri transienti, padding randomico ad ogni tiling? o con crescita esponenziale randomica..
    #
    ir_rate, ir = noise.get_impulse_response("_augmentation_data/mit_rirs/h006_Bedroom_42txts.wav", len(audio), ir_right_padding_ms=2500)
    

    af = AudioFun(audio, sample_rate)\
        .apply_convolution(ir)\
        .wow_flutter(speed=0.1)\
        .bandpass_filter()\
        .normalize_to_peak_db(-24)\
        .save_audio("tape_noise.wav", clip=True)

