import numpy as np

import sys
sys.path.append("../")
from audiofun import AudioFun, noise

if __name__ == "__main__":

    sr, audio = AudioFun.read_file("ilerep_83_main_snaf_trimmed.wav")
    

    # 
    # BUONO per 'transienti' periodici
    # sperimentare con altri transienti, padding randomico ad ogni tiling? o con crescita esponenziale randomica..
    #
    # il problema e' che se la finestra e' troppo grande fa effetto delay 
    ir_rate, ir = noise.get_impulse_response("_augmentation_data/mit_rirs/h006_Bedroom_42txts.wav", len(audio), ir_right_padding_ms=2500)

    af = AudioFun(audio, sr)\
        .apply_batch_convolution(ir, 2000)\
            .downsample_raw(2)\
            .bitcrush(4)\
            .wow_flutter(speed=0.1)\
            .saturate(amount=1.1)\
            .bandpass_filter()\
            .normalize_to_peak_db(-3)\
            .save_audio("ilerep_83_main_snaf_trimmed_fx.wav")

