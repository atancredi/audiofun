import numpy as np
from fx import get_audio_channel, get_impulse_response, apply_convolution, apply_batch_convolution, wow_flutter, bandpass_filter, save_audio, normalize

if __name__ == "__main__":
    sample_rate = 44100
    # sample_rate, audio  = get_noise(10)
    sample_rate, audio = get_audio_channel("ilerep_83_main_snaf_trimmed.wav")
    # audio = normalize(audio, 1)
    # print(min(list(audio)), max(list(audio)))
    
    # 
    # BUONO per 'transienti' periodici
    # sperimentare con altri transienti, padding randomico ad ogni tiling? o con crescita esponenziale randomica..
    #
    # il problema e' che se la finestra e' troppo grande fa effetto delay 
    ir_rate, ir = get_impulse_response("_augmentation_data/mit_rirs/h006_Bedroom_42txts.wav", len(audio), ir_right_padding_ms=2500)
    # audio = apply_convolution(audio, ir)
    audio = apply_batch_convolution(audio, ir, sample_rate, 2000)
    print(min(list(audio)), max(list(audio)))

    # audio = downsample(audio, 2)
    # audio = bitcrush(audio, 4)
    # audio = wow_flutter(audio, sample_rate, speed=0.1)

    # audio = saturate(audio, amount=1.1)
    # audio = bandpass_filter(audio, sample_rate)
    # audio = make_loop(audio, len(audio), sample_rate)

    audio = normalize(audio)
    print(min(list(audio)), max(list(audio)))
    save_audio("ilerep_83_main_snaf_trimmed_fx.wav", sample_rate, audio)

