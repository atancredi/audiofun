from fx import get_noise, get_impulse_response, apply_convolution, wow_flutter, bandpass_filter, save_audio, normalize_to_peak_db
import numpy as np

if __name__ == "__main__":
    sample_rate = 44100
    sample_rate, audio  = get_noise(10)

    # 
    # BUONO per 'transienti' periodici
    # sperimentare con altri transienti, padding randomico ad ogni tiling? o con crescita esponenziale randomica..
    #
    ir_rate, ir = get_impulse_response("_augmentation_data/mit_rirs/h006_Bedroom_42txts.wav", len(audio), ir_right_padding_ms=2500)
    audio = apply_convolution(audio, ir)

    audio = wow_flutter(audio, sample_rate, speed=0.1)

    audio = bandpass_filter(audio, sample_rate)

    audio = normalize_to_peak_db(audio, -24)

    # Clip , convert to int16, sve
    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)
    save_audio("tape_noise.wav", sample_rate, audio)

