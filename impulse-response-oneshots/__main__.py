from pathlib import Path
import numpy as np
from scipy.io.wavfile import write
import audiomentations
import datasets

clips_directory = Path("./_test_data")
file_pattern = "*.wav"
paths_to_clips = [str(i) for i in Path(clips_directory).glob(file_pattern)]
audio_dataset = datasets.Dataset.from_dict(
    {"audio": [str(i) for i in paths_to_clips]}
).cast_column("audio", datasets.Audio())

# Convert all clips to 16 kHz sampling rate when accessed
audio_dataset = audio_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))

el = list(audio_dataset.take(1))[0]
print(el)

augment = audiomentations.Compose(
    transforms=[
        audiomentations.ApplyImpulseResponse(
            p=1,
            ir_path=Path("./_augmentation_data") / "mit_rirs",
        ),
        audiomentations.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        audiomentations.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    ]
)


rate = 16000
audio = el["audio"]["array"]
output_audio = augment(audio, sample_rate=rate)

scaled = np.int16(output_audio / np.max(np.abs(output_audio)) * 32767)
write("_test_results/test.wav", rate, scaled)
