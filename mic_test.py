"""
NEXUS - voice/mic_test.py
Check the microphone and the Whisper speech-to-text model.

Run from the NEXUS folder (with venv active):
    python -m voice.mic_test
It records 5 seconds from the microphone, prints how loud it was,
then converts your speech to text.
"""

import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
SECONDS = 5
MODEL_SIZE = "base.en"     # "tiny.en" = faster, "base.en" = more accurate
MIC_DEVICE = None          # None = Windows default microphone; or put a device number from the list


def main():
    print("Input devices (microphones):")
    for i, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            print(f"  {i}: {device['name']}")
    if MIC_DEVICE is not None:
        sd.default.device = (MIC_DEVICE, None)
    print("Default input device number:", sd.default.device[0])

    print(f"\nLoading the Whisper model '{MODEL_SIZE}' (the first time it downloads about 150 MB)...")
    t0 = time.time()
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print(f"Model ready in {time.time() - t0:.1f} s")

    input(f"\nPress Enter, then speak a full sentence for {SECONDS} seconds... ")
    audio = sd.rec(int(SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    audio = audio[:, 0]
    print(f"Loudness (RMS): {np.sqrt(np.mean(audio ** 2)):.4f} | Peak: {np.max(np.abs(audio)):.3f}")

    t0 = time.time()
    segments, info = model.transcribe(audio, language="en", beam_size=1,
                                      condition_on_previous_text=False)
    text = " ".join(s.text.strip() for s in segments).strip()
    print(f"Heard (took {time.time() - t0:.1f} s): {text!r}")


if __name__ == "__main__":
    main()