"""
NEXUS - voice/speech_to_text.py
Converts speech to text with Whisper (a pretrained model that runs on your CPU).
"""

import numpy as np
from faster_whisper import WhisperModel

MODEL_SIZE = "base.en"      # "tiny.en" = faster, "base.en" = more accurate


class SpeechToText:
    def __init__(self, model_size=MODEL_SIZE):
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio):
        """audio: float32 mono 16 kHz. Returns (text, confidence from 0 to 1)."""
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak < 1e-4:
            return "", 0.0
        audio = audio * min(0.7 / peak, 8.0)             # bring quiet speech up to a normal level
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=1,
            condition_on_previous_text=False, temperature=0.0)
        segments = list(segments)
        if not segments:
            return "", 0.0
        text = " ".join(s.text.strip() for s in segments).strip()
        avg_logprob = float(np.mean([s.avg_logprob for s in segments]))
        no_speech = float(max(s.no_speech_prob for s in segments))
        return text, float(np.exp(avg_logprob) * (1.0 - no_speech))