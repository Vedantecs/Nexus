"""
NEXUS - voice/audio_capture.py
Listens to the microphone and cuts the sound into utterances (one spoken phrase each).
An utterance starts when the sound gets louder than the background noise and ends
after a short silence.
"""

import queue
import threading
from collections import deque

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
BLOCK_S = 0.05            # the microphone is processed in 50 ms blocks
MIN_THRESHOLD = 0.006     # never trigger below this loudness (RMS)
NOISE_FACTOR = 3.0        # speech must be this many times louder than the background noise
PRE_ROLL_S = 0.3          # keep this much sound from before the speech started
END_SILENCE_S = 0.8       # this much silence ends an utterance
MIN_SPEECH_S = 0.25       # shorter sounds (clicks, bumps) are ignored
MAX_UTTERANCE_S = 15.0    # force a cut after this long


class UtteranceListener:
    def __init__(self, device=None):
        self.device = device
        self.utterances = queue.Queue()      # finished utterances (float32 arrays, 16 kHz)
        self.level = 0.0                     # current loudness
        self.threshold = MIN_THRESHOLD
        self.speaking = False
        self.ready = threading.Event()       # set after the background noise is measured
        self._blocks = queue.Queue()
        self._stop = threading.Event()
        self._stream = None
        self._thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                      blocksize=int(SAMPLE_RATE * BLOCK_S), device=self.device,
                                      callback=self._callback)
        self._stream.start()
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def _callback(self, indata, frames, time_info, status):
        self._blocks.put(indata[:, 0].copy())

    def _next_block(self):
        while not self._stop.is_set():
            try:
                return self._blocks.get(timeout=0.2)
            except queue.Empty:
                continue
        return None

    def _worker(self):
        # 1) measure the background noise for about one second
        noise = []
        while len(noise) < int(1.0 / BLOCK_S):
            block = self._next_block()
            if block is None:
                return
            noise.append(float(np.sqrt(np.mean(block ** 2))))
        self.threshold = max(MIN_THRESHOLD, NOISE_FACTOR * float(np.median(noise)))
        self.ready.set()

        # 2) cut the sound into utterances
        pre_roll = deque(maxlen=int(PRE_ROLL_S / BLOCK_S))
        current, quiet = [], 0
        while not self._stop.is_set():
            block = self._next_block()
            if block is None:
                return
            level = float(np.sqrt(np.mean(block ** 2)))
            self.level = level
            if not self.speaking:
                pre_roll.append(block)
                if level > self.threshold:
                    self.speaking = True
                    current, quiet = list(pre_roll), 0
                continue
            current.append(block)
            quiet = quiet + 1 if level < 0.7 * self.threshold else 0
            too_long = len(current) * BLOCK_S > MAX_UTTERANCE_S
            if quiet * BLOCK_S >= END_SILENCE_S or too_long:
                audio = np.concatenate(current)
                speech_s = len(audio) / SAMPLE_RATE - quiet * BLOCK_S
                self.speaking = False
                current, quiet = [], 0
                pre_roll.clear()
                if speech_s >= MIN_SPEECH_S:
                    self.utterances.put(audio)