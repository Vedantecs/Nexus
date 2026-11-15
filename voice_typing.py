"""
NEXUS - voice/voice_typing.py
Voice typing with two modes:
    CONTROL mode (default): everything you say is ignored, except "start typing".
    TYPING mode: everything you say is typed where the cursor is.
Commands:
    "start typing"  -> TYPING mode
    "stop typing"   -> back to CONTROL mode
    "new line"      -> presses Enter                          (TYPING mode)
    "delete that"   -> deletes the last phrase you typed      (TYPING mode)
SAFETY: NEXUS refuses to type into VS Code or a terminal. HOLD ESC to quit.

Run from the NEXUS folder (with venv active):
    python -m voice.voice_typing
"""

import ctypes
import queue
import time
import winsound

from actions.keyboard_actions import VK_RETURN, backspace, press_key, type_text, typing_allowed
from voice.audio_capture import UtteranceListener
from voice.speech_to_text import SpeechToText
from voice.voice_commands import match_command

MIN_CONFIDENCE = 0.35     # below this, the speech is ignored as "not sure"
VK_ESCAPE = 0x1B


def esc_held():
    return bool(ctypes.windll.user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000)


def beep(high):
    winsound.Beep(1000 if high else 600, 120)


def main():
    print("Loading the speech model...")
    stt = SpeechToText()
    listener = UtteranceListener()
    listener.start()
    print("Measuring the background noise - please stay quiet for 1 second...")
    listener.ready.wait()
    print(f"Ready. Loudness threshold: {listener.threshold:.4f}")
    print('Say "start typing". Hold ESC to quit.')

    mode = "CONTROL"
    typed = []          # length of each typed phrase, so "delete that" can remove the last one

    try:
        while not esc_held():
            try:
                audio = listener.utterances.get(timeout=0.2)
            except queue.Empty:
                continue

            t0 = time.time()
            text, confidence = stt.transcribe(audio)
            print(f"[{mode}] heard (confidence {confidence:.2f}, {time.time() - t0:.1f} s): {text!r}")
            if len(text) < 2 or confidence < MIN_CONFIDENCE:
                print("   -> ignored (not sure what was said)")
                continue
            command = match_command(text)

            if mode == "CONTROL":
                if command == "START_TYPING":
                    mode = "TYPING"
                    beep(True)
                    print("   -> TYPING MODE. Click your text box, then speak...")
                else:
                    print('   -> ignored (say "start typing" first)')
                continue

            # ---- TYPING mode ----
            if command == "STOP_TYPING":
                mode = "CONTROL"
                typed.clear()
                beep(False)
                print("   -> back to CONTROL MODE")
                continue

            allowed, title = typing_allowed()
            if not allowed:
                print(f"   -> BLOCKED: the window in front is '{title}'. Click your text box first.")
                continue

            if command == "NEW_LINE":
                press_key(VK_RETURN)
                typed.append(1)
                print("   -> new line")
            elif command == "DELETE_LAST":
                if typed:
                    backspace(typed.pop())
                print("   -> deleted the last phrase")
            else:
                phrase = text + " "
                type_text(phrase)
                typed.append(len(phrase))
                print(f"   -> typed into '{title}'")
    except KeyboardInterrupt:
        pass

    listener.stop()
    print("Stopped.")


if __name__ == "__main__":
    main()