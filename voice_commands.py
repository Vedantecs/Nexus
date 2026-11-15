"""
NEXUS - voice/voice_commands.py
Recognises fixed spoken commands in the text Whisper returns.
A command is accepted only when the whole utterance is (almost) the command phrase,
so a sentence that merely CONTAINS "stop typing" is typed like normal text.
"""

import re
from difflib import SequenceMatcher

COMMANDS = {
    "start typing": "START_TYPING",
    "stop typing": "STOP_TYPING",
    "new line": "NEW_LINE",
    "delete that": "DELETE_LAST",
    "stop": "STOP",                 # emergency stop
    "calibrate": "CALIBRATE",
    "click": "CLICK",
    "double click": "DOUBLE_CLICK",
    "right click": "RIGHT_CLICK",
    "address bar": "ADDRESS_BAR",   # presses Ctrl+L in a browser
    "volume up": "VOLUME_UP",
    "volume down": "VOLUME_DOWN",
    "mute": "MUTE",
    "undo": "UNDO",
    "redo": "REDO",
    "lock computer": "LOCK",
    "big screen": "HUD_BIG",
    "small screen": "HUD_SMALL",
}
MIN_MATCH = 0.85      # how close the spoken text must be to a command phrase (0 to 1)


def match_command(text):
    """Returns the command name (for example 'STOP_TYPING') or None."""
    spoken = " ".join(re.sub(r"[^a-z ]", "", text.lower()).split())
    if not spoken:
        return None
    best_name, best_score = None, 0.0
    for phrase, name in COMMANDS.items():
        score = SequenceMatcher(None, spoken, phrase).ratio()
        if score > best_score:
            best_name, best_score = name, score
    return best_name if best_score >= MIN_MATCH else None