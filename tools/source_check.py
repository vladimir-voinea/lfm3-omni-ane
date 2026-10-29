# The vision, audio and speech paths read tower weights that are not part of the converted
# build. Fail with the fix rather than a FileNotFoundError inside some tensor load.
#
# towers.safetensors is the 0.88 GB slice of the checkpoint those three paths actually read
# (the 4.3 GB text backbone is already converted and is not needed at run time).
import os

from paths import SOURCE, TOWERS

DOWNLOAD = ("huggingface-cli download LiquidAI/LFM3-Omni-2B --local-dir " + SOURCE + "\n"
            "  python tools/extract_towers.py   # keeps towers.safetensors, drops the rest")


def require_source(what):
    """Raise a message that says what is missing and how to get it."""
    needed = [TOWERS]
    if what == 'speech':
        needed.append(os.path.join(SOURCE, 'audio_tokenizer', 'model.safetensors'))
    missing = [path for path in needed if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            '%s needs tower weights, which are not part of the converted build.\n'
            'Missing: %s\nGet them with:\n  %s'
            % (what, ', '.join(os.path.relpath(m, SOURCE) for m in missing), DOWNLOAD))
    return SOURCE