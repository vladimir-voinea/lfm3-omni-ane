# The vision, audio and speech paths read the original safetensors, which are not part of
# the converted build. Fail with the fix rather than a FileNotFoundError in some tensor load.
import os

from paths import SOURCE

SOURCE = SOURCE
DOWNLOAD = ("huggingface-cli download LiquidAI/LFM3-Omni-2B --local-dir " + SOURCE)


def require_source(what):
    """Raise a message that says what is missing and how to get it."""
    needed = [os.path.join(SOURCE, 'model.safetensors')]
    if what == 'speech':
        needed.append(os.path.join(SOURCE, 'audio_tokenizer', 'model.safetensors'))
    missing = [path for path in needed if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            '%s needs the original weights, which are not part of the converted build.\n'
            'Missing: %s\nGet them with:\n  %s'
            % (what, ', '.join(os.path.relpath(m, SOURCE) for m in missing), DOWNLOAD))
    return SOURCE