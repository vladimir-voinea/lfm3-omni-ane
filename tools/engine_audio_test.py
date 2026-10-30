import base64
import sys

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
sys.path.insert(0, '/Users/vladimir/dev/dory-ane-worktrees/DANE-5')

from anemll.runtime.dory_sampling import SamplingParams
from lfm3_engine import Engine, render_chat
from lfm3_audio_api import rows_from_base64

payload = base64.b64encode(open('/tmp/speech.wav', 'rb').read()).decode()
rows = rows_from_base64(payload)
print({'audio_rows': len(rows)}, flush=True)
messages = [{'role': 'user',
             'content': '<|audio_start|><|audio|><|audio_end|>Transcribe the audio.'}]
print({'rendered': repr(render_chat(messages)[:70])}, flush=True)
engine = Engine()
text = ''.join(engine.tokenizer.decode([t]) for t, _ in
               engine.stream(messages, SamplingParams(), max_tokens=24, audio_rows=rows))
print({'answer': text}, flush=True)