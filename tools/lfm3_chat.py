# Render prompts with the model's own chat template (tools, images, audio, thinking)
# instead of a hand-written approximation.
import os

from paths import MODEL
import re

import jinja2

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get('LFM3_MODEL', MODEL)
TEMPLATE = os.path.join(MODEL_DIR, 'chat_template.jinja')

_ENV = None


def _env():
    global _ENV
    if _ENV is None:
        text = open(TEMPLATE).read()
        # `{% generation %}` only marks the assistant span for transformers' masking; it is
        # not valid jinja2 and carries no text, so drop it
        text = re.sub(r'\{%-?\s*(end)?generation\s*-?%\}', '', text)
        environment = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)
        environment.filters['tojson'] = lambda v: __import__('json').dumps(v, ensure_ascii=False)
        _ENV = environment.from_string(text)
    return _ENV


def render(messages, tools=None, enable_thinking=False, add_generation_prompt=True,
           bos='<|startoftext|>'):
    kwargs = {'messages': messages, 'tools': tools, 'enable_thinking': enable_thinking,
              'add_generation_prompt': add_generation_prompt, 'bos_token': bos}
    try:
        return _env().render(**kwargs)
    except Exception:
        if tools is not None:
            kwargs['tools'] = None
            return _env().render(**kwargs)
        raise