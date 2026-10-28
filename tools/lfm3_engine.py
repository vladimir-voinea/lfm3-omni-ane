# Incremental generation: yield as the tokens are produced, with the prompt/past
# tokens already in the session, so the server can stream token by token.
from __future__ import annotations

import json
import os

from paths import MODEL, PROJECT
import sys
import threading
import time

sys.path.insert(0, PROJECT)

from sampling import Sampler, SamplingParams  # noqa: E402

from lfm3_runtime import Runtime  # noqa: E402

MODEL_DIR = os.environ.get('LFM3_MODEL', MODEL)
ANE_DIR = os.environ.get('LFM3_ANE', MODEL)
IMAGE_TOKEN = 125030
AUDIO_TOKEN = 125031
IM_END = 124900


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))


def render_chat(messages, tools=None, enable_thinking=False):
    """Use the model's own chat template when it is available: it is the only thing that
    knows how tools, images and thinking markers must be spelled."""
    try:
        from lfm3_chat import render
        return render(messages, tools=tools, enable_thinking=enable_thinking,
                      add_generation_prompt=True)
    except Exception:
        parts = [('<|im_start|>%s\n%s<|im_end|>\n' % (m['role'], m['content'])) for m in messages]
        parts.append('<|im_start|>assistant\n')
        return ''.join(parts)

class Tracked:
    def __init__(self, session):
        self.session = session
        self.created = time.time()

    @property
    def tokens(self):
        return self.session.tokens

    def can_extend(self, prompt_tokens) -> bool:
        return (len(self.tokens) > 0 and len(prompt_tokens) >= len(self.tokens)
                and list(prompt_tokens[:len(self.tokens)]) == self.tokens)


class Engine:
    def __init__(self, ane_dir: str = ANE_DIR, max_sessions: int = 2):
        self.lock = threading.Lock()
        self.tokenizer = load_tokenizer()
        self.runtime = Runtime(ane_dir)
        self.max_sessions = max_sessions
        self.sessions = []

    def _session_for(self, prompt_tokens):
        for tracked in self.sessions:
            if tracked.can_extend(prompt_tokens):
                return tracked, prompt_tokens[len(tracked.tokens):], True
        if len(self.sessions) >= self.max_sessions:
            self.sessions.pop(0)
        tracked = Tracked(self.runtime.new_session())
        self.sessions.append(tracked)
        return tracked, prompt_tokens, False

    def _decode(self, tracked, logits, params, max_tokens, first, reused, pending_len, prompt_len):
        sampler = Sampler(params)
        produced = []
        decode_start = time.perf_counter()
        while len(produced) < max_tokens:
            token = sampler.next_token(logits, tracked.session.tokens)
            if token == IM_END:
                break
            produced.append(token)
            tracked.session.tokens.append(token)
            yield token, {'first_token_seconds': first, 'prefix_reused': reused,
                          'prefill_tokens': pending_len,
                          'prompt_tokens': prompt_len,
                          'decode_start': decode_start}
            logits = self.runtime.step(tracked.session, [token])

    def _stream_spliced(self, tracked, rows, placeholder, params, max_tokens, prompt_tokens):
        """Feed vectors into the residual stream where a placeholder token sits."""
        session = tracked.session
        at = prompt_tokens.index(placeholder)
        self.runtime.step(session, prompt_tokens[:at])
        logits = None
        for row in rows:
            logits = self.runtime.step_hidden(session, row)
        session.tokens.extend(prompt_tokens[:at + 1])
        started = time.perf_counter()
        pending = prompt_tokens[at + 1:]
        if pending:
            logits = self.runtime.step(session, pending)
        session.tokens.extend(pending)
        first = time.perf_counter() - started
        for token, info in self._decode(tracked, logits, params, max_tokens, first, False,
                                        len(pending), len(prompt_tokens)):
            yield token, info

    def stream(self, messages, params: SamplingParams, max_tokens: int = 256,
               image_rows=None, audio_rows=None, tools=None, enable_thinking=False):
        prompt = render_chat(messages, tools=tools, enable_thinking=enable_thinking)
        prompt_tokens = [1] + self.tokenizer.encode(prompt).ids
        with self.lock:
            if image_rows or audio_rows:
                # media cannot be reused by token prefix, so it always starts a session and
                # its vectors go into the residual stream where the placeholder sits
                tracked = Tracked(self.runtime.new_session())
                rows, placeholder = ((image_rows, IMAGE_TOKEN) if image_rows
                                     else (audio_rows, AUDIO_TOKEN))
                for token, info in self._stream_spliced(tracked, rows, placeholder, params,
                                                        max_tokens, prompt_tokens):
                    yield token, info
                return
            tracked, pending, reused = self._session_for(prompt_tokens)
            started = time.perf_counter()
            logits = self.runtime.step(tracked.session, pending) if pending else None
            tracked.session.tokens.extend(pending)
            first = time.perf_counter() - started
            for token, info in self._decode(tracked, logits, params, max_tokens, first, reused,
                                            len(pending), len(prompt_tokens)):
                yield token, info

    def generate(self, messages, params: SamplingParams, max_tokens: int = 256,
                 image_rows=None, audio_rows=None, tools=None, enable_thinking=False):
        produced = []
        meta = {}
        started = time.perf_counter()
        for token, info in self.stream(messages, params, max_tokens, image_rows=image_rows,
                                       audio_rows=audio_rows, tools=tools,
                                       enable_thinking=enable_thinking):
            produced.append(token)
            meta.update(info)
        elapsed = time.perf_counter() - started
        decode_time = elapsed - sum([0.0]) - meta.get('first_token_seconds', 0.0)
        rate = len(produced) / decode_time if decode_time > 0 else 0.0
        return {'text': self.tokenizer.decode(produced), 'tokens': produced,
                'prompt_tokens': meta.get('prompt_tokens', 0),
                'first_token_seconds': meta.get('first_token_seconds', 0.0),
                'decode_tokens_per_second': rate,
                'prefix_reused': meta.get('prefix_reused', False),
                'prefill_tokens': meta.get('prefill_tokens', 0)}


if __name__ == '__main__':
    engine = Engine()
    cold = engine.generate([{'role': 'user', 'content': 'What is the capital of Japan?'}],
                           SamplingParams(temperature=0.0), max_tokens=24)
    print({'cold': cold['text'], 'first_token_s': round(cold['first_token_seconds'], 3),
           'reused': cold['prefix_reused'], 'tps': round(cold['decode_tokens_per_second'], 1)},
          flush=True)
    messages = [{'role': 'user', 'content': 'What is the capital of Japan?'},
                {'role': 'assistant', 'content': cold['text']},
                {'role': 'user', 'content': 'And of Italy?'}]
    warm = engine.generate(messages, SamplingParams(temperature=0.0), max_tokens=24)
    print({'warm': warm['text'], 'first_token_s': round(warm['first_token_seconds'], 3),
           'reused': warm['prefix_reused'], 'prefill_tokens': warm['prefill_tokens'],
           'tps': round(warm['decode_tokens_per_second'], 1)}, flush=True)