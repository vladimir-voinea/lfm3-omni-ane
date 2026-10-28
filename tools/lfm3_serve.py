# OpenAI-compatible HTTP server for LFM3 on the engine.  Same shape as the Dory one:
# chunked SSE that closes the connection after the terminating chunk, a usage-only
# final chunk when asked, and a health endpoint that reports live decode speed.
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
sys.path.insert(0, '/Users/vladimir/dev/dory-ane-worktrees/DANE-5')

from anemll.runtime.dory_sampling import SamplingParams  # noqa: E402

from lfm3_engine import Engine  # noqa: E402

MODEL_ID = 'lfm3-omni-2b-ane'
ENGINE = None
ENGINE_LOCK = threading.Lock()
SPEECH_LOCK = threading.Lock()


def say(text, max_frames=48):
    """Speech runs in its own process.  Loading the depth decoder and codec beside the 26
    resident programs deadlocked the server, and a child process cannot leak that into the
    serving path."""
    import subprocess
    import tempfile

    handle = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    path = handle.name
    handle.close()
    result = subprocess.run(
        [sys.executable, '/Users/vladimir/dev/lfm3-ane/tools/say.py',
         text, path, str(max_frames)],
        capture_output=True, timeout=1800)
    if result.returncode:
        raise RuntimeError(result.stderr.decode()[-300:])
    with open(path, 'rb') as reader:
        return reader.read()


def engine() -> Engine:
    global ENGINE
    with ENGINE_LOCK:
        if ENGINE is None:
            ENGINE = Engine()
    return ENGINE


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        sys.stderr.write('%s %s\n' % (self.address_string(), fmt % args))

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.end_headers()

    def do_GET(self):
        if self.path.startswith('/health'):
            eng = engine()
            return self._json({'status': 'ok', 'model': MODEL_ID,
                               'sessions': len(eng.sessions),
                               'context_length': 28672})
        if self.path.startswith('/v1/models'):
            return self._json({'object': 'list', 'data': [{'id': MODEL_ID,
                                                           'object': 'model',
                                                           'owned_by': 'local'}]})
        return self._json({'error': 'not found'}, 404)

    def _chunk(self, text: str):
        data = text.encode()
        self.wfile.write(('%x\r\n' % len(data)).encode() + data + b'\r\n')
        self.wfile.flush()

    def _speech(self, request):
        """OpenAI-style text-to-speech: returns a 24 kHz mono WAV."""
        import wave
        from io import BytesIO

        import numpy as np

        text = (request.get('input') or '').strip()
        if not text:
            return self._json({'error': 'input is required'}, 400)
        speed_frames = int(request.get('max_frames') or 48)
        with SPEECH_LOCK:
            try:
                body = say(text, speed_frames)
            except RuntimeError as error:
                # the engine already holds all 26 programs; a child process cannot load a
                # second copy on this machine (make_state fails), so speech stays a CLI path
                return self._json({
                    'error': 'speech_unavailable_in_server',
                    'detail': str(error)[-200:],
                    'use': "python tools/say.py 'your text' out.wav  (24 kHz WAV)"}, 501)
        self.send_response(200)
        self.send_header('Content-Type', 'audio/wav')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        request = json.loads(self.rfile.read(length) or b'{}')
        if self.path.startswith('/v1/audio/speech'):
            return self._speech(request)
        if not self.path.startswith('/v1/chat/completions'):
            return self._json({'error': 'not found'}, 404)

        messages = request.get('messages') or []
        stream = bool(request.get('stream'))
        max_tokens = int(request.get('max_tokens') or 256)
        params = SamplingParams(temperature=float(request.get('temperature', 0.0) or 0.0),
                                top_p=float(request.get('top_p', 1.0) or 1.0),
                                repetition_penalty=float(request.get('repetition_penalty', 1.0) or 1.0))

        if not stream:
            result = engine().generate(messages, params, max_tokens=max_tokens)
            return self._json({
                'id': 'chatcmpl-lfm3', 'object': 'chat.completion', 'model': MODEL_ID,
                'choices': [{'index': 0, 'finish_reason': 'stop',
                             'message': {'role': 'assistant', 'content': result['text']}}],
                'usage': {'prompt_tokens': result['prompt_tokens'],
                          'completion_tokens': len(result['tokens']),
                          'total_tokens': result['prompt_tokens'] + len(result['tokens'])}})

        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Transfer-Encoding', 'chunked')
        self.send_header('Connection', 'close')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.close_connection = True
        try:
            # token by token, so the first chunk arrives with the first token instead of
            # after the whole answer
            produced = []
            meta = {}
            chunk_delay = 0.0
            for token, info in engine().stream(messages, params, max_tokens=max_tokens):
                meta.update(info)
                produced.append(token)
                piece = engine().tokenizer.decode([token])
                if not piece:
                    continue
                chunk_delay = chunk_delay or 0.0
                payload = {'id': 'chatcmpl-lfm3', 'object': 'chat.completion.chunk',
                           'model': MODEL_ID,
                           'choices': [{'index': 0, 'delta': {'content': piece},
                                        'finish_reason': None}]}
                self._chunk('data: ' + json.dumps(payload) + '\n\n')
            prompt_tokens = len(engine().tokenizer.encode(
                __import__('lfm3_engine').render_chat(messages)).ids) + 1
            final = {'id': 'chatcmpl-lfm3', 'object': 'chat.completion.chunk',
                     'model': MODEL_ID,
                     'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
                     'usage': {'prompt_tokens': prompt_tokens,
                               'completion_tokens': len(produced),
                               'total_tokens': prompt_tokens + len(produced)}}
            self._chunk('data: ' + json.dumps(final) + '\n\n')
            self._chunk('data: [DONE]\n\n')
            self.wfile.write(b'0\r\n\r\n')
            self.wfile.flush()
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write('stream failed: %s\n' % exc)


if __name__ == '__main__':
    port = int(os.environ.get('LFM3_PORT', '8010'))
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print('LFM3 ANE server on http://127.0.0.1:%d' % port, flush=True)
    engine()
    print('model loaded', flush=True)
    server.serve_forever()