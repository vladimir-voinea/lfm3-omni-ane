"""One command that proves the stack works: text, tools, image, audio, speech.

    python tools/smoke_test.py            # starts its own server on a free port

Every check prints PASS/FAIL with the evidence, and the exit code is the number of failures.
"""
import base64
import glob
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modes
from paths import PROJECT

PORT = int(os.environ.get('LFM3_SMOKE_PORT', '8099'))
BASE = 'http://127.0.0.1:%d' % PORT
FAILURES = []


def check(name, condition, evidence):
    print(('PASS  ' if condition else 'FAIL  ') + name + ' -- ' + str(evidence)[:110], flush=True)
    if not condition:
        FAILURES.append(name)


def post(path, payload, timeout=600, expect_error=False):
    data = json.dumps(payload).encode()
    request = urllib.request.Request(BASE + path, data=data,
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        if not expect_error:
            raise
        return json.loads(error.read())


def get(path, timeout=60):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as response:
        return json.loads(response.read())


def solid_png(red, green, blue, size=224):
    raw = b''.join(b'\x00' + bytes([red, green, blue]) * size for _ in range(size))
    def chunk(tag, body):
        return (struct.pack('>I', len(body)) + tag + body
                + struct.pack('>I', zlib.crc32(tag + body) & 0xffffffff))
    header = struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header)
            + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def resample_wav(source, target_rate=16000):
    """speech out writes 24 kHz; the audio encoder wants 16 kHz mono."""
    import wave
    with wave.open(source, 'rb') as handle:
        rate, channels, frames = handle.getframerate(), handle.getnchannels(), handle.readframes(handle.getnframes())
    samples = struct.unpack('<%dh' % (len(frames) // 2), frames)
    if channels > 1:
        samples = samples[::channels]
    ratio = rate / float(target_rate)
    count = int(len(samples) / ratio)
    out = []
    for index in range(count):
        position = index * ratio
        left = int(position)
        right = min(left + 1, len(samples) - 1)
        weight = position - left
        out.append(int(samples[left] * (1 - weight) + samples[right] * weight))
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(target_rate)
        handle.writeframes(struct.pack('<%dh' % len(out), *out))
    return buffer.getvalue()


def message_text(response):
    return (response['choices'][0]['message'].get('content') or '').strip()


def main():
    mode = os.environ.get('LFM3_SMOKE_MODE', modes.DEFAULT_MODE)
    environment = dict(os.environ, LFM3_PORT=str(PORT), LFM3_MODE=mode)
    log = open(os.path.join(PROJECT, 'smoke_server.log'), 'w')
    server = subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lfm3_serve.py')],
        stdout=log, stderr=subprocess.STDOUT, env=environment)
    try:
        for _ in range(180):
            try:
                get('/health', timeout=2)
                break
            except Exception:
                if server.poll() is not None:
                    print('server exited early', flush=True)
                    return 1
                time.sleep(2)
        health = get('/health')
        check('health', health.get('status') == 'ok', health)
        check('mode reported', health.get('mode') == mode and
              health.get('context_length') == modes.resolve(mode).context, health)
        check('mode exactness', health.get('exact') == modes.resolve(mode).exact,
              {'exact': health.get('exact')})
        oversize = post('/v1/chat/completions',
                        {'messages': [{'role': 'user', 'content': 'hi'}],
                         'max_tokens': modes.resolve(mode).context + 1}, expect_error=True)
        check('context guard', oversize.get('error') == 'context_exceeded', oversize)

        reply = post('/v1/chat/completions', {'messages': [{'role': 'user',
                     'content': 'What is the capital of Japan? Answer in one word.'}],
                     'max_tokens': 24})
        text = message_text(reply)
        check('text', 'tokyo' in text.lower(), text)

        tools = [{'type': 'function', 'function': {'name': 'get_weather',
                  'description': 'Current weather for a city',
                  'parameters': {'type': 'object',
                                 'properties': {'city': {'type': 'string'}},
                                 'required': ['city']}}}]
        call = post('/v1/chat/completions', {'messages': [{'role': 'user',
                   'content': 'What is the weather in Paris? Use the tool.'}],
                   'tools': tools, 'max_tokens': 96})
        calls = call['choices'][0]['message'].get('tool_calls') or []
        check('tool call', bool(calls) and calls[0]['function']['name'] == 'get_weather',
              calls[0]['function'] if calls else 'no tool_calls')

        if calls:
            follow = post('/v1/chat/completions', {'messages': [
                {'role': 'user', 'content': 'What is the weather in Paris? Use the tool.'},
                {'role': 'assistant', 'content': None, 'tool_calls': calls},
                {'role': 'tool', 'tool_call_id': calls[0].get('id', 'call_1'),
                 'content': '18 degrees and cloudy'}], 'max_tokens': 64})
            answer = message_text(follow)
            check('tool result used', '18' in answer, answer)

        image = base64.b64encode(solid_png(220, 30, 30)).decode()
        seen = post('/v1/chat/completions', {'messages': [{'role': 'user', 'content': [
               {'type': 'text', 'text': 'What colour is this image? One word.'},
               {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + image}}]}],
               'max_tokens': 24})
        colour = message_text(seen)
        check('image in', 'red' in colour.lower(), colour)

        wav = os.path.join(PROJECT, 'restored.wav')
        if os.path.exists(wav):
            audio = base64.b64encode(resample_wav(wav)).decode()
            heard = post('/v1/chat/completions', {'messages': [{'role': 'user', 'content': [
                    {'type': 'text', 'text': 'Transcribe this audio.'},
                    {'type': 'input_audio', 'input_audio': {'data': audio, 'format': 'wav'}}]}],
                    'max_tokens': 48})
            transcript = message_text(heard)
            check('audio in', len(transcript) > 5, transcript)
        else:
            print('SKIP  audio in -- no restored.wav to feed back', flush=True)

        request = urllib.request.Request(
            BASE + '/v1/audio/speech', data=json.dumps({'input': 'Hello there.'}).encode(),
            headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                body = response.read()
                is_wav = body[:4] == b'RIFF' and body[8:12] == b'WAVE'
                check('speech out', is_wav and len(body) > 8000,
                      '%d bytes, %s' % (len(body), 'WAV' if is_wav else 'not WAV'))
        except urllib.error.HTTPError as error:
            body = error.read()
            detail = body[:120].decode('utf-8', 'replace')
            check('speech out', False, 'HTTP %d %s' % (error.code, detail))
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
        # a terminated server never runs its atexit hook, so its compiled programs stay behind
        for compiled in glob.glob(os.path.join(tempfile.gettempdir(), '*.mlmodelc')):
            shutil.rmtree(compiled, ignore_errors=True)
    print({'failures': FAILURES}, flush=True)
    return len(FAILURES)


if __name__ == '__main__':
    sys.exit(main())