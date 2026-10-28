import base64
import json
import urllib.request

data = base64.b64encode(open('/tmp/speech.wav', 'rb').read()).decode()
payload = {'model': 'lfm3',
           'messages': [{'role': 'user', 'content': [
               {'type': 'text', 'text': 'Transcribe the audio.'},
               {'type': 'input_audio', 'input_audio': {'data': data, 'format': 'wav'}}]}],
           'max_tokens': 24}
request = urllib.request.Request('http://127.0.0.1:8010/v1/chat/completions',
                                 data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(request, timeout=900) as response:
    body = json.loads(response.read())
print({'answer': body['choices'][0]['message']['content']}, flush=True)