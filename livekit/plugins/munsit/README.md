# Munsit plugin for LiveKit Agents

Arabic [TTS](https://docs.munsit.com/text-to-speech/get-started) and
[STT](https://docs.munsit.com/speech-to-text/transcribe) over the Munsit / Faseeh API, in the same
shape as the other LiveKit plugins. Depends only on `livekit-agents`.

# TTS

## Usage

```python
from livekit.plugins import munsit

tts = munsit.TTS(
    voice_id="ar-najdi-male-2",   # from GET /voices
    model="faseeh-v1-preview",    # from GET /models
    stability=0.5,                # 0.0-1.0
    speed=1.0,                    # 0.7-1.2
    sample_rate=24000,            # 8000-48000; 48000 is engine-native
    dialect="auto",               # auto | emirati | fusha
    streaming=True,
    api_key=...,                  # or MUNSIT_API_KEY
    base_url=...,                 # or MUNSIT_BASE_URL; UAE: https://ae.api.faseeh.ai/api/v1
)
```

Through the agent config (`get_tts`):

```json
{
  "type": "MUNSIT",
  "voice": "ar-najdi-male-2",
  "model": "faseeh-v1-preview",
  "api_key": "...",
  "stability": 0.5,
  "speed": 1.0,
  "sample_rate": 24000,
  "dialect": "auto",
  "streaming": true,
  "sentence_tokenizer": { "type": "BLINGFIRE", "min_sentence_len": 1 }
}
```

## The two modes

Munsit has one endpoint — `POST /text-to-speech/{model_id}` — and a `streaming` flag on the body.

| | capability | request | response |
|---|---|---|---|
| `streaming=True` (default) | streaming + aligned transcript: the session drives `stream()` | one request per sentence | chunked raw PCM16 mono |
| `streaming=False` | the session wraps `synthesize()` in `tts.StreamAdapter` | one request per sentence | complete `audio/wav` |

Munsit takes no incremental text, so `SynthesizeStream` works like the framework's `StreamAdapter`:
it splits the LLM output into sentences (the `tokenizer` argument) and sends one chunked request
per sentence, in order.

- A sentence that fails after earlier ones have played is retried on its own (up to `max_retry`).
- Audio is released to LiveKit as each chunk arrives, so a pause in Munsit's stream never holds
  back audio already received (that caused a tick and short mute ~130 ms into replies).
- PCM goes to LiveKit in whole samples; before livekit-agents 1.8.3 a half sample could turn the
  rest of a reply into static ([livekit/agents#7391](https://github.com/livekit/agents/pull/7391)).
- Lower `min_sentence_len` to cut first-audio latency; blingfire's default of 20 **characters**
  groups short Arabic sentences into one request.

## Other bits

- `await tts.list_voices()` / `await tts.list_models()` — `GET /voices` and `GET /models`.
- `<break time="500ms"/>` in the text inserts silence (max 3s per tag, 20 tags, ~30s total).
- Errors carry Munsit's own message and code (`40101` bad key, `40301` no access, `42901`
  concurrency limit) as an `APIStatusError`; 429 is retried, other 4xx are not.

# STT

## Usage

```python
from livekit.plugins import munsit

stt = munsit.STT(
    model="munsit",            # or "munsit-en-ar" for Arabic/English code-switching
    language="ar",             # only "ar" in v1
    streaming=True,
    hotwords="عبد القادر,أديب",
    listen=munsit.ListenOptions(
        sample_rate=16000,     # 8000 or 16000
        endpointing_ms=300,    # 100-5000 ms of silence that ends a turn
        smart_turn=False,
    ),
    # streaming=False only: transcribe=munsit.TranscribeOptions(return_turns=True, ...)
)
```

Through the agent config (`get_stt`):

```json
{
  "type": "MUNSIT",
  "api_key": "...",
  "model": "munsit",
  "language": "ar",
  "streaming": true,
  "enable_interim_results": true,
  "sample_rate": 16000,
  "endpointing": 300,
  "smart_turn": false,
  "hotwords": "عبد القادر,أديب",
  "correlation_id": "call-8371",
  "metadata": { "room": "r1" }
}
```

`endpointing` and `enable_interim_results` reuse the config keys the other STT providers use.
Transcribe-only extras (`return_confidence`, `return_timestamps`, `return_turns`,
`return_gender`, `return_sentiment`) go into `TranscribeOptions`.

When LiveKit's VAD decides the turn (`turn_detection="vad"`), set `smart_turn` off and
`endpointing` low (~300): otherwise Munsit's own turn wait is added before every reply.

## The two modes

| | endpoint | driven by |
|---|---|---|
| `streaming=True` | `wss://…/api/v1/listen` | `stream()` → `SpeechStream` |
| `streaming=False` | `POST /api/v1/audio/transcribe` (multipart wav) | `recognize()` |

The key goes in the `x-api-key` header on both, including the websocket handshake (the `?api_key=`
form would put it in access logs).

Event mapping on the socket:

| Munsit | LiveKit |
|---|---|
| `SpeechStarted` | `START_OF_SPEECH` (also synthesized from the first `Results`) |
| `Results` `is_final:false` | `INTERIM_TRANSCRIPT` |
| `Results` `is_final:true` | `FINAL_TRANSCRIPT` (+ `END_OF_SPEECH` when `speech_final`) |
| `UtteranceEnd` | `END_OF_SPEECH` |
| `Error` `recoverable:false` | `APIError` → reconnect (code `4002` fails fast) |
| `Error` `recoverable:true` | warning log |
| `Metadata` / `Gender` / `Sentiment` | debug log only |

- `is_final` without `speech_final` is a forced split in long speech (~60s), not a turn end.
- A `KeepAlive` goes out every 5s (Munsit closes after 12s without audio); a 30s heartbeat catches
  half-open sockets.
- Close codes `1008` (auth / limit / balance) and `4002` (bad parameters) fail fast; others
  reconnect.
- Audio is always sent as `linear16` mono, which is what LiveKit hands the stream.

# Test

Runs the plugin against a local fake of the API; no key or network needed:

```bash
pip install -e . && python tests/test_munsit.py
```
