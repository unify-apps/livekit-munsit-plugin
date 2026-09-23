# Munsit plugin for LiveKit Agents

Arabic [TTS](https://docs.munsit.com/text-to-speech/get-started) and
[STT](https://docs.munsit.com/speech-to-text/transcribe) over the Munsit / Faseeh API, in the same
shape as the other LiveKit plugins. Lives in this folder for now; it will move to its own GitHub
repo + `requirements.txt` entry later, so it depends on nothing outside `livekit-agents` and
`aiohttp`.

# TTS

## Usage

```python
import munsit

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

| | request | response | used by |
|---|---|---|---|
| `streaming=False` | one request for the whole text | complete `audio/wav` | `synthesize()` |
| `streaming=True` | one request per sentence | chunked raw PCM16 mono | `stream()`, and `synthesize()` reads the body as it arrives |

`streaming` on the constructor sets `TTSCapabilities.streaming`, which is what decides whether the
agent session drives `stream()` or wraps `synthesize()` in a `StreamAdapter`.

Munsit's streaming endpoint takes no incremental text input, so `SynthesizeStream` tokenizes the
incoming text into sentences (blingfire by default) and issues one chunked request per sentence,
in order, pushing every chunk into a single audio segment. `stream()` always asks for chunks even
when `streaming=False`, because concatenated WAV headers mid-segment would not decode.

Lower `min_sentence_len` on the tokenizer to cut first-audio latency; blingfire's default of 20
**characters** groups short Arabic sentences into one request.

## Other bits

- `await tts.list_voices()` / `await tts.list_models()` — `GET /voices` and `GET /models`.
- `<break time="500ms"/>` in the text inserts silence (max 3s per tag, 20 tags, ~30s total).
- Errors carry Munsit's own message and code (`40101` bad key, `40301` no access, `42901`
  concurrency limit) as an `APIStatusError`; 429 is retried, other 4xx are not.

# STT

## Usage

```python
import munsit

stt = munsit.STT(
    model="munsit",            # or "munsit-en-ar" for Arabic/English code-switching
    language="ar",             # only "ar" in v1
    streaming=True,
    interim_results=True,
    sample_rate=16000,         # 8000 or 16000
    endpointing_ms=800,        # 100-5000 ms of silence that ends a turn
    smart_turn=True,
    hotwords="عبد القادر,أديب",
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
  "endpointing": 800,
  "smart_turn": true,
  "hotwords": "عبد القادر,أديب",
  "correlation_id": "call-8371",
  "metadata": { "room": "r1" }
}
```

`endpointing` and `enable_interim_results` reuse the config keys the other STT providers already
use. Transcribe-only extras: `return_confidence`, `return_timestamps`, `return_turns`,
`return_gender`, `return_sentiment`.

## The two modes

| | endpoint | driven by |
|---|---|---|
| `streaming=True` | `wss://…/api/v1/listen` | `stream()` → `SpeechStream` |
| `streaming=False` | `POST /api/v1/audio/transcribe` (multipart wav) | `recognize()` |

`streaming` sets `STTCapabilities.streaming`, which is what decides whether the agent session opens
a socket or buffers an utterance and posts it. The key goes in the `x-api-key` header on both,
including the websocket handshake — the `?api_key=` form the docs offer for browsers would put it
in every access log.

Event mapping on the socket:

| Munsit | LiveKit |
|---|---|
| `SpeechStarted` | `START_OF_SPEECH` (also synthesized from the first `Results`) |
| `Results` `is_final:false` | `INTERIM_TRANSCRIPT` |
| `Results` `is_final:true` | `FINAL_TRANSCRIPT` (+ `END_OF_SPEECH` when `speech_final`) |
| `UtteranceEnd` | `END_OF_SPEECH` |
| `Error` `recoverable:false` | `APIStatusError` → reconnect |
| `Metadata` / `Gender` / `Sentiment` | debug log only |

`is_final: true` with `speech_final: false` is a forced split during long speech (~60s), not a turn
end, so it does not close the turn. `Gender` and `Sentiment` arrive *after* the turn's final
transcript, so there is no event left to attach them to — they are logged at debug.

A `KeepAlive` goes out every 5s (the socket closes with `1011` after 12s without audio) and the
connection carries a 30s heartbeat so a half-open socket triggers a reconnect instead of hanging.
Close code `1008` (auth / session limit / no balance) is mapped to a non-retryable error; the rest
reconnect.

# Test

Both drive the real plugin against a local stand-in for the API — no key and no network needed:

```bash
venv/bin/python munsit_test.py       # TTS: both modes + error path
venv/bin/python munsit_stt_test.py   # STT: transcribe + websocket event sequence
```

`munsit_try.py` runs the TTS against the live API and writes a wav.
