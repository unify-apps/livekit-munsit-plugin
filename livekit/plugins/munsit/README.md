# Munsit plugin for LiveKit Agents

Arabic [TTS](https://docs.munsit.com/text-to-speech/get-started) and
[STT](https://docs.munsit.com/speech-to-text/transcribe) over the Munsit / Faseeh API, in the same
shape as the other LiveKit plugins. Lives in this folder for now; it will move to its own GitHub
repo + `requirements.txt` entry later, so it depends on nothing outside `livekit-agents` and
`aiohttp`.

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

The endpoint takes no incremental text input, so `SynthesizeStream` works like the framework's
`StreamAdapter`: it tokenizes the LLM output into sentences (the `tokenizer` argument, blingfire
by default), sends one chunked request per sentence, in order, and stamps each sentence's text
with the time its audio starts. Audio arrives faster than it plays, so the next request usually
starts while the previous sentence is still playing and its time-to-first-byte is not heard.

Once part of a reply has played, the framework no longer retries the stream, so a sentence that
fails before producing any audio is retried on its own (up to `max_retry`, like the Soniox
plugin); before that, the framework's own retry replays the stream. PCM is handed to LiveKit in
whole 16-bit samples: network chunks can end mid-sample, and before livekit-agents 1.8.3 a
mid-stream flush dropped that half sample and turned the rest of the reply into static
([livekit/agents#7391](https://github.com/livekit/agents/pull/7391)).

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
from livekit.plugins import munsit

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
| `Error` `recoverable:false` | `APIError` → reconnect (code `4002` fails fast) |
| `Error` `recoverable:true` | warning log |
| `Metadata` / `Gender` / `Sentiment` | debug log only |

`is_final: true` with `speech_final: false` is a forced split during long speech (~60s), not a turn
end, so it does not close the turn. `Gender` and `Sentiment` arrive *after* the turn's final
transcript, so there is no event left to attach them to — they are logged at debug.

A `KeepAlive` goes out every 5s (the socket closes with `1011` after 12s without audio) and the
connection carries a 30s heartbeat so a half-open socket triggers a reconnect instead of hanging.
Close codes `1008` (auth / session limit / no balance) and `4002` (invalid connection parameters)
map to a non-retryable error; the rest reconnect.

Audio always goes out as `linear16` mono, which is what LiveKit hands the stream. `encoding` and
`num_channels` are accepted for configs that spell them out, but only as `"linear16"` and `1` —
Munsit's `mulaw` / `alaw` would mislabel PCM16 bytes, so any other value raises.

# Test

Drives the real plugin against a local stand-in for the API — no key and no network needed:

```bash
pip install -e . && python tests/test_munsit.py
```
