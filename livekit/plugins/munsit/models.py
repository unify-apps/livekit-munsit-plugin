from typing import Literal

# https://docs.munsit.com/text-to-speech/models — `GET /models` returns the live list.
TTSModels = Literal["faseeh-v1-preview"]

# https://docs.munsit.com/text-to-speech/synthesize — `dialect` field.
TTSDialects = Literal["auto", "emirati", "fusha"]

# https://docs.munsit.com/speech-to-text/transcribe — `munsit-en-ar` code-switches.
STTModels = Literal["munsit", "munsit-en-ar"]

# Munsit's `encoding` also takes mulaw / alaw, but LiveKit hands the stream PCM16, so
# linear16 is the only one that describes the bytes sent (like Cartesia's pcm_s16le).
STTEncodings = Literal["linear16"]
