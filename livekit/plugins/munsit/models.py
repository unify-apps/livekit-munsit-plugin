from typing import Literal

# https://docs.munsit.com/text-to-speech/models — `GET /models` returns the live list.
TTSModels = Literal["faseeh-v1-preview"]

# https://docs.munsit.com/text-to-speech/synthesize — `dialect` field.
TTSDialects = Literal["auto", "emirati", "fusha"]

# https://docs.munsit.com/speech-to-text/transcribe — `munsit-en-ar` code-switches.
STTModels = Literal["munsit", "munsit-en-ar"]

# Munsit also takes mulaw / alaw, but LiveKit sends PCM16 (like Cartesia's pcm_s16le).
STTEncodings = Literal["linear16"]
