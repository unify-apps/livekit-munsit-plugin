from typing import Literal

# https://docs.munsit.com/text-to-speech/models — `GET /models` returns the live list.
TTSModels = Literal["faseeh-v1-preview"]

# https://docs.munsit.com/text-to-speech/synthesize — `dialect` field.
TTSDialects = Literal["auto", "emirati", "fusha"]

# https://docs.munsit.com/speech-to-text/transcribe — `munsit-en-ar` code-switches.
STTModels = Literal["munsit", "munsit-en-ar"]
