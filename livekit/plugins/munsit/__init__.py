from . import models
from .stt import STT, ListenOptions, SpeechStream, TranscribeOptions
from .tts import TTS, ChunkedStream, SynthesizeStream
from .version import __version__

__all__ = [
    "TTS",
    "ChunkedStream",
    "SynthesizeStream",
    "STT",
    "ListenOptions",
    "TranscribeOptions",
    "SpeechStream",
    "models",
    "__version__",
]


from livekit.agents import Plugin

from .log import logger


class MunsitPlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__, __package__ or __name__, logger)


Plugin.register_plugin(MunsitPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
