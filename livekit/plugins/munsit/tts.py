from __future__ import annotations

import asyncio
import os
import weakref
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Any, Optional

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGiven, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TimedString

from .log import logger
from .models import TTSDialects, TTSModels

# https://docs.munsit.com/authentication
BASE_URL = "https://api.munsit.com/api/v1"
AUTH_HEADER = "x-api-key"

NUM_CHANNELS = 1
BYTES_PER_SAMPLE = 2 * NUM_CHANNELS  # PCM16

DEFAULT_MODEL = "faseeh-v1-preview"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_STABILITY = 0.5
DEFAULT_SPEED = 1.0
DEFAULT_DIALECT = "auto"

# ranges documented on https://docs.munsit.com/text-to-speech/synthesize
STABILITY_RANGE = (0.0, 1.0)
SPEED_RANGE = (0.7, 1.2)
SAMPLE_RATE_RANGE = (8000, 48000)

# covers the whole streamed body, like the upstream HTTP TTS plugins
REQUEST_TOTAL_TIMEOUT = 30.0


def _timeout(conn_options: APIConnectOptions) -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(
        total=REQUEST_TOTAL_TIMEOUT, sock_connect=conn_options.timeout
    )


def _flag(value: Optional[bool], default: bool) -> bool:
    """Resolve a config flag: `None` means the default."""
    return default if value is None else bool(value)


def _string(value: Optional[str], default: str) -> str:
    """Resolve a config string: `None` or empty means the default."""
    if not value:
        return default
    return value


def _api_key(api_key: NotGivenOr[str]) -> str:
    key = (api_key if is_given(api_key) else None) or os.environ.get("MUNSIT_API_KEY")
    if not key:
        raise ValueError("munsit: API key required. Set MUNSIT_API_KEY or pass api_key.")
    return key


def _base_url(base_url: NotGivenOr[str]) -> str:
    url = base_url if is_given(base_url) and base_url else os.environ.get("MUNSIT_BASE_URL")
    return (url or BASE_URL).rstrip("/")


def _number(name: str, value: Any, default: float, bounds: tuple[float, float]) -> float:
    """Resolve a config number (JSON may send `null` or a string); out of range raises."""
    if value is None:
        return default

    value = float(value)
    lo, hi = bounds
    if not lo <= value <= hi:
        raise ValueError(f"munsit: {name} must be between {lo} and {hi}, got {value}")
    return value


@dataclass
class _TTSOptions:
    model: str
    voice_id: str
    stability: float
    speed: float
    sample_rate: int
    dialect: str
    streaming: bool
    base_url: str
    api_key: str
    ssl: bool

    @property
    def synthesize_url(self) -> str:
        return f"{self.base_url}/text-to-speech/{self.model}"

    @property
    def headers(self) -> dict[str, str]:
        return {AUTH_HEADER: self.api_key, "Content-Type": "application/json"}

    def payload(self, text: str, *, streaming: bool | None = None) -> dict[str, Any]:
        return {
            "voice_id": self.voice_id,
            "text": text,
            "stability": self.stability,
            "speed": self.speed,
            "sample_rate": self.sample_rate,
            "dialect": self.dialect,
            "streaming": self.streaming if streaming is None else streaming,
        }


async def _pcm_samples(res: aiohttp.ClientResponse) -> AsyncIterator[bytes]:
    """Yield a PCM16 body in whole samples.

    A chunk can end mid-sample; livekit-agents < 1.8.3 drops that half sample on a
    mid-stream flush and the rest of the reply plays as static (livekit/agents#7391).
    A stray byte at the end of a response is dropped, not carried into the next one.
    """
    carry = b""
    async for data in res.content.iter_any():
        if carry:
            data = carry + data
        whole = len(data) - len(data) % BYTES_PER_SAMPLE
        data, carry = data[:whole], data[whole:]
        if data:
            yield data

    if carry:
        logger.warning("munsit tts response ended mid-sample, dropped %d byte(s)", len(carry))


def _emit_now(output_emitter: tts.AudioEmitter, data: bytes) -> None:
    """Push PCM and flush it, so the emitter never holds audio back.

    Audio held to fill a frame made playback run dry ~130 ms into replies (a tick + mute).
    """
    output_emitter.push(data)
    output_emitter.flush()


async def _raise_for_status(res: aiohttp.ClientResponse, request_id: str) -> None:
    if 200 <= res.status < 300:
        return

    try:
        body: Any = await res.json()
    except Exception:
        try:
            body = await res.text()
        except Exception:
            body = None

    message = res.reason or "munsit tts request failed"
    if isinstance(body, dict):
        # the API sends errorCode/errorMessage; the docs say code/message
        message = body.get("errorMessage") or body.get("message") or message

    raise APIStatusError(message, status_code=res.status, request_id=request_id, body=body)


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice_id: str,
        model: Optional[TTSModels | str] = DEFAULT_MODEL,
        stability: Optional[float] = DEFAULT_STABILITY,
        speed: Optional[float] = DEFAULT_SPEED,
        sample_rate: Optional[int] = DEFAULT_SAMPLE_RATE,
        dialect: Optional[TTSDialects | str] = DEFAULT_DIALECT,
        streaming: Optional[bool] = True,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        http_session: Optional[aiohttp.ClientSession] = None,
        ssl: bool = True,
    ) -> None:
        """Create a new instance of Munsit TTS.

        Args:
            voice_id: Voice to speak with, from `GET /voices` (see `list_voices`).
            model: Model id, from `GET /models` (see `list_models`).
            stability: 0.0-1.0. Lower is more expressive, higher is more consistent.
            speed: 0.7-1.2 playback rate.
            sample_rate: 8000-48000 Hz. 48000 is the engine-native rate.
            dialect: "auto", "emirati" or "fusha".
            streaming: True streams text in through `stream()` and plays PCM16 chunks
                as they are generated. False makes the session call `synthesize()` per
                sentence, each answered with a complete WAV.
            api_key: Munsit API key, or `MUNSIT_API_KEY` in the environment.
            base_url: API root; defaults to `MUNSIT_BASE_URL` or the global endpoint.
                Use `https://ae.api.faseeh.ai/api/v1` for UAE data residency.
            tokenizer: Splits streamed text into sentences, one request each.
            http_session: Session to reuse instead of the agent's shared one.
            ssl: Verify TLS certificates.

        Every argument but `voice_id` also accepts `None`, meaning the default above.
        """
        streaming = _flag(streaming, True)
        stability = _number("stability", stability, DEFAULT_STABILITY, STABILITY_RANGE)
        speed = _number("speed", speed, DEFAULT_SPEED, SPEED_RANGE)
        sample_rate = int(
            _number("sample_rate", sample_rate, DEFAULT_SAMPLE_RATE, SAMPLE_RATE_RANGE)
        )

        super().__init__(
            # sentence-level timed transcripts
            capabilities=tts.TTSCapabilities(streaming=streaming, aligned_transcript=streaming),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        munsit_api_key = _api_key(api_key)
        if not voice_id:
            raise ValueError("munsit: voice_id is required, see https://api.munsit.com/api/v1/voices")

        self._opts = _TTSOptions(
            model=_string(model, DEFAULT_MODEL),
            voice_id=voice_id,
            stability=stability,
            speed=speed,
            sample_rate=sample_rate,
            dialect=_string(dialect, DEFAULT_DIALECT),
            streaming=streaming,
            base_url=_base_url(base_url),
            api_key=munsit_api_key,
            ssl=ssl,
        )
        self._sentence_tokenizer = (
            tokenizer
            if is_given(tokenizer)
            else tokenize.blingfire.SentenceTokenizer(retain_format=True)
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "munsit"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        stability: NotGivenOr[float] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        dialect: NotGivenOr[TTSDialects | str] = NOT_GIVEN,
    ) -> None:
        """Apply to requests made from here on; `sample_rate` and `streaming` are fixed."""
        # isinstance, not is_given: PyCharm doesn't narrow is_given() on Literal unions
        if not isinstance(voice_id, NotGiven):
            self._opts.voice_id = voice_id
        if not isinstance(model, NotGiven):
            self._opts.model = model
        if not isinstance(stability, NotGiven):
            self._opts.stability = _number("stability", stability, DEFAULT_STABILITY, STABILITY_RANGE)
        if not isinstance(speed, NotGiven):
            self._opts.speed = _number("speed", speed, DEFAULT_SPEED, SPEED_RANGE)
        if not isinstance(dialect, NotGiven):
            self._opts.dialect = dialect

    async def list_voices(self) -> list[dict[str, Any]]:
        """`GET /voices` — the voices this key can use, each with its `voice_id`."""
        return await self._get("/voices")

    async def list_models(self) -> list[dict[str, Any]]:
        """`GET /models` — the models this key can use, each with its `model_id`."""
        return await self._get("/models")

    async def _get(self, path: str) -> list[dict[str, Any]]:
        try:
            async with self._ensure_session().get(
                f"{self._opts.base_url}{path}",
                headers={AUTH_HEADER: self._opts.api_key},
                timeout=_timeout(DEFAULT_API_CONNECT_OPTIONS),
                ssl=self._opts.ssl,
            ) as res:
                await _raise_for_status(res, path)
                return await res.json()
        except APIStatusError:
            raise
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except Exception as e:
            raise APIConnectionError() from e

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in self._streams.copy():  # a snapshot: aclose() awaits
            await stream.aclose()
        self._streams.clear()
        await super().aclose()


class ChunkedStream(tts.ChunkedStream):
    """One request for the whole text: PCM16 read as it arrives, or a WAV if not streaming."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()

        try:
            async with self._tts._ensure_session().post(
                self._opts.synthesize_url,
                headers=self._opts.headers,
                json=self._opts.payload(self._input_text),
                timeout=_timeout(self._conn_options),
                ssl=self._opts.ssl,
            ) as res:
                await _raise_for_status(res, request_id)

                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm" if self._opts.streaming else "audio/wav",
                )

                if self._opts.streaming:
                    async for data in _pcm_samples(res):
                        _emit_now(output_emitter, data)
                else:
                    async for data in res.content.iter_any():
                        output_emitter.push(data)

                output_emitter.flush()
        except APIStatusError:
            raise
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=request_id, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """One request per sentence, in order, like the framework's `StreamAdapter`.

    Munsit takes no incremental text, so each sentence is its own chunked PCM16
    request, and its text is pushed as a timed transcript where its audio starts.
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=utils.shortuuid())

        sent_stream = self._tts._sentence_tokenizer.stream()

        async def _tokenize_input() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_stream.flush()
                    continue
                sent_stream.push_text(data)

            sent_stream.end_input()

        async def _synthesize() -> None:
            duration = 0.0
            async for ev in sent_stream:
                # the transcript keeps the whitespace; Munsit gets it stripped
                if not (text := ev.token.strip()):
                    continue
                self._mark_started()
                timed = TimedString(text=ev.token, start_time=duration)
                output_emitter.push_timed_transcript(timed)
                duration += await self._synthesize_sentence(
                    text, request_id, output_emitter, retry=duration > 0
                )

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_synthesize()),
        ]
        try:
            await asyncio.gather(*tasks)
        except APIError:
            raise
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            await sent_stream.aclose()

    async def _synthesize_sentence(
        self, text: str, request_id: str, output_emitter: tts.AudioEmitter, *, retry: bool
    ) -> float:
        """Stream one sentence into the emitter and return its duration in seconds.

        Once audio has played the framework no longer retries, so with `retry` a failed
        sentence is re-requested while it has pushed nothing (the Soniox plugin's rule).
        """
        pushed = 0
        attempt = 0
        while True:
            try:
                async with self._tts._ensure_session().post(
                    self._opts.synthesize_url,
                    headers=self._opts.headers,
                    json=self._opts.payload(text, streaming=True),
                    timeout=_timeout(self._conn_options),
                    ssl=self._opts.ssl,
                ) as res:
                    await _raise_for_status(res, request_id)

                    logger.debug("munsit tts sentence started", extra={"request_id": request_id})
                    async for data in _pcm_samples(res):
                        _emit_now(output_emitter, data)
                        pushed += len(data)

                return pushed / (self._opts.sample_rate * BYTES_PER_SAMPLE)
            except (APIError, asyncio.TimeoutError, aiohttp.ClientError) as e:
                retryable = e.retryable if isinstance(e, APIError) else True
                if not (
                    retry
                    and retryable
                    and pushed == 0
                    and attempt < self._conn_options.max_retry
                ):
                    raise  # _run maps it to an APIError

                interval = self._conn_options._interval_for_retry(attempt)
                logger.warning(
                    "munsit tts sentence failed: %s, retrying in %ss",
                    e,
                    interval,
                    extra={"request_id": request_id, "attempt": attempt + 1},
                )
                await asyncio.sleep(interval)
                attempt += 1
