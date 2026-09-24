from __future__ import annotations

import asyncio
import base64
import json
import time
import weakref
from dataclasses import dataclass, replace
from typing import Any, Optional
from urllib.parse import urlencode

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGiven, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given
from livekit.agents.voice.io import TimedString

from .log import logger
from .models import STTModels
from .tts import AUTH_HEADER, _api_key, _base_url, _flag, _raise_for_status, _string

DEFAULT_STT_MODEL = "munsit"
DEFAULT_LANGUAGE = "ar"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_ENDPOINTING_MS = 800
ENCODING = "linear16"

# https://docs.munsit.com/speech-to-text/streaming — connection parameters
SAMPLE_RATES = (8000, 16000)
ENDPOINTING_RANGE = (100, 5000)

# the socket closes with 1011 after 12s without audio, so keep well inside it
KEEPALIVE_INTERVAL = 5.0
# notices a half-open socket, which would otherwise hang _recv_events
WS_HEARTBEAT = 30.0
# seconds of streamed audio per usage report, the cadence LiveKit's own plugins use
USAGE_REPORT_INTERVAL = 5.0

# transcribe takes up to 60 minutes of audio, so allow a slow upload
TRANSCRIBE_TOTAL_TIMEOUT = 300.0


# message types that end a socket; unexpected unless we are closing it
_SOCKET_ENDED = (
    aiohttp.WSMsgType.CLOSED,
    aiohttp.WSMsgType.CLOSE,
    aiohttp.WSMsgType.CLOSING,
    aiohttp.WSMsgType.ERROR,
)


def _language(value: Any, default: str) -> LanguageCode:
    """The language to report: `value` when it is a non-empty string, else `default`."""
    return LanguageCode(value if isinstance(value, str) and value else default)


def _word_time(word: dict, key: str, offset: float) -> float:
    # a null timing must not raise, or the transcript is lost
    return (word.get(key) or 0.0) + offset


def _integer(value: Optional[int], default: int) -> int:
    """Resolve a config integer: `None` means the default."""
    return default if value is None else int(value)


def _socket_error(ws: aiohttp.ClientWebSocketResponse, msg: aiohttp.WSMessage) -> APIError:
    """The error for a socket that ended while it should still be open."""
    if msg.type == aiohttp.WSMsgType.ERROR:
        error = APIConnectionError("munsit connection lost")
        error.__cause__ = ws.exception()  # as `raise ... from ws.exception()`
        return error

    code = ws.close_code or -1
    return APIStatusError(
        message=f"munsit connection closed unexpectedly ({code})",
        # 1008 auth / limit / balance, 4002 bad parameters: a retry can't fix them
        status_code=400 if code in (1008, 4002) else 500,
        body=f"{msg.data=} {msg.extra=}",
    )


def _check_endpointing(endpointing_ms: int) -> None:
    if not ENDPOINTING_RANGE[0] <= endpointing_ms <= ENDPOINTING_RANGE[1]:
        raise ValueError(
            f"munsit: endpointing_ms must be between {ENDPOINTING_RANGE[0]} and "
            f"{ENDPOINTING_RANGE[1]}, got {endpointing_ms}"
        )


@dataclass
class ListenOptions:
    """`/listen` websocket settings, used when `streaming=True`. `None` means the default."""

    sample_rate: Optional[int] = DEFAULT_SAMPLE_RATE  # 8000 or 16000; audio is resampled to it
    interim_results: Optional[bool] = True  # emit partial transcripts
    endpointing_ms: Optional[int] = DEFAULT_ENDPOINTING_MS  # 100-5000 ms of silence ends a turn
    smart_turn: Optional[bool] = True  # semantic end-of-turn model on top of the silence timer
    correlation_id: Optional[str] = None  # your session id, echoed back in Metadata
    metadata: Optional[dict[str, Any]] = None  # JSON (<= 2 KB), sent base64-encoded


@dataclass
class TranscribeOptions:
    """Extra `/audio/transcribe` fields, used when `streaming=False`. `None` means the default.

    Turns and analysis come back in `SpeechData.metadata`.
    """

    return_confidence: Optional[bool] = False  # per-word confidence
    return_timestamps: Optional[bool] = True  # word timings
    return_turns: Optional[bool] = False  # the turns array
    return_gender: Optional[bool] = False  # gender analysis per turn
    return_sentiment: Optional[bool] = False  # sentiment analysis per turn


@dataclass
class _STTOptions:
    model: str
    language: str
    api_key: str
    base_url: str
    ssl: bool
    # streaming (query parameters on /listen)
    sample_rate: int
    interim_results: bool
    endpointing_ms: int
    smart_turn: bool
    correlation_id: Optional[str]
    metadata: Optional[dict[str, Any]]
    # both surfaces
    hotwords: Optional[str]
    # transcribe only (form fields on /audio/transcribe)
    return_confidence: bool
    return_timestamps: bool
    return_turns: bool
    return_gender: bool
    return_sentiment: bool

    @property
    def transcribe_url(self) -> str:
        return f"{self.base_url}/audio/transcribe"

    @property
    def listen_url(self) -> str:
        ws_base = self.base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        query: dict[str, Any] = {
            "model": self.model,
            "language": self.language,
            # LiveKit hands the stream mono PCM16, resampled to sample_rate
            "encoding": ENCODING,
            "sample_rate": self.sample_rate,
            "channels": 1,
            "interim_results": str(self.interim_results).lower(),
            "endpointing": self.endpointing_ms,
            "smart_turn": str(self.smart_turn).lower(),
        }
        if self.hotwords:
            query["hotwords"] = self.hotwords
        if self.correlation_id:
            query["correlation_id"] = self.correlation_id
        if self.metadata:
            query["metadata"] = base64.b64encode(
                json.dumps(self.metadata).encode()
            ).decode()

        return f"{ws_base}/listen?{urlencode(query)}"


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: Optional[STTModels | str] = DEFAULT_STT_MODEL,
        language: Optional[str] = DEFAULT_LANGUAGE,
        streaming: Optional[bool] = True,
        hotwords: Optional[str] = None,
        listen: Optional[ListenOptions] = None,
        transcribe: Optional[TranscribeOptions] = None,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: Optional[aiohttp.ClientSession] = None,
        ssl: bool = True,
    ) -> None:
        """Create a new instance of Munsit STT.

        Args:
            model: "munsit" (Arabic) or "munsit-en-ar" (Arabic/English code-switching).
            language: Only "ar" in v1.
            streaming: True uses the `/listen` websocket, so partial transcripts arrive
                while the caller is still talking. False uses `/audio/transcribe`, which
                takes a complete utterance and answers once.
            hotwords: Comma-separated custom vocabulary. Ignored by "munsit-en-ar".
            listen: Websocket settings (sample rate, endpointing, smart turn, ...).
            transcribe: Extra `/audio/transcribe` fields (confidence, turns, ...).
            api_key: Munsit API key, or `MUNSIT_API_KEY` in the environment.
            base_url: API root; defaults to `MUNSIT_BASE_URL` or the global endpoint.
                The websocket url is derived from it.
            http_session: Session to reuse instead of the agent's shared one.
            ssl: Verify TLS certificates.

        Every argument accepts `None`, meaning the default. Audio always goes out as
        mono linear16, which is what LiveKit hands the stream.
        """
        streaming = _flag(streaming, True)
        listen = listen or ListenOptions()
        transcribe = transcribe or TranscribeOptions()
        interim_results = _flag(listen.interim_results, True)
        sample_rate = _integer(listen.sample_rate, DEFAULT_SAMPLE_RATE)
        endpointing_ms = _integer(listen.endpointing_ms, DEFAULT_ENDPOINTING_MS)

        if sample_rate not in SAMPLE_RATES:
            raise ValueError(f"munsit: sample_rate must be one of {SAMPLE_RATES}, got {sample_rate}")
        _check_endpointing(endpointing_ms)

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=streaming,
                interim_results=streaming and interim_results,
            )
        )

        self._opts = _STTOptions(
            model=_string(model, DEFAULT_STT_MODEL),
            language=_string(language, DEFAULT_LANGUAGE),
            api_key=_api_key(api_key),
            base_url=_base_url(base_url),
            ssl=ssl,
            sample_rate=sample_rate,
            interim_results=interim_results,
            endpointing_ms=endpointing_ms,
            smart_turn=_flag(listen.smart_turn, True),
            correlation_id=listen.correlation_id,
            metadata=listen.metadata,
            hotwords=hotwords,
            return_confidence=bool(transcribe.return_confidence),
            return_timestamps=_flag(transcribe.return_timestamps, True),
            return_turns=bool(transcribe.return_turns),
            return_gender=bool(transcribe.return_gender),
            return_sentiment=bool(transcribe.return_sentiment),
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

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
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        hotwords: NotGivenOr[str] = NOT_GIVEN,
        endpointing_ms: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """Update the options. Live streams re-apply them by reconnecting."""
        if not isinstance(endpointing_ms, NotGiven):
            _check_endpointing(endpointing_ms)  # first, so a bad value changes nothing
        # isinstance, not is_given: PyCharm doesn't narrow is_given() on Literal unions
        if not isinstance(model, NotGiven):
            self._opts.model = model
        if not isinstance(language, NotGiven):
            self._opts.language = language
        if not isinstance(hotwords, NotGiven):
            self._opts.hotwords = hotwords
        if not isinstance(endpointing_ms, NotGiven):
            self._opts.endpointing_ms = endpointing_ms

        for stream in self._streams:
            stream.update_options(
                model=model,
                language=language,
                hotwords=hotwords,
                endpointing_ms=endpointing_ms,
            )

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """POST the whole utterance as a wav to `/audio/transcribe`.

        The endpoint takes no language field; `language` only labels the result.
        """
        opts = self._opts
        wav = rtc.combine_audio_frames(buffer).to_wav_bytes()

        form = aiohttp.FormData()
        form.add_field("file", wav, filename="audio.wav", content_type="audio/wav")
        form.add_field("model", opts.model)
        for field, value in (
            ("return_confidence", opts.return_confidence),
            ("return_timestamps", opts.return_timestamps),
            ("return_turns", opts.return_turns),
            ("return_gender", opts.return_gender),
            ("return_sentiment", opts.return_sentiment),
        ):
            form.add_field(field, str(value).lower())
        if opts.hotwords:
            form.add_field("hotwords", opts.hotwords)

        try:
            async with self._ensure_session().post(
                opts.transcribe_url,
                data=form,
                headers={AUTH_HEADER: opts.api_key},
                timeout=aiohttp.ClientTimeout(
                    total=TRANSCRIBE_TOTAL_TIMEOUT, sock_connect=conn_options.timeout
                ),
                ssl=opts.ssl,
            ) as res:
                await _raise_for_status(res, "transcribe")
                body = await res.json()
        except APIStatusError:
            raise
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except Exception as e:
            raise APIConnectionError() from e

        data = body.get("data") or {}
        words = [
            TimedString(
                text=w.get("word", ""),
                start_time=w.get("start", NOT_GIVEN),
                end_time=w.get("end", NOT_GIVEN),
                confidence=w.get("confidence", NOT_GIVEN),
            )
            for w in data.get("timestamps") or []
        ]

        # per-word only, and only with return_confidence
        confidences = [w.confidence for w in words if isinstance(w.confidence, (int, float))]

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=data.get("transcriptionId", ""),
            alternatives=[
                stt.SpeechData(
                    language=_language(language, opts.language),
                    text=data.get("transcription", ""),
                    start_time=0.0,
                    end_time=data.get("duration", 0.0) or 0.0,
                    confidence=sum(confidences) / len(confidences) if confidences else 1.0,
                    words=words or None,
                    # turns / analysis only exist when the matching return_* flag is on
                    metadata={
                        k: data[k] for k in ("turns", "analysis", "stats") if data.get(k)
                    }
                    or None,
                )
            ],
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        opts = replace(self._opts)
        if is_given(language):
            opts.language = language

        stream = SpeechStream(stt=self, opts=opts, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in self._streams.copy():  # a snapshot: aclose() awaits
            await stream.aclose()
        self._streams.clear()
        await super().aclose()


class SpeechStream(stt.SpeechStream):
    _KEEPALIVE_MSG: str = json.dumps({"type": "KeepAlive"})
    _CLOSE_MSG: str = json.dumps({"type": "CloseStream"})

    def __init__(self, *, stt: STT, opts: _STTOptions, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._stt: STT = stt
        self._session = stt._ensure_session()
        self._speaking = False
        self._closing_ws = False
        self._reconnect_event = asyncio.Event()

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        hotwords: NotGivenOr[str] = NOT_GIVEN,
        endpointing_ms: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        if not isinstance(endpointing_ms, NotGiven):
            _check_endpointing(endpointing_ms)  # first, so a bad value changes nothing
        if not isinstance(model, NotGiven):
            self._opts.model = model
        if not isinstance(language, NotGiven):
            self._opts.language = language
        if not isinstance(hotwords, NotGiven):
            self._opts.hotwords = hotwords
        if not isinstance(endpointing_ms, NotGiven):
            self._opts.endpointing_ms = endpointing_ms

        self._reconnect_event.set()

    async def _run(self) -> None:
        self._closing_ws = False
        while True:
            ws: aiohttp.ClientWebSocketResponse | None = None
            try:
                ws = await self._connect_ws()
                # Munsit's clock restarts with every socket; this is an upper bound that
                # _send_audio pulls back to when the socket's first audio was captured
                self.start_time = time.time()
                if not await self._run_connection(ws):
                    return
            finally:
                if ws is not None:
                    await ws.close()

    async def _run_connection(self, ws: aiohttp.ClientWebSocketResponse) -> bool:
        """Serve one socket; True when update_options asked for a reconnect."""
        tasks = [
            asyncio.create_task(self._send_audio(ws)),
            asyncio.create_task(self._recv_events(ws)),
        ]
        tasks_group = asyncio.gather(*tasks)
        # outside the task group, so its sleep never delays closing; _recv_events and
        # the heartbeat notice a dropped socket
        keepalive = asyncio.create_task(self._keepalive(ws))
        wait_reconnect = asyncio.create_task(self._reconnect_event.wait())
        try:
            done, _ = await asyncio.wait(
                (tasks_group, wait_reconnect), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if task != wait_reconnect:
                    task.result()

            if wait_reconnect not in done:
                return False

            self._reconnect_event.clear()
            return True
        finally:
            await utils.aio.gracefully_cancel(*tasks, keepalive, wait_reconnect)
            tasks_group.cancel()
            tasks_group.exception()  # retrieve the exception

    def _closing(self) -> bool:
        """A socket error is expected once CloseStream went out or the session is gone."""
        return self._closing_ws or self._session.closed

    async def _keepalive(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        try:
            while True:
                await ws.send_str(SpeechStream._KEEPALIVE_MSG)
                await asyncio.sleep(KEEPALIVE_INTERVAL)
        except (aiohttp.ClientError, ConnectionError):
            return

    @utils.log_exceptions(logger=logger)
    async def _send_audio(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        # 50ms per frame, inside the 20-200ms the docs recommend
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            samples_per_channel=self._opts.sample_rate // 20,
        )
        sent = 0.0  # seconds of audio on this socket: Munsit's clock
        unreported = 0.0  # of it, not yet reported as usage
        try:
            async for data in self._input_ch:
                frames: list[rtc.AudioFrame] = []
                if isinstance(data, rtc.AudioFrame):
                    frames.extend(audio_bstream.write(data.data.tobytes()))
                elif isinstance(data, self._FlushSentinel):
                    frames.extend(audio_bstream.flush())

                for frame in frames:
                    await ws.send_bytes(frame.data.tobytes())
                    sent += frame.duration
                    unreported += frame.duration
                    # audio buffered during the handshake goes out first, so anchoring at
                    # connect put every timestamp late; the frame just sent was captured
                    # no later than now, and the smallest (now - sent) is the true start
                    self.start_time = min(self.start_time, time.time() - sent)
                    if unreported >= USAGE_REPORT_INTERVAL:
                        self._report_usage(unreported)
                        unreported = 0.0

            self._closing_ws = True
            await ws.send_str(SpeechStream._CLOSE_MSG)
        except (aiohttp.ClientError, ConnectionError) as e:
            if self._closing():
                return
            raise APIConnectionError("munsit connection closed unexpectedly") from e
        finally:
            # Munsit bills the audio it received: the tail too, on close, reconnect or drop
            self._report_usage(unreported)

    def _report_usage(self, audio_duration: float) -> None:
        """LiveKit counts streamed STT usage only from these events."""
        if audio_duration > 0:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.RECOGNITION_USAGE,
                    recognition_usage=stt.RecognitionUsage(audio_duration=audio_duration),
                )
            )

    @utils.log_exceptions(logger=logger)
    async def _recv_events(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while True:
            msg = await ws.receive()
            if msg.type in _SOCKET_ENDED:
                if self._closing():
                    return
                raise _socket_error(ws, msg)

            if msg.type != aiohttp.WSMsgType.TEXT:
                logger.warning("unexpected munsit message type %s", msg.type)
                continue

            self._handle_text(msg.data)

    def _handle_text(self, text: str) -> None:
        try:
            self._process_stream_event(json.loads(text))
        except APIError:
            # a non-recoverable Error event: _main_task decides on the retry
            raise
        except Exception:
            logger.exception("failed to process munsit message")

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        try:
            return await asyncio.wait_for(
                self._session.ws_connect(
                    self._opts.listen_url,
                    # the query-parameter form would put the key in every access log
                    headers={AUTH_HEADER: self._opts.api_key},
                    heartbeat=WS_HEARTBEAT,
                    ssl=self._opts.ssl,
                ),
                self._conn_options.timeout,
            )
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(message=e.message, status_code=e.status) from None
        except Exception as e:
            # DNS / refused / TLS: retryable, or one network blip ends the session
            raise APIConnectionError("failed to connect to munsit") from e

    def _process_stream_event(self, data: dict) -> None:
        event_type = data.get("type")
        if event_type == "SpeechStarted":
            self._start_speaking(speech_start_time=data.get("ts"))
        elif event_type == "Results":
            self._on_results(data)
        elif event_type == "UtteranceEnd":
            self._end_speaking(speech_end_time=data.get("last_word_end"))
        elif event_type == "Error":
            self._on_error(data)
        elif event_type in ("Metadata", "Gender", "Sentiment"):
            # Gender / Sentiment arrive after the final transcript: nothing to attach them to
            logger.debug("munsit %s: %s", event_type, data)
        else:
            logger.warning("unexpected munsit event %s", event_type)

    def _on_results(self, data: dict) -> None:
        transcript = data.get("transcript") or ""
        is_final = bool(data.get("is_final"))
        # is_final alone is a forced split (~60s of speech), not the end of a turn
        speech_final = is_final and bool(data.get("speech_final"))
        raw_words = data.get("words") or []
        if not transcript:
            if speech_final:
                self._end_speaking()
            return

        # smart_turn can suppress SpeechStarted, so Results may come first
        self._start_speaking()
        self._event_ch.send_nowait(self._transcript_event(data, transcript, is_final, raw_words))
        if speech_final:
            # the UtteranceEnd that follows is a no-op, so take the end time here
            self._end_speaking(speech_end_time=raw_words[-1].get("end") if raw_words else None)

    def _transcript_event(
        self, data: dict, transcript: str, is_final: bool, raw_words: list[dict]
    ) -> stt.SpeechEvent:
        # socket-relative timings, offset to stay linear across retries
        offset = self.start_time_offset
        words = [
            TimedString(
                text=w.get("word", ""),
                start_time=_word_time(w, "start", offset),
                end_time=_word_time(w, "end", offset),
                confidence=w.get("confidence", NOT_GIVEN),
            )
            for w in raw_words
        ]
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT
            if is_final
            else stt.SpeechEventType.INTERIM_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    language=_language(data.get("language"), self._opts.language),
                    text=transcript,
                    start_time=_word_time(raw_words[0], "start", offset) if raw_words else 0.0,
                    end_time=_word_time(raw_words[-1], "end", offset) if raw_words else 0.0,
                    confidence=data.get("confidence", 0.0) or 0.0,
                    words=words or None,
                )
            ],
        )

    def _on_error(self, data: dict) -> None:
        message = data.get("message", "unknown munsit error")
        if data.get("recoverable"):
            logger.warning("munsit error: %s", message, extra={"code": data.get("code")})
            return

        # auth / limit / balance failures close with 1008, so a bare Error is
        # worth a reconnect, except 4002 (bad parameters would be resent)
        code = data.get("code")
        raise APIError(f"munsit error {code}: {message}", body=data, retryable=code != 4002)

    def _start_speaking(self, *, speech_start_time: float | None = None) -> None:
        if self._speaking:
            return
        self._speaking = True
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.START_OF_SPEECH,
                speech_start_time=(
                    self.start_time + speech_start_time if speech_start_time is not None else None
                ),
            )
        )

    def _end_speaking(self, *, speech_end_time: float | None = None) -> None:
        if not self._speaking:
            return
        self._speaking = False
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.END_OF_SPEECH,
                speech_end_time=(
                    self.start_time + speech_end_time if speech_end_time is not None else None
                ),
            )
        )
