from __future__ import annotations

import asyncio
import base64
import json
import os
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
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given
from livekit.agents.voice.io import TimedString

from .log import logger
from .models import STTEncodings, STTModels
from .tts import AUTH_HEADER, BASE_URL, _raise_for_status

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
# without a heartbeat a half-open socket parks recv_task forever and the
# exception-driven reconnect below never runs
WS_HEARTBEAT = 30.0


def _check_endpointing(endpointing_ms: int) -> None:
    if not ENDPOINTING_RANGE[0] <= endpointing_ms <= ENDPOINTING_RANGE[1]:
        raise ValueError(
            f"munsit: endpointing_ms must be between {ENDPOINTING_RANGE[0]} and "
            f"{ENDPOINTING_RANGE[1]}, got {endpointing_ms}"
        )


# transcribe accepts up to 60 minutes of audio; a slow upload should not look like a hang
TRANSCRIBE_TOTAL_TIMEOUT = 300.0


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
        interim_results: Optional[bool] = True,
        encoding: Optional[STTEncodings | str] = ENCODING,
        sample_rate: Optional[int] = DEFAULT_SAMPLE_RATE,
        num_channels: Optional[int] = 1,
        endpointing_ms: Optional[int] = DEFAULT_ENDPOINTING_MS,
        smart_turn: Optional[bool] = True,
        hotwords: Optional[str] = None,
        correlation_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        return_confidence: Optional[bool] = False,
        return_timestamps: Optional[bool] = True,
        return_turns: Optional[bool] = False,
        return_gender: Optional[bool] = False,
        return_sentiment: Optional[bool] = False,
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
            interim_results: Emit partial transcripts. Streaming only.
            encoding: Must be "linear16". LiveKit hands the stream PCM16, so Munsit's
                mulaw / alaw would mislabel the audio; anything else raises.
            sample_rate: 8000 or 16000 Hz. Streaming only; audio is resampled to it.
            num_channels: Must be 1; the stream is mono.
            endpointing_ms: 100-5000 ms of silence that ends a turn. Streaming only.
            smart_turn: Semantic turn-completion model on top of the silence timer.
            hotwords: Comma-separated custom vocabulary. Ignored by "munsit-en-ar".
            correlation_id: Your own session id, echoed back in Metadata. Streaming only.
            metadata: JSON-serializable dict (<=2 KB), sent base64-encoded. Streaming only.
            return_confidence: Add per-word confidence. Transcribe only.
            return_timestamps: Word timings; on by default for "munsit". Transcribe only.
            return_turns: Add the turns array. Transcribe only.
            return_gender: Add gender analysis per turn. Transcribe only.
            return_sentiment: Add sentiment analysis per turn. Transcribe only.
            api_key: Munsit API key, or `MUNSIT_API_KEY` in the environment.
            base_url: API root; defaults to `MUNSIT_BASE_URL` or the global endpoint.
                The websocket url is derived from it.
            http_session: Session to reuse instead of the agent's shared one.
            ssl: Verify TLS certificates.

        Every argument accepts `None`, meaning the default above.
        """
        streaming = True if streaming is None else bool(streaming)
        interim_results = True if interim_results is None else bool(interim_results)
        sample_rate = DEFAULT_SAMPLE_RATE if sample_rate is None else int(sample_rate)
        endpointing_ms = (
            DEFAULT_ENDPOINTING_MS if endpointing_ms is None else int(endpointing_ms)
        )

        if sample_rate not in SAMPLE_RATES:
            raise ValueError(f"munsit: sample_rate must be one of {SAMPLE_RATES}, got {sample_rate}")
        _check_endpointing(endpointing_ms)
        # kept as arguments so configs that spell out the defaults still load; any
        # other value would make Munsit misread the audio, so fail loudly instead
        if (encoding or ENCODING) != ENCODING:
            raise ValueError(f"munsit: encoding must be {ENCODING!r}, got {encoding!r}")
        if (1 if num_channels is None else int(num_channels)) != 1:
            raise ValueError(f"munsit: num_channels must be 1, got {num_channels!r}")

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=streaming,
                interim_results=streaming and interim_results,
            )
        )

        munsit_api_key = (api_key if is_given(api_key) else None) or os.environ.get("MUNSIT_API_KEY")
        if not munsit_api_key:
            raise ValueError("munsit: API key required. Set MUNSIT_API_KEY or pass api_key.")

        url = base_url if is_given(base_url) and base_url else os.environ.get("MUNSIT_BASE_URL")

        self._opts = _STTOptions(
            model=model or DEFAULT_STT_MODEL,
            language=language or DEFAULT_LANGUAGE,
            api_key=munsit_api_key,
            base_url=(url or BASE_URL).rstrip("/"),
            ssl=ssl,
            sample_rate=sample_rate,
            interim_results=interim_results,
            endpointing_ms=endpointing_ms,
            smart_turn=True if smart_turn is None else bool(smart_turn),
            correlation_id=correlation_id,
            metadata=metadata,
            hotwords=hotwords,
            return_confidence=bool(return_confidence),
            return_timestamps=True if return_timestamps is None else bool(return_timestamps),
            return_turns=bool(return_turns),
            return_gender=bool(return_gender),
            return_sentiment=bool(return_sentiment),
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
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language
        if is_given(hotwords):
            self._opts.hotwords = hotwords
        if is_given(endpointing_ms):
            _check_endpointing(endpointing_ms)
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
                    language=language if is_given(language) else opts.language,
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
        for stream in list(self._streams):
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
        self._reconnect_event = asyncio.Event()

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        hotwords: NotGivenOr[str] = NOT_GIVEN,
        endpointing_ms: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language
        if is_given(hotwords):
            self._opts.hotwords = hotwords
        if is_given(endpointing_ms):
            _check_endpointing(endpointing_ms)
            self._opts.endpointing_ms = endpointing_ms

        self._reconnect_event.set()

    async def _run(self) -> None:
        closing_ws = False

        async def keepalive_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            # the socket closes with 1011 after 12s of silence. Kept out of the task
            # group: its sleep would hold the stream open for up to 5s after
            # CloseStream, and a dropped socket already surfaces in recv_task
            # (heartbeat covers the half-open case)
            try:
                while True:
                    await ws.send_str(SpeechStream._KEEPALIVE_MSG)
                    await asyncio.sleep(KEEPALIVE_INTERVAL)
            except (aiohttp.ClientError, ConnectionError):
                return

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            # 50ms per frame, inside the 20-200ms the docs recommend
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=self._opts.sample_rate // 20,
            )
            try:
                async for data in self._input_ch:
                    frames: list[rtc.AudioFrame] = []
                    if isinstance(data, rtc.AudioFrame):
                        frames.extend(audio_bstream.write(data.data.tobytes()))
                    elif isinstance(data, self._FlushSentinel):
                        frames.extend(audio_bstream.flush())

                    for frame in frames:
                        await ws.send_bytes(frame.data.tobytes())

                closing_ws = True
                await ws.send_str(SpeechStream._CLOSE_MSG)
            except (aiohttp.ClientError, ConnectionError) as e:
                if closing_ws or self._session.closed:
                    return
                raise APIConnectionError("munsit connection closed unexpectedly") from e

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws or self._session.closed:
                        return

                    code = ws.close_code or -1
                    raise APIStatusError(
                        message=f"munsit connection closed unexpectedly ({code})",
                        # 1008 is auth / session limit / no balance, 4002 invalid
                        # connection parameters — retrying cannot fix either
                        status_code=400 if code in (1008, 4002) else 500,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type == aiohttp.WSMsgType.ERROR:
                    if closing_ws or self._session.closed:
                        return
                    raise APIConnectionError("munsit connection lost") from ws.exception()

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected munsit message type %s", msg.type)
                    continue

                try:
                    self._process_stream_event(json.loads(msg.data))
                except APIError:
                    # a non-recoverable Error event; let _main_task decide on the retry
                    raise
                except Exception:
                    logger.exception("failed to process munsit message")

        while True:
            ws: aiohttp.ClientWebSocketResponse | None = None
            try:
                ws = await self._connect_ws()
                # server timestamps restart with every socket, including the
                # update_options reconnect that the base retry loop never sees
                self.start_time = time.time()
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                ]
                tasks_group = asyncio.gather(*tasks)
                keepalive = asyncio.create_task(keepalive_task(ws))
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())
                try:
                    done, _ = await asyncio.wait(
                        (tasks_group, wait_reconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, keepalive, wait_reconnect_task)
                    tasks_group.cancel()
                    tasks_group.exception()  # retrieve the exception
            finally:
                if ws is not None:
                    await ws.close()

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
            transcript = data.get("transcript") or ""
            is_final = bool(data.get("is_final"))
            # is_final without speech_final is a forced split mid-speech (~60s), not
            # the end of a turn — UtteranceEnd is the real signal
            speech_final = is_final and bool(data.get("speech_final"))
            if not transcript:
                if speech_final:
                    self._end_speaking()
                return

            # Munsit only sends Results once speech is under way, but an interim can
            # be the first thing we see when smart_turn suppresses SpeechStarted
            self._start_speaking()
            # word timings are socket-relative; the offset keeps them linear across
            # retries. A null timing must not raise: that would drop the transcript
            offset = self.start_time_offset
            raw_words = data.get("words") or []
            words = [
                TimedString(
                    text=w.get("word", ""),
                    start_time=(w.get("start") or 0.0) + offset,
                    end_time=(w.get("end") or 0.0) + offset,
                    confidence=w.get("confidence", NOT_GIVEN),
                )
                for w in raw_words
            ]
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT
                    if is_final
                    else stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(
                            language=data.get("language") or self._opts.language,
                            text=transcript,
                            start_time=words[0].start_time if words else 0.0,
                            end_time=words[-1].end_time if words else 0.0,
                            confidence=data.get("confidence", 0.0) or 0.0,
                            words=words or None,
                        )
                    ],
                )
            )
            if speech_final:
                # UtteranceEnd is a no-op once this ends the turn, so carry its end time
                self._end_speaking(speech_end_time=raw_words[-1].get("end") if raw_words else None)

        elif event_type == "UtteranceEnd":
            self._end_speaking(speech_end_time=data.get("last_word_end"))

        elif event_type == "Error":
            message = data.get("message", "unknown munsit error")
            if data.get("recoverable"):
                logger.warning("munsit error: %s", message, extra={"code": data.get("code")})
            else:
                # auth / balance / session limit arrive as close code 1008 (non-retryable
                # in recv_task), so a bare Error is worth a reconnect — except 4002,
                # invalid connection parameters, which a reconnect would resend;
                # max_retry bounds one that keeps coming back
                code = data.get("code")
                raise APIError(f"munsit error {code}: {message}", body=data, retryable=code != 4002)

        elif event_type in ("Metadata", "Gender", "Sentiment"):
            # Metadata carries session ids and closing billing; Gender/Sentiment arrive
            # after the turn's final transcript, so there is no event left to attach
            # them to — surfaced in the log for anyone who wants them
            logger.debug("munsit %s: %s", event_type, data)

        else:
            logger.warning("unexpected munsit event %s", event_type)

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
