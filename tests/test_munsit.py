"""Plugin checks against a local fake of the Munsit API. No key, no network.

    pip install -e . && python tests/test_munsit.py
"""

import asyncio
import json
import time

import aiohttp
import numpy as np
from aiohttp import web

from livekit.agents import APIConnectOptions
from livekit.agents import stt as lkstt
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT
from livekit.plugins import munsit

T = lkstt.SpeechEventType
FAST_RETRY = dict(retry_interval=0.05, timeout=2.0)


async def serve(routes) -> tuple[web.AppRunner, str]:
    app = web.Application()
    app.add_routes(routes)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"


def listen_route(script: list[dict], connections: list[int]):
    """/listen that plays `script`, then closes on CloseStream."""

    async def handler(req):
        connections.append(1)
        ws = web.WebSocketResponse()
        await ws.prepare(req)
        for ev in script:
            await ws.send_str(json.dumps(ev))
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT and json.loads(msg.data)["type"] == "CloseStream":
                break
        await ws.close()
        return ws

    return web.get("/listen", handler)


async def test_stt_turn(http):
    script = [
        {"type": "SpeechStarted", "ts": 0.1},
        {"type": "Results", "transcript": "مرح", "is_final": False},
        {"type": "Results", "transcript": "مرحبا بك", "is_final": True, "speech_final": True,
         "words": [{"word": "مرحبا", "start": 0.2, "end": 0.6}, {"word": "بك", "start": 0.7, "end": 0.9}]},
        {"type": "UtteranceEnd", "last_word_end": 0.9},
        # an empty speech_final still closes a turn
        {"type": "SpeechStarted", "ts": 1.5},
        {"type": "Results", "transcript": "", "is_final": True, "speech_final": True},
        {"type": "Error", "code": 1, "message": "hiccup", "recoverable": True},
        # a null word timing must not drop the transcript
        {"type": "SpeechStarted", "ts": 2.5},
        {"type": "Results", "transcript": "نعم", "is_final": True, "speech_final": True,
         "words": [{"word": "نعم", "start": None, "end": None}]},
    ]
    runner, url = await serve([listen_route(script, [])])
    stream = munsit.STT(api_key="x", base_url=url, http_session=http).stream()

    events = []

    async def collect() -> None:
        async for ev in stream:
            events.append(ev)
            if len(events) == 9:
                return

    try:
        await asyncio.wait_for(collect(), 5)
    except asyncio.TimeoutError:
        raise AssertionError(f"stream stalled after {[e.type for e in events]}") from None
    assert [e.type for e in events] == [
        T.START_OF_SPEECH, T.INTERIM_TRANSCRIPT, T.FINAL_TRANSCRIPT, T.END_OF_SPEECH,
        T.START_OF_SPEECH, T.END_OF_SPEECH,
        T.START_OF_SPEECH, T.FINAL_TRANSCRIPT, T.END_OF_SPEECH,
    ], [e.type for e in events]
    assert events[7].alternatives[0].text == "نعم"

    final = events[2].alternatives[0]
    assert final.text == "مرحبا بك"
    got = [final.start_time, final.end_time] + [t for w in final.words for t in (w.start_time, w.end_time)]
    assert np.allclose(got, [0.2, 0.9, 0.2, 0.6, 0.7, 0.9], atol=0.01), got
    # speech_final carries the last word's end; UtteranceEnd after it is a no-op
    assert abs(events[3].speech_end_time - (stream.start_time + 0.9)) < 1e-6

    # CloseStream → server close; the keepalive's sleep must not hold the stream open
    t0 = time.monotonic()
    stream.end_input()
    async for _ in stream:
        pass
    assert time.monotonic() - t0 < 1.0, f"stream took {time.monotonic() - t0:.2f}s to close"
    await runner.cleanup()


async def test_stt_error_retry(http):
    for code, max_retry, want_connections in ((1, 1, 2), (4002, 3, 1)):
        connections: list[int] = []
        runner, url = await serve(
            [listen_route([{"type": "Error", "code": code, "message": "x", "recoverable": False}], connections)]
        )
        stream = munsit.STT(api_key="x", base_url=url, http_session=http).stream(
            conn_options=APIConnectOptions(max_retry=max_retry, **FAST_RETRY)
        )
        try:
            async for _ in stream:
                pass
        except Exception:
            pass
        await stream.aclose()
        assert len(connections) == want_connections, (code, len(connections))
        await runner.cleanup()


async def test_stt_connect_refused_is_retryable(http):
    errors = []
    stt = munsit.STT(api_key="x", base_url="http://127.0.0.1:1", http_session=http)
    stt.on("error", lambda e: errors.append(e))
    stream = stt.stream(conn_options=APIConnectOptions(max_retry=1, **FAST_RETRY))
    try:
        async for _ in stream:
            pass
    except Exception:
        pass
    await stream.aclose()
    assert [e.recoverable for e in errors] == [True, False], errors


async def test_stt_transcribe_confidence(http):
    async def handler(req):
        form = await req.post()
        assert "language" not in form
        return web.json_response({"data": {
            "transcriptionId": "t1", "transcription": "مرحبا بك", "duration": 1.0,
            "timestamps": [{"word": "مرحبا", "start": 0.1, "end": 0.5, "confidence": 0.8},
                           {"word": "بك", "start": 0.6, "end": 0.9, "confidence": 0.6},
                           {"word": "و", "start": 0.9, "end": 1.0, "confidence": None}],
        }}, status=201)

    runner, url = await serve([web.post("/audio/transcribe", handler)])
    stt = munsit.STT(api_key="x", base_url=url, http_session=http, streaming=False)
    from livekit import rtc

    ev = await stt.recognize(rtc.AudioFrame.create(16000, 1, 1600))
    alt = ev.alternatives[0]
    assert alt.text == "مرحبا بك" and abs(alt.confidence - 0.7) < 1e-9, alt
    await runner.cleanup()


TTS_PATH = "/text-to-speech/faseeh-v1-preview"
SENTENCES = [
    "The first sentence is the slow one here.",
    "The second sentence comes after it.",
    "The third sentence closes the reply now.",
]


def sentence_index(text: str) -> int:
    return next(i for i, s in enumerate(SENTENCES) if s in text)


def tone(i: int, seconds: float = 0.1) -> bytes:
    """A steady 24 kHz PCM16 tone whose value names the sentence: 1000, 2000, ..."""
    return np.full(int(24000 * seconds), (i + 1) * 1000, dtype=np.int16).tobytes()


def runs(samples: np.ndarray) -> list[int]:
    values = [int(v) for v in samples if v]
    return [v for j, v in enumerate(values) if j == 0 or v != values[j - 1]]


async def speak(http, url: str, **stream_kwargs) -> tuple[np.ndarray, Exception | None]:
    stream = munsit.TTS(api_key="x", voice_id="v", base_url=url, http_session=http).stream(
        **stream_kwargs
    )
    stream.push_text(" ".join(SENTENCES))
    stream.end_input()
    frames, error = [], None
    try:
        async for ev in stream:
            frames.append(np.frombuffer(ev.frame.data, dtype=np.int16))
    except Exception as e:
        error = e
    await stream.aclose()
    return (np.concatenate(frames) if frames else np.zeros(0, np.int16)), error


async def test_tts_streaming(http):
    sentences = SENTENCES
    spans: dict[int, list[float]] = {}
    bodies: list[dict] = []

    async def handler(req):
        body = await req.json()
        bodies.append(body)
        i = next(i for i, s in enumerate(sentences) if s in body["text"])
        spans[i] = [time.monotonic()]
        res = web.StreamResponse()
        await res.prepare(req)
        pcm = np.full(2400, (i + 1) * 1000, dtype=np.int16).tobytes()  # 0.1s per sentence
        await res.write(pcm[:2400])
        await asyncio.sleep(0.3 if i == 0 else 0.0)
        await res.write(pcm[2400:])
        spans[i].append(time.monotonic())
        return res

    runner, url = await serve([web.post("/text-to-speech/faseeh-v1-preview", handler)])
    munsit_tts = munsit.TTS(api_key="x", voice_id="v", base_url=url, http_session=http)
    assert munsit_tts.capabilities.streaming and munsit_tts.capabilities.aligned_transcript
    assert not munsit.TTS(api_key="x", voice_id="v", streaming=False).capabilities.streaming

    stream = munsit_tts.stream()
    stream.push_text(" ".join(sentences))
    stream.end_input()
    frames = [ev.frame async for ev in stream]
    samples = np.concatenate([np.frombuffer(f.data, dtype=np.int16) for f in frames])

    values = [int(v) for v in samples if v]
    runs = [v for j, v in enumerate(values) if j == 0 or v != values[j - 1]]
    assert runs == [1000, 2000, 3000], runs  # pushed strictly in sentence order
    # one request at a time: a stream holds at most one of the account's concurrent slots
    assert spans[0][1] <= spans[1][0] and spans[1][1] <= spans[2][0], spans
    assert all(b["streaming"] and b["text"] == b["text"].strip() for b in bodies), bodies

    # each sentence's text is stamped where its audio starts
    timed = [t for f in frames for t in f.userdata.get(USERDATA_TIMED_TRANSCRIPT, [])]
    assert [str(t).strip() for t in timed] == sentences, timed
    assert np.allclose([t.start_time for t in timed], [0.0, 0.1, 0.2], atol=1e-6), timed
    await runner.cleanup()


async def test_tts_pcm_stays_aligned(http):
    """A chunk ending mid-sample (each chunk is flushed as it arrives, livekit/agents#7391),
    then a stray trailing byte: no later sample may shift.
    """

    async def handler(req):
        i = sentence_index((await req.json())["text"])
        res = web.StreamResponse()
        await res.prepare(req)
        if i == 0:
            pcm = tone(0, 0.7)
            await res.write(pcm[:24001])  # 0.5s + half a sample
            await asyncio.sleep(0.05)  # keep the two writes as separate reads
            await res.write(pcm[24001:] + b"\x00")
        else:
            await res.write(tone(i))
        return res

    runner, url = await serve([web.post(TTS_PATH, handler)])
    samples, error = await speak(http, url)
    assert error is None, error
    assert runs(samples) == [1000, 2000, 3000], runs(samples)[:6]

    # synthesize() reads a PCM body the same way
    munsit_tts = munsit.TTS(api_key="x", voice_id="v", base_url=url, http_session=http)
    async with munsit_tts.synthesize(SENTENCES[0]) as chunked:
        frames = [np.frombuffer(ev.frame.data, dtype=np.int16) async for ev in chunked]
    assert runs(np.concatenate(frames)) == [1000], runs(np.concatenate(frames))[:4]
    await runner.cleanup()


async def test_tts_sentence_retry(http):
    cases = [
        # failing sentence, status, failures, audio heard, requests for that sentence
        (1, 500, 1, [1000, 2000, 3000], 2),  # audio already played: retried in place
        (1, 400, 1, [1000], 1),  # not retryable: the reply ends there
        (0, 500, 99, [], 3),  # nothing played yet: only the framework retries
    ]
    for failing, status, failures, want_runs, want_calls in cases:
        calls: list[int] = []

        async def handler(req, failing=failing, status=status, failures=failures, calls=calls):
            i = sentence_index((await req.json())["text"])
            if i == failing:
                calls.append(1)
                if len(calls) <= failures:
                    return web.json_response({"errorMessage": "boom"}, status=status)
            res = web.StreamResponse()
            await res.prepare(req)
            await res.write(tone(i))
            return res

        runner, url = await serve([web.post(TTS_PATH, handler)])
        samples, error = await speak(
            http, url, conn_options=APIConnectOptions(max_retry=2, retry_interval=0.05)
        )
        case = (failing, status)
        assert runs(samples) == want_runs, (case, runs(samples))
        assert len(calls) == want_calls, (case, len(calls))
        assert (error is None) == (want_runs == [1000, 2000, 3000]), (case, error)
        await runner.cleanup()


async def test_tts_releases_audio_as_it_arrives(http):
    """Playback must not run dry while received audio is still held back.

    Munsit's measured timing (223 ms in 20 ms, then nothing until 147 ms) used to run dry
    ~130 ms in: the tick and mute heard near each reply start. With a 100 ms burst and then
    nothing until 300 ms, playback may only run dry once all 100 ms has played.
    """
    cases = {
        "measured": ([(0, 138), (20, 85), (147, 226), (264, 258)], None),
        "slow": ([(0, 100), (300, 200), (420, 200)], 0.095),
    }
    for name, (chunks, dry_not_before) in cases.items():

        async def handler(req, chunks=chunks):
            res = web.StreamResponse()
            await res.prepare(req)
            start = time.monotonic()
            for at_ms, audio_ms in chunks:
                await asyncio.sleep(max(0.0, at_ms / 1000 - (time.monotonic() - start)))
                await res.write(np.full(24 * audio_ms, 1000, dtype=np.int16).tobytes())
            return res

        runner, url = await serve([web.post(TTS_PATH, handler)])
        munsit_tts = munsit.TTS(api_key="x", voice_id="v", base_url=url, http_session=http)
        for mode in ("stream", "synthesize"):
            if mode == "stream":
                stream = munsit_tts.stream()
                stream.push_text(SENTENCES[0])
                stream.end_input()
            else:
                stream = munsit_tts.synthesize(SENTENCES[0])
            async with stream:
                arrivals = [(time.monotonic(), ev.frame.duration) async for ev in stream]

            start = played_until = arrivals[0][0]
            dry_at = []
            for at, duration in arrivals:
                if at > played_until + 0.005:
                    dry_at.append(played_until - start)
                    played_until = at
                played_until += duration
            if dry_not_before is None:
                assert not dry_at, (name, mode, dry_at)
            else:
                assert all(t >= dry_not_before for t in dry_at), (name, mode, dry_at)
        await runner.cleanup()


def test_constructor_validation():
    # null config values mean the defaults
    stt = munsit.STT(
        api_key="x",
        listen=munsit.ListenOptions(sample_rate=None, endpointing_ms=None, smart_turn=None),
        transcribe=munsit.TranscribeOptions(return_timestamps=None),
    )
    assert (stt._opts.sample_rate, stt._opts.endpointing_ms, stt._opts.smart_turn) == (16000, 800, True)
    assert stt._opts.return_timestamps is True
    for build in (
        lambda: munsit.STT(api_key="x", listen=munsit.ListenOptions(endpointing_ms=50)),
        lambda: munsit.STT(api_key="x", listen=munsit.ListenOptions(sample_rate=44100)),
        lambda: munsit.STT(api_key="x").update_options(endpointing_ms=6000),
    ):
        try:
            build()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

async def main():
    test_constructor_validation()
    async with aiohttp.ClientSession() as http:
        for test in (
            test_stt_turn,
            test_stt_error_retry,
            test_stt_connect_refused_is_retryable,
            test_stt_transcribe_confidence,
            test_tts_streaming,
            test_tts_pcm_stays_aligned,
            test_tts_sentence_retry,
            test_tts_releases_audio_as_it_arrives,
        ):
            await test(http)
            print("ok", test.__name__)


if __name__ == "__main__":
    asyncio.run(main())
