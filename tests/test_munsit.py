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
    ]
    runner, url = await serve([listen_route(script, [])])
    stream = munsit.STT(api_key="x", base_url=url, http_session=http).stream()

    events = []
    async for ev in stream:
        events.append(ev)
        if len(events) == 6:
            break
    assert [e.type for e in events] == [
        T.START_OF_SPEECH, T.INTERIM_TRANSCRIPT, T.FINAL_TRANSCRIPT, T.END_OF_SPEECH,
        T.START_OF_SPEECH, T.END_OF_SPEECH,
    ], [e.type for e in events]

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
                           {"word": "بك", "start": 0.6, "end": 0.9, "confidence": 0.6}],
        }}, status=201)

    runner, url = await serve([web.post("/audio/transcribe", handler)])
    stt = munsit.STT(api_key="x", base_url=url, http_session=http, streaming=False)
    from livekit import rtc

    ev = await stt.recognize(rtc.AudioFrame.create(16000, 1, 1600))
    alt = ev.alternatives[0]
    assert alt.text == "مرحبا بك" and abs(alt.confidence - 0.7) < 1e-9, alt
    await runner.cleanup()


async def test_tts_streaming(http):
    sentences = [
        "The first sentence is the slow one here.",
        "The second sentence comes after it.",
        "The third sentence closes the reply now.",
    ]
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


def test_constructor_validation():
    munsit.STT(api_key="x", encoding="linear16", num_channels=1)  # configs spelling out defaults
    for build in (
        lambda: munsit.STT(api_key="x", endpointing_ms=50),
        lambda: munsit.STT(api_key="x").update_options(endpointing_ms=6000),
        lambda: munsit.STT(api_key="x", encoding="mulaw"),
        lambda: munsit.STT(api_key="x", num_channels=2),
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
        ):
            await test(http)
            print("ok", test.__name__)


if __name__ == "__main__":
    asyncio.run(main())
