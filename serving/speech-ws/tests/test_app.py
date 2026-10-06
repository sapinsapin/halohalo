"""
The whole server against the model-free engines: every HTTP route, both
sockets, auth and tickets, and the wire invariants devices depend on.
"""

import json

import numpy as np
import pytest

import protocol as P
import tickets

KEY = "test-serving-key"


def test_health_and_index(client):
    assert client.get("/health").status_code == 200
    hz = client.get("/healthz").json()
    assert hz["ready"] is True and hz["langs"] == ["fil", "ceb"] and hz["asr_langs"] == ["fil", "ceb"]
    assert "/tts" in client.get("/").text


def test_v1_routes_need_the_key(client, auth):
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/voices").status_code == 401
    m = client.get("/v1/models", headers=auth).json()
    assert m["data"][0]["id"] == "sapin-speech" and m["data"][0]["type"] == "speech"
    assert m["data"][0]["capabilities"]["tts"] == ["fil", "ceb"]
    v = client.get("/v1/voices", headers=auth).json()
    assert v["default_lang"] == "fil"
    assert v["languages"]["ceb"]["voices"][0]["id"] == "CEB_0200"
    assert "pag" not in v["languages"]


def test_tickets_mint_and_admit(client, auth):
    assert client.post("/v1/tickets").status_code == 401
    r = client.post("/v1/tickets", headers=auth, json={"scope": "tts", "ttl_s": 120})
    assert r.status_code == 200
    body = r.json()
    assert tickets.verify(KEY, body["ticket"], "tts")
    assert body["tts_path"].startswith("/tts?ticket=")
    assert client.post("/v1/tickets", headers=auth, json={"scope": "video"}).status_code == 400
    # a tts ticket opens /tts and not /asr
    with client.websocket_connect(body["tts_path"]) as ws:
        ws.send_text(json.dumps({"text": "Kumusta po."}))
        audio, term = _collect(ws)
        assert audio and json.loads(term) == {"type": "end"}
    with client.websocket_connect(body["asr_path"]) as ws:
        msg = json.loads(ws.receive_text())
        assert msg == {"type": "error", "message": P.ERR_UNAUTHORIZED}


def test_socket_refuses_without_credentials_and_accepts_key(client):
    with client.websocket_connect("/tts") as ws:
        assert json.loads(ws.receive_text())["message"] == P.ERR_UNAUTHORIZED
    with client.websocket_connect(f"/tts?key={KEY}") as ws:
        ws.send_text(json.dumps({"text": "Maayong buntag.", "lang": "Cebuano", "voice": "CEB_0288", "meta": True}))
        start = json.loads(ws.receive_text())
        assert start == {"type": "start", "sample_rate": 16000, "channels": 1,
                         "format": "pcm_s16le", "lang": "ceb", "voice": "CEB_0288", "segments": 1}
        audio, term = _collect(ws)
        assert term.encode() == P.END


def _collect(ws, max_frames=500):
    audio, term = [], None
    for _ in range(max_frames):
        msg = ws.receive()
        if msg.get("bytes") is not None:
            audio.append(msg["bytes"])
        elif msg.get("text") is not None:
            term = msg["text"]
            break
        else:
            break
    return audio, term


def test_tts_frames_are_the_contract(client):
    text = "Magandang umaga po sa inyong lahat. Salamat sa inyong pakikinig, hanggang sa muli."
    with client.websocket_connect(f"/tts?key={KEY}") as ws:
        ws.send_text(json.dumps({"text": text}))
        audio, term = _collect(ws)
        assert json.loads(term) == {"type": "end"}
        assert all(len(f) % 2 == 0 for f in audio)
        assert all(len(f) <= 3200 for f in audio)
        assert all(len(f) == 3200 for f in audio[:-1])        # only the last may be short
        pcm = np.frombuffer(b"".join(audio), dtype="<i2")
        # two segments: tone for every character plus the inter-segment gap
        assert pcm.size >= 16000 * 0.06 * (len(text) - 2) + 16000 * 0.12 - 1
        assert np.abs(pcm).max() > 2000
        # the socket is reusable after end
        ws.send_text(json.dumps({"text": "Uli."}))
        audio2, term2 = _collect(ws)
        assert audio2 and json.loads(term2) == {"type": "end"}


@pytest.mark.parametrize("payload,contains", [
    ("not json", P.ERR_BAD_JSON),
    ("{}", P.ERR_NO_TEXT),
    ('{"text":""}', P.ERR_EMPTY_TEXT),
    ('{"text":"' + "A" * 501 + '"}', P.ERR_TOO_LONG),
    ('{"text":"hoy","lang":"klingon"}', P.ERR_UNKNOWN_LANG),
])
def test_tts_errors_are_terminal_and_alone(client, payload, contains):
    with client.websocket_connect(f"/tts?key={KEY}") as ws:
        ws.send_text(payload)
        audio, term = _collect(ws)
        assert not audio
        obj = json.loads(term)
        assert obj["type"] == "error" and contains in obj["message"]
        if contains == P.ERR_UNKNOWN_LANG:
            assert "fil, ceb" in obj["message"]


def test_text_sent_with_binary_opcode_is_accepted(client):
    with client.websocket_connect(f"/tts?key={KEY}") as ws:
        ws.send_bytes(json.dumps({"text": "Hoy."}).encode())
        audio, term = _collect(ws)
        assert audio and json.loads(term) == {"type": "end"}


def _pcm(seconds, amp=0.3):
    t = np.arange(int(16000 * seconds)) / 16000
    return (np.sin(2 * np.pi * 220 * t) * amp * 32767).astype("<i2").tobytes()


def test_asr_socket_streams_partials_and_finals(client):
    with client.websocket_connect(f"/asr?key={KEY}") as ws:
        ws.send_text(json.dumps({"type": "start", "lang": "ceb"}))
        stream = _pcm(0.3, 0.0) + _pcm(2.2) + _pcm(1.0, 0.0) + _pcm(0.8) + _pcm(0.9, 0.0)
        for i in range(0, len(stream), 3200):
            ws.send_bytes(stream[i:i + 3200])
        ws.send_text(json.dumps({"type": "stop"}))
        frames = []
        for _ in range(50):
            msg = json.loads(ws.receive_text())
            frames.append(msg)
            if msg["type"] in ("end", "error"):
                break
        assert frames[-1] == {"type": "end"}
        finals = [f for f in frames if f["type"] == "final"]
        assert len(finals) == 2
        assert all(f["text"].startswith("[ceb]") for f in finals)
        assert finals[0]["end"] > finals[0]["start"] >= 0
        assert finals[1]["start"] > finals[0]["end"]
        partials = [f for f in frames if f["type"] == "partial"]
        assert partials and partials[0]["text"].startswith("[ceb]")


def test_asr_socket_rejects_bad_frames(client):
    with client.websocket_connect(f"/asr?key={KEY}") as ws:
        ws.send_bytes(b"\x00\x01\x02")
        assert json.loads(ws.receive_text()) == {"type": "error", "message": P.ERR_ODD_FRAME}
    with client.websocket_connect(f"/asr?key={KEY}") as ws:
        ws.send_text(json.dumps({"type": "start", "lang": "elvish"}))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error" and P.ERR_UNKNOWN_LANG in msg["message"]
    with client.websocket_connect(f"/asr?key={KEY}") as ws:
        ws.send_text(json.dumps({"type": "stop"}))             # nothing said: still a clean end
        assert json.loads(ws.receive_text()) == {"type": "end"}


def test_http_speech_pcm_and_wav(client, auth):
    r = client.post("/v1/audio/speech", json={"input": "Kumusta po.", "voice": "female"})
    assert r.status_code == 401
    r = client.post("/v1/audio/speech", headers=auth, json={"input": "Kumusta po. Maayo.", "language": "fil"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/pcm")
    assert r.headers["x-sample-rate"] == "16000" and r.headers["x-lang"] == "fil"
    assert len(r.content) % 2 == 0 and len(r.content) > 16000
    r = client.post("/v1/audio/speech", headers=auth,
                    json={"input": "Maayong buntag.", "language": "ceb", "response_format": "wav"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF" and float(r.headers["x-audio-seconds"]) > 0.5
    assert client.post("/v1/audio/speech", headers=auth, json={"input": ""}).status_code == 400
    assert client.post("/v1/audio/speech", headers=auth,
                       json={"input": "x", "response_format": "mp3"}).status_code == 400
    assert client.post("/v1/audio/speech", headers=auth,
                       json={"input": "x", "language": "pag"}).status_code == 400


def test_http_transcriptions(client, auth):
    import io
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.frombuffer(_pcm(1.5), dtype="<i2"), 16000, format="WAV", subtype="PCM_16")
    files = {"file": ("clip.wav", buf.getvalue(), "audio/wav")}
    assert client.post("/v1/audio/transcriptions", files=files).status_code == 401
    r = client.post("/v1/audio/transcriptions", headers=auth, files=files, data={"language": "ceb"})
    assert r.status_code == 200 and r.json()["text"] == "[ceb] 1.5s"
    r = client.post("/v1/audio/transcriptions", headers=auth, files=files,
                    data={"language": "fil", "response_format": "verbose_json"})
    assert r.json()["duration"] == 1.5 and r.json()["language"] == "fil"
    r = client.post("/v1/audio/transcriptions", headers=auth, files=files, data={"response_format": "text"})
    assert r.text == "[fil] 1.5s"
    # raw PCM declared as such, at 48 kHz? no: raw is taken as 16 kHz; a bad file is a 400
    bad = {"file": ("x.bin", b"not audio at all", "application/octet-stream")}
    assert client.post("/v1/audio/transcriptions", headers=auth, files=bad).status_code == 400
