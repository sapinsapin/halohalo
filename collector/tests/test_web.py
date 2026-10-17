"""Tests for collector/web/app.py using an injected FakeCollector (no network)."""
from __future__ import annotations

import json
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from collector.web.app import create_app

LANGS = ["bcl", "ceb", "eng", "fil", "hil", "ilo", "pag", "pam", "tsg", "war"]


def ev(**kw):
    base = {"id": None, "type": None, "ts": None, "text": None, "tool": None, "input": None,
            "permission": None, "stop_reason": None, "result": None, "explanation": None,
            "iteration": None, "raw_type": None}
    base.update(kw)
    return base


SCRIPT = [
    ev(id="e1", type="text", text="hello", raw_type="agent.message"),
    ev(id="e2", type="tool_call", tool="bash", input={"command": "ls"}, raw_type="agent.tool_use"),
    ev(id="e3", type="tool_ask", tool="bash", input={"command": "rm -rf x"}, permission="ask",
       raw_type="agent.tool_use"),
    ev(id="e4", type="outcome", result="satisfied", iteration=1, explanation="fine",
       raw_type="outcome.graded"),
]


class FakeCollector:
    def __init__(self, script=SCRIPT):
        self.script = script
        self.started = []
        self.confirms = []
        self.interrupts = []
        self.downloads = []

    def start(self, spec, title=None):
        self.started.append((spec, title))
        return f"sess_{len(self.started)}"

    def events(self, session_id, *, include_history=True):
        yield from self.script

    def confirm(self, session_id, tool_event_id, allow, message=None):
        self.confirms.append((session_id, tool_event_id, allow, message))

    def interrupt(self, session_id):
        self.interrupts.append(session_id)

    def status(self, session_id):
        return {"id": session_id, "status": "idle", "outcome": "satisfied",
                "usage": {"input_tokens": 10, "output_tokens": 5}}

    def outputs(self, session_id):
        return [{"id": "f1", "filename": "report.md", "size_bytes": 12},
                {"id": "f2", "filename": "text/ceb/shard-00000.parquet", "size_bytes": 4}]

    def download(self, session_id, file_id, dest):
        self.downloads.append((session_id, file_id))
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"report-bytes" if file_id == "f1" else b"PAR1")
        return dest

    def console_url(self, session_id):
        return f"https://console.example/sessions/{session_id}"


@pytest.fixture
def fake():
    return FakeCollector()


@pytest.fixture
def client(fake, tmp_path):
    app = create_app(collector=fake, state_dir=tmp_path)
    with TestClient(app) as c:
        yield c


@contextmanager
def live_server(app):
    """Serve `app` with uvicorn in a thread and yield an httpx client bound to it.

    Starlette's TestClient buffers the whole response body before returning,
    so it cannot read an SSE stream incrementally; the tests that need to act
    while the stream is still open (answer a tool ask) use a real server.
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
            yield client
    finally:
        server.should_exit = True
        t.join(timeout=5)


def _start(client, **over):
    body = {"kind": "collect", "languages": ["ceb", "war"], "max_docs": 10}
    body.update(over)
    return client.post("/api/jobs", json=body)


def _sse(client, sid):
    """Return (list of parsed data events, saw_end)."""
    data, saw_end = [], False
    with client.stream("GET", f"/api/jobs/{sid}/events") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        pending_event = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                pending_event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                payload = line.split(":", 1)[1].strip()
                if pending_event == "end":
                    saw_end = True
                    break
                data.append(json.loads(payload))
                pending_event = None
    return data, saw_end


def test_languages(client):
    r = client.get("/api/languages")
    assert r.status_code == 200
    assert r.json() == LANGS


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "halo-collector" in r.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_post_job_happy(client, fake, tmp_path):
    r = _start(client, push=True, notes="hi")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["session_id"] == "sess_1"
    assert j["console_url"].endswith("sess_1")
    assert "collect" in j["title"] and "ceb,war" in j["title"]
    spec, title = fake.started[0]
    assert spec.kind == "collect" and spec.languages == ["ceb", "war"]
    assert spec.max_docs == 10 and spec.push is True and spec.notes == "hi"
    jobs = client.get("/api/jobs").json()
    assert len(jobs) == 1 and jobs[0]["session_id"] == "sess_1"
    assert jobs[0]["kind"] == "collect" and jobs[0]["languages"] == ["ceb", "war"]
    persisted = json.loads((tmp_path / "jobs.json").read_text())
    assert persisted["jobs"][0]["session_id"] == "sess_1"


@pytest.mark.parametrize("over", [
    {"kind": "nope"},
    {"languages": []},
    {"languages": ["xx"]},
    {"languages": ["ceb", "ceb"]},
    {"max_docs": 0},
    {"max_sentences": -3},
    {"max_docs": "lots"},
])
def test_post_job_422(client, fake, over):
    r = _start(client, **over)
    assert r.status_code == 422, r.text
    assert fake.started == []


def test_registry_reload(fake, tmp_path):
    app = create_app(collector=fake, state_dir=tmp_path)
    with TestClient(app) as c:
        assert _start(c).status_code == 200
    app2 = create_app(collector=fake, state_dir=tmp_path)
    with TestClient(app2) as c:
        assert [j["session_id"] for j in c.get("/api/jobs").json()] == ["sess_1"]


def test_sse_streams_and_ends(client):
    sid = _start(client).json()["session_id"]
    data, saw_end = _sse(client, sid)
    assert saw_end
    assert [d["id"] for d in data] == ["e1", "e2", "e3", "e4"]
    assert data[2]["type"] == "tool_ask"
    assert client.get("/api/jobs/nope/events").status_code == 404


def test_pending_asks_after_tool_ask_streamed(fake, tmp_path):
    gate = threading.Event()

    class Blocking(FakeCollector):
        def events(self, session_id, *, include_history=True):
            yield SCRIPT[0]
            yield SCRIPT[2]  # tool_ask
            gate.wait(timeout=5)
            yield SCRIPT[3]

    col = Blocking()
    app = create_app(collector=col, state_dir=tmp_path)
    with live_server(app) as client:
        sid = _start(client).json()["session_id"]
        with client.stream("GET", f"/api/jobs/{sid}/events") as r:
            lines = r.iter_lines()
            got = []
            for line in lines:
                if line.startswith("data:"):
                    got.append(json.loads(line.split(":", 1)[1]))
                if len(got) == 2:
                    break
            assert got[1]["type"] == "tool_ask"
            # Status now reports the pending ask so the UI can re-render it after a reload.
            st = client.get(f"/api/jobs/{sid}/status").json()
            assert [a["id"] for a in st["pending_asks"]] == ["e3"]
            assert st["status"] == "idle" and st["usage"]["input_tokens"] == 10
            assert st["job"]["session_id"] == sid
            # Confirm forwards to the collector and clears the pending ask.
            c = client.post(f"/api/jobs/{sid}/confirm",
                            json={"tool_event_id": "e3", "allow": True, "message": "go"})
            assert c.status_code == 200 and c.json() == {"ok": True}
            assert col.confirms == [(sid, "e3", True, "go")]
            assert client.get(f"/api/jobs/{sid}/status").json()["pending_asks"] == []
            gate.set()
            rest = list(lines)
            assert any(l.startswith("event: end") for l in rest)


def test_confirm_and_interrupt(client, fake):
    sid = _start(client).json()["session_id"]
    r = client.post(f"/api/jobs/{sid}/confirm", json={"tool_event_id": "e3", "allow": False})
    assert r.status_code == 200
    assert fake.confirms == [(sid, "e3", False, None)]
    assert client.post(f"/api/jobs/{sid}/confirm", json={"allow": True}).status_code == 422
    assert client.post(f"/api/jobs/{sid}/interrupt").status_code == 200
    assert fake.interrupts == [sid]
    assert client.post("/api/jobs/missing/interrupt").status_code == 404


def test_outputs_and_download(client, fake):
    sid = _start(client).json()["session_id"]
    files = client.get(f"/api/jobs/{sid}/outputs").json()
    assert [f["filename"] for f in files] == ["report.md", "text/ceb/shard-00000.parquet"]
    r = client.get(f"/api/jobs/{sid}/outputs/f1")
    assert r.status_code == 200
    assert r.content == b"report-bytes"
    assert "report.md" in r.headers["content-disposition"]
    assert fake.downloads == [(sid, "f1")]
    r = client.get(f"/api/jobs/{sid}/outputs/f2")
    assert r.content == b"PAR1"
    assert "shard-00000.parquet" in r.headers["content-disposition"]


def test_demo_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO", "1")
    monkeypatch.setenv("DEMO_DELAY", "0")
    app = create_app(state_dir=tmp_path)  # collector built lazily -> DemoCollector
    with live_server(app) as client:
        r = _start(client, languages=["war"])
        assert r.status_code == 200, r.text
        sid = r.json()["session_id"]
        assert sid.startswith("demo-")
        from collector.web.demo import DemoCollector
        assert isinstance(app.state.collector, DemoCollector)

        got, saw_ask = [], False
        with client.stream("GET", f"/api/jobs/{sid}/events") as resp:
            lines = resp.iter_lines()
            for line in lines:
                if line.startswith("data:"):
                    got.append(json.loads(line.split(":", 1)[1]))
                    if got[-1]["type"] == "tool_ask":
                        saw_ask = True
                        break
            assert saw_ask and len(got) >= 3
            assert got[0]["type"] == "status" and got[1]["type"] == "text"
            ask_id = got[-1]["id"]
            # answer the ask so the demo script runs to completion, then drain to the end
            assert client.post(f"/api/jobs/{sid}/confirm",
                               json={"tool_event_id": ask_id, "allow": True}).status_code == 200
            rest = list(lines)
        assert any(l.startswith("event: end") for l in rest)
        types = [json.loads(l.split(":", 1)[1]).get("type") for l in rest if l.startswith("data:")]
        assert "outcome" in types and "usage" in types

        st = client.get(f"/api/jobs/{sid}/status").json()
        assert st["status"] == "idle" and st["outcome"] == "satisfied"
        files = client.get(f"/api/jobs/{sid}/outputs").json()
        assert [f["filename"] for f in files] == ["report.md", "text/war/shard-00000.parquet"]
        d = client.get(f"/api/jobs/{sid}/outputs/{files[0]['id']}")
        assert d.status_code == 200 and b"demo" in d.content


# -- hardening (security review) --------------------------------------------

def test_download_refuses_unlisted_file_id(client, fake):
    sid = _start(client).json()["session_id"]
    r = client.get(f"/api/jobs/{sid}/outputs/file_from_another_session")
    assert r.status_code == 404
    assert fake.downloads == []          # never fetched from the Files API


def test_download_confines_crafted_filenames(client, fake, tmp_path):
    fake.outputs = lambda sid: [{"id": "evil", "filename": "../../escape.txt", "size_bytes": 1},
                                {"id": "win", "filename": "text\\..\\escape2.txt", "size_bytes": 1}]
    sid = _start(client).json()["session_id"]
    for fid in ("evil", "win"):
        r = client.get(f"/api/jobs/{sid}/outputs/{fid}")
        assert r.status_code in (200, 400)
    escaped = [p for p in tmp_path.rglob("escape*.txt")]
    base = tmp_path / "downloads" / sid
    assert all(base in p.parents for p in escaped), escaped


def test_cross_origin_write_refused(client, fake):
    r = client.post("/api/jobs", content='{"kind":"collect","languages":["ceb"],"push":true}',
                    headers={"Origin": "https://evil.example", "Content-Type": "text/plain"})
    assert r.status_code in (403, 422)
    assert fake.started == []
    r = client.post("/api/jobs", json={"kind": "collect", "languages": ["ceb"]},
                    headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    assert fake.started == []
    r = client.post("/api/jobs", json={"kind": "collect", "languages": ["ceb"]},
                    headers={"Origin": "http://testserver"})
    assert r.status_code == 200


def test_foreign_host_refused(client):
    r = client.get("/api/languages", headers={"Host": "attacker.example"})
    assert r.status_code == 403


def test_error_bodies_are_generic(client, fake):
    fake.start = lambda spec, title=None: (_ for _ in ()).throw(RuntimeError("/secret/path token=abc"))
    r = _start(client)
    assert r.status_code == 502
    assert "abc" not in r.text and "/secret" not in r.text
