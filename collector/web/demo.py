"""DemoCollector: a canned, time-paced stand-in for collector.runtime.Collector.

Used when the web app runs with DEMO=1 so the UI can be exercised without
credentials or network.  Implements the Collector interface from SPEC.md:
start / events / confirm / interrupt / status / outputs / download / console_url.

Pacing is controlled by DEMO_DELAY (seconds between events, default 0.4).
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Iterator

Event = dict


def _delay() -> float:
    try:
        return float(os.environ.get("DEMO_DELAY", "0.4"))
    except ValueError:
        return 0.4


def _event(**kw) -> Event:
    base = {
        "id": None, "type": None, "ts": None, "text": None, "tool": None,
        "input": None, "permission": None, "stop_reason": None, "result": None,
        "explanation": None, "iteration": None, "raw_type": None,
    }
    base.update(kw)
    if base["raw_type"] is None:
        base["raw_type"] = base["type"]
    if base["ts"] is None:
        base["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return base


class _Session:
    def __init__(self, sid: str, spec, title: str):
        self.sid = sid
        self.spec = spec
        self.title = title
        self.history: list[Event] = []
        self.cond = threading.Condition()
        self.done = False
        self.status = "running"
        self.outcome = None
        self.answers: dict[str, tuple[bool, str | None]] = {}
        self.answered = threading.Event()
        self.interrupted = threading.Event()
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self.thread: threading.Thread | None = None


class DemoCollector:
    """A fake Collector that replays a scripted run with pauses between events."""

    def __init__(self):
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    # -- interface ---------------------------------------------------------
    def start(self, spec, title: str | None = None) -> str:
        sid = "demo-" + uuid.uuid4().hex[:10]
        langs = ",".join(getattr(spec, "languages", []) or [])
        kind = getattr(spec, "kind", "collect")
        sess = _Session(sid, spec, title or f"{kind} · {langs}")
        with self._lock:
            self._sessions[sid] = sess
        sess.thread = threading.Thread(target=self._run, args=(sess,), daemon=True)
        sess.thread.start()
        return sid

    def events(self, session_id: str, *, include_history: bool = True) -> Iterator[Event]:
        sess = self._get(session_id)
        i = 0 if include_history else len(sess.history)
        while True:
            with sess.cond:
                while i >= len(sess.history) and not sess.done:
                    sess.cond.wait(timeout=1.0)
                if i < len(sess.history):
                    ev = sess.history[i]
                    i += 1
                elif sess.done:
                    return
                else:
                    continue
            yield ev

    def confirm(self, session_id: str, tool_event_id: str, allow: bool, message: str | None = None) -> None:
        sess = self._get(session_id)
        sess.answers[tool_event_id] = (bool(allow), message)
        sess.answered.set()

    def interrupt(self, session_id: str) -> None:
        sess = self._get(session_id)
        sess.interrupted.set()
        sess.answered.set()

    def status(self, session_id: str) -> dict:
        sess = self._get(session_id)
        return {"id": sess.sid, "status": sess.status, "outcome": sess.outcome,
                "usage": dict(sess.usage)}

    def outputs(self, session_id: str) -> list[dict]:
        self._get(session_id)
        lang = "ceb"
        langs = getattr(self._sessions[session_id].spec, "languages", None) or []
        if langs:
            lang = langs[0]
        return [
            {"id": "file_demo_report", "filename": "report.md", "size_bytes": 412},
            {"id": "file_demo_shard", "filename": f"text/{lang}/shard-00000.parquet", "size_bytes": 2048},
        ]

    def download(self, session_id: str, file_id: str, dest: Path) -> Path:
        self._get(session_id)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if file_id == "file_demo_report":
            dest.write_text("# halo-collector demo report\n\nThis is placeholder output "
                            "from the DEMO collector. No data was collected.\n")
        else:
            dest.write_bytes(b"PAR1" + b"\x00" * 60 + b"demo-placeholder" + b"PAR1")
        return dest

    def console_url(self, session_id: str) -> str:
        return f"https://console.anthropic.com/agents/sessions/{session_id}"

    # -- internals ---------------------------------------------------------
    def _get(self, sid: str) -> _Session:
        try:
            return self._sessions[sid]
        except KeyError:
            raise KeyError(f"unknown demo session {sid}") from None

    def _emit(self, sess: _Session, ev: Event) -> None:
        if ev.get("id") is None:
            ev["id"] = f"{sess.sid}-ev{len(sess.history) + 1}"
        with sess.cond:
            sess.history.append(ev)
            sess.cond.notify_all()

    def _finish(self, sess: _Session, status: str) -> None:
        sess.status = status
        with sess.cond:
            sess.done = True
            sess.cond.notify_all()

    def _run(self, sess: _Session) -> None:
        d = _delay()
        spec = sess.spec
        langs = getattr(spec, "languages", None) or ["ceb"]
        kind = getattr(spec, "kind", "collect")
        lang = langs[0]

        def step(ev: Event) -> bool:
            if sess.interrupted.is_set():
                return False
            self._emit(sess, ev)
            sess.usage["input_tokens"] += 900
            sess.usage["output_tokens"] += 120
            time.sleep(d)
            return not sess.interrupted.is_set()

        script = [
            _event(type="status", text="Session started (demo mode, no network)", raw_type="session.status_running"),
            _event(type="text", text=f"Starting a {kind} job for {', '.join(langs)}. "
                                    "I will read the pipeline docs first, then run the scraper.",
                   raw_type="agent.message"),
            _event(type="tool_call", tool="bash", input={"command": "cat /mnt/repo/docs/pipeline.md | head -40"},
                   permission="allow", raw_type="agent.tool_use"),
            _event(type="tool_result", tool="bash", result="# Pipeline\n...(40 lines)...",
                   raw_type="agent.tool_result"),
            _event(type="text", text=f"Running the scraper for `{lang}` with a cap of "
                                    f"{getattr(spec, 'max_docs', 300)} documents.",
                   raw_type="agent.message"),
        ]
        for ev in script:
            if not step(ev):
                return self._finish(sess, "terminated")

        # A paused tool call that waits for /confirm.
        ask_id = f"{sess.sid}-ask-1"
        ask = _event(id=ask_id, type="tool_ask", tool="bash", permission="ask",
                     input={"command": f"python /mnt/repo/scripts/scrape_web.py --lang {lang} "
                                       f"--max-docs {getattr(spec, 'max_docs', 300)} "
                                       f"--out /mnt/session/outputs/text/{lang}"},
                     text="The agent wants to run the scraper. Allow?",
                     raw_type="agent.tool_use")
        self._emit(sess, ask)
        while not sess.answered.wait(timeout=0.5):
            pass
        if sess.interrupted.is_set():
            return self._finish(sess, "terminated")
        allow, message = sess.answers.get(ask_id, (True, None))
        sess.answered.clear()

        if allow:
            tail = [
                _event(type="tool_result", tool="bash",
                       result=f"accepted=212 rejected=38 shards=1 -> text/{lang}/shard-00000.parquet",
                       raw_type="agent.tool_result"),
                _event(type="text", text="Scrape finished. Writing manifest.jsonl and report.md.",
                       raw_type="agent.message"),
                _event(type="tool_call", tool="bash",
                       input={"command": "ls -la /mnt/session/outputs"}, permission="allow",
                       raw_type="agent.tool_use"),
                _event(type="outcome", result="satisfied", iteration=1,
                       explanation="Shards, manifest and report are present; sample rows pass the "
                                   "language check. Push skipped (push=False).",
                       raw_type="outcome.graded"),
            ]
            final = "idle"
            sess.outcome = "satisfied"
        else:
            tail = [
                _event(type="text", text="The scraper call was denied"
                       + (f" (\"{message}\")" if message else "") + ". Stopping without outputs.",
                       raw_type="agent.message"),
                _event(type="outcome", result="unsatisfied", iteration=1,
                       explanation="No shards were produced because the scraper was not allowed to run.",
                       raw_type="outcome.graded"),
            ]
            final = "idle"
            sess.outcome = "unsatisfied"
        for ev in tail:
            if not step(ev):
                return self._finish(sess, "terminated")
        self._emit(sess, _event(type="usage", text=f"input {sess.usage['input_tokens']} · "
                                                    f"output {sess.usage['output_tokens']} tokens",
                                raw_type="session.usage"))
        self._emit(sess, _event(type="status", text="Session idle", stop_reason="end_turn",
                                raw_type="session.status_idle"))
        self._finish(sess, final)
