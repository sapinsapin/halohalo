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


class _DemoStream:
    """Stand-in for the SDK stream object handed to `on_open`; only close() matters."""

    def __init__(self):
        self.closed = False

    def close(self) -> None:
        self.closed = True


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

    def events(self, session_id: str, *, include_history: bool = True, on_open=None) -> Iterator[Event]:
        sess = self._get(session_id)
        handle = _DemoStream()
        if on_open is not None:
            on_open(handle)   # like the SDK stream: close() makes this iterator end
        i = 0 if include_history else len(sess.history)
        while not handle.closed:
            with sess.cond:
                while i >= len(sess.history) and not sess.done and not handle.closed:
                    sess.cond.wait(timeout=0.5)
                if handle.closed:
                    return
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
        sess = self._get(session_id)
        langs = getattr(sess.spec, "languages", None) or ["ceb"]
        kind = getattr(sess.spec, "kind", "collect")
        per_lang = {"collect": "text/{lang}/shard-00000.parquet", "validate": "validated/{lang}.jsonl",
                    "instruct": "instruct/{lang}.jsonl", "voice_discover": "voice/{lang}.jsonl"}
        files = [{"id": "file_demo_report", "filename": "report.md", "size_bytes": 412}]
        for i, lang in enumerate(langs):
            files.append({"id": f"file_demo_shard_{i}", "filename": per_lang.get(kind, per_lang["collect"]).format(lang=lang),
                          "size_bytes": 2048})
        return files

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

        def step(ev: Event) -> bool:
            if sess.interrupted.is_set():
                return False
            self._emit(sess, ev)
            sess.usage["input_tokens"] += 900
            sess.usage["output_tokens"] += 120
            time.sleep(d)
            return not sess.interrupted.is_set()

        # One narrative per job kind, covering every requested language.
        cap = getattr(spec, "max_sentences", 500) if kind != "collect" else getattr(spec, "max_docs", 300)
        lang_list = ",".join(langs)
        if kind == "collect":
            cmd = (f"python /mnt/repo/scrape_web.py --langs {lang_list} --max-docs {cap} "
                   f"--no-glotlid --out /mnt/session/outputs")
            plan = "run the seeded scrape, then gate the pages by language ID"
            done = " ".join(f"{l}: accepted={212 - 17 * i} rejected={38 + 5 * i} shards=1" for i, l in enumerate(langs))
            wrote = "text/{lang}/shard-00000.parquet, manifest.jsonl and report.md"
            verdict = "Shards, manifest and report are present; sample rows pass the language check."
        elif kind == "validate":
            cmd = (f"python -c 'import datasets' && python /mnt/repo/collector_demo_validate.py "
                   f"--langs {lang_list} --max {cap}")
            plan = f"load up to {cap} sentences per language and judge each one"
            done = " ".join(f"{l}: ok={int(cap * 0.8)} reject={cap - int(cap * 0.8)}" for l in langs)
            wrote = "validated/{lang}.jsonl and report.md"
            verdict = "Every row has verdict, reason, confidence in [0,1] and validator=model."
        elif kind == "instruct":
            cmd = f"python /mnt/repo/collector_demo_instruct.py --langs {lang_list} --max {cap}"
            plan = f"write up to {cap} instruction pairs per language from validated rows"
            done = " ".join(f"{l}: pairs={cap} task_types=6" for l in langs)
            wrote = "instruct/{lang}.jsonl and report.md"
            verdict = "Pairs carry instruction, output, task_type, source_row and license."
        else:
            cmd = f"python /mnt/repo/scrape_web.py --voice --langs {lang_list} --no-text"
            plan = "search YouTube with the language seeds and keep CC-licensed videos only"
            done = " ".join(f"{l}: cc_videos={14 + i}" for i, l in enumerate(langs))
            wrote = "voice/{lang}.jsonl and report.md"
            verdict = "Every row has video_id, url, license (CC) and duration_s."

        script = [
            _event(type="status", text="Session started (demo mode, no network)", raw_type="session.status_running"),
            _event(type="text", text=f"Starting a {kind} job for {lang_list}. "
                                    f"I will read the pipeline docs first, then {plan}.",
                   raw_type="agent.message"),
            _event(type="tool_call", tool="bash", input={"command": "cat /mnt/repo/docs/web_scrape_pipeline.md | head -40"},
                   permission="allow", raw_type="agent.tool_use"),
            _event(type="tool_result", tool="bash", result="# Seeded web scrape\n...(40 lines)...",
                   raw_type="agent.tool_result"),
            _event(type="text", text=f"Working on {lang_list} with a cap of {cap} per language.",
                   raw_type="agent.message"),
        ]
        for ev in script:
            if not step(ev):
                return self._finish(sess, "terminated")

        # A paused tool call that waits for /confirm.
        ask_id = f"{sess.sid}-ask-1"
        ask = _event(id=ask_id, type="tool_ask", tool="bash", permission="ask",
                     input={"command": cmd},
                     text="The agent wants to run the pipeline step. Allow?",
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
                _event(type="tool_result", tool="bash", result=done, raw_type="agent.tool_result"),
                _event(type="text", text=f"Step finished. Writing {wrote}.", raw_type="agent.message"),
                _event(type="tool_call", tool="bash",
                       input={"command": "ls -la /mnt/session/outputs"}, permission="allow",
                       raw_type="agent.tool_use"),
                _event(type="outcome", result="satisfied", iteration=1,
                       explanation=verdict + " Push skipped (push=False).",
                       raw_type="outcome.graded"),
            ]
            final = "idle"
            sess.outcome = "satisfied"
        else:
            tail = [
                _event(type="text", text="The pipeline step was denied"
                       + (f" (\"{message}\")" if message else "") + ". Stopping without outputs.",
                       raw_type="agent.message"),
                _event(type="outcome", result="unsatisfied", iteration=1,
                       explanation="No outputs were produced because the pipeline step was not allowed to run.",
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
