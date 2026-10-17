"""FastAPI app for halo-collector (see collector/SPEC.md, "collector/web/app.py").

Routes
  GET  /                              static index
  GET  /api/languages                 LANGUAGES
  GET  /api/jobs                      job registry (persisted to jobs.json)
  POST /api/jobs                      start a job -> {session_id, title, console_url}
  GET  /api/jobs/{sid}/events         SSE stream of normalized Events
  POST /api/jobs/{sid}/confirm        answer a paused tool call
  POST /api/jobs/{sid}/interrupt
  GET  /api/jobs/{sid}/status         {..., pending_asks: [...]}
  GET  /api/jobs/{sid}/outputs        [{id, filename, size_bytes}]
  GET  /api/jobs/{sid}/outputs/{fid}  file download

The Collector is created lazily on the first request that needs it, unless
`app.state.collector` was injected (tests) or DEMO=1 (DemoCollector).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

STATIC_DIR = Path(__file__).parent / "static"
KINDS = ("collect", "validate", "instruct", "voice_discover")
FALLBACK_LANGUAGES = ["bcl", "ceb", "eng", "fil", "hil", "ilo", "pag", "pam", "tsg", "war"]
log = logging.getLogger(__name__)
HEARTBEAT_SECONDS = 15.0
MAX_STREAMS_PER_JOB = 4   # open SSE connections per job; each one is a thread and an SDK stream


def _languages() -> list[str]:
    try:
        from collector.jobs import LANGUAGES  # worker A's module
        return list(LANGUAGES)
    except Exception:  # pragma: no cover - module may not exist yet
        return list(FALLBACK_LANGUAGES)


# --- request models ----------------------------------------------------------

class JobIn(BaseModel):
    kind: Literal["collect", "validate", "instruct", "voice_discover"]
    languages: list[str] = Field(min_length=1)
    max_docs: int = Field(default=300, gt=0)
    max_sentences: int = Field(default=500, gt=0)
    push: bool = False
    notes: str = ""
    max_iterations: int = Field(default=4, gt=0)


class ConfirmIn(BaseModel):
    tool_event_id: str = Field(min_length=1)
    allow: bool
    message: Optional[str] = None


# --- job registry ------------------------------------------------------------

class JobRegistry:
    """In-memory dict of jobs, mirrored to <state_dir>/jobs.json."""

    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)
        self.path = self.state_dir / "jobs.json"
        self.jobs: dict[str, dict] = {}
        self.pending_asks: dict[str, dict[str, dict]] = {}
        self.lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text())
            for job in data.get("jobs", []):
                self.jobs[job["session_id"]] = job
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def _save(self) -> None:
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"jobs": list(self.jobs.values())}, indent=1))
            tmp.replace(self.path)
        except OSError:
            pass

    def add(self, job: dict) -> None:
        with self.lock:
            self.jobs[job["session_id"]] = job
            self._save()

    def get(self, sid: str) -> dict | None:
        return self.jobs.get(sid)

    def list(self) -> list[dict]:
        return sorted(self.jobs.values(), key=lambda j: j.get("created_at", ""), reverse=True)

    def set_status(self, sid: str, status: str) -> None:
        with self.lock:
            job = self.jobs.get(sid)
            if job and job.get("last_status") != status:
                job["last_status"] = status
                self._save()

    def note_event(self, sid: str, ev: dict) -> None:
        """Track pending tool asks and coarse status from a streamed event."""
        if ev.get("type") == "tool_ask" and ev.get("id"):
            self.pending_asks.setdefault(sid, {})[ev["id"]] = ev
            self.set_status(sid, "needs_decision")

    def resolve_ask(self, sid: str, tool_event_id: str) -> None:
        asks = self.pending_asks.get(sid)
        if asks:
            asks.pop(tool_event_id, None)
            if not asks:
                self.set_status(sid, "running")

    def asks(self, sid: str) -> list[dict]:
        return list(self.pending_asks.get(sid, {}).values())


# --- app factory -------------------------------------------------------------

def _make_collector():
    if os.environ.get("DEMO") == "1":
        from collector.web.demo import DemoCollector
        return DemoCollector()
    from collector.config import load_ids
    from collector.runtime import Collector
    ids = load_ids()
    return Collector(
        agent_id=ids.get("agent_id"),
        environment_id=ids.get("environment_id"),
        vault_ids=[ids["vault_id"]] if ids.get("vault_id") else None,
        repo_url=os.environ.get("HALO_REPO_URL", "https://github.com/sapinsapin/halohalo"),
        repo_branch=os.environ.get("HALO_REPO_BRANCH", "main"),
        repo_token=os.environ.get("GITHUB_TOKEN"),
    )


def get_collector(request: Request):
    app = request.app
    col = getattr(app.state, "collector", None)
    if col is None:
        with app.state.collector_lock:
            col = getattr(app.state, "collector", None)
            if col is None:
                try:
                    col = _make_collector()
                except Exception as e:  # missing ids / modules
                    raise HTTPException(503, f"collector unavailable: {e}") from e
                app.state.collector = col
    return col


def _registry(request: Request) -> JobRegistry:
    return request.app.state.registry


def _job_or_404(request: Request, sid: str) -> dict:
    job = _registry(request).get(sid)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


def _build_spec(body: JobIn):
    """Build a JobSpec via collector.jobs when available, else a plain object."""
    kw = dict(kind=body.kind, languages=list(body.languages), max_docs=body.max_docs,
              max_sentences=body.max_sentences, push=body.push, notes=body.notes,
              max_iterations=body.max_iterations)
    try:
        from collector import jobs as jobs_mod
    except Exception:
        jobs_mod = None
    if jobs_mod is not None:
        spec = jobs_mod.JobSpec(**kw)
        validate = getattr(jobs_mod, "validate_spec", None)
        if validate is not None:
            try:
                validate(spec)
            except ValueError as e:
                raise HTTPException(422, str(e)) from e
        title_fn = getattr(jobs_mod, "job_title", None)
        title = title_fn(spec) if title_fn else None
    else:
        from types import SimpleNamespace
        spec = SimpleNamespace(**kw)
        title = None
    if not title:
        title = f"{body.kind} · {','.join(body.languages)} · {time.strftime('%Y-%m-%d')}"
    return spec, title


def create_app(collector=None, state_dir: str | os.PathLike | None = None) -> FastAPI:
    app = FastAPI(title="halo-collector")
    app.state.collector = collector
    app.state.collector_lock = threading.Lock()
    app.state.open_streams = {}
    app.state.registry = JobRegistry(Path(state_dir or os.environ.get("COLLECTOR_STATE_DIR", "./collector_state")))

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/languages")
    def languages():
        return _languages()

    @app.middleware("http")
    async def same_origin_guard(request: Request, call_next):
        """Local tool, no auth: refuse requests that don't come from this host.

        Blocks cross-site POSTs from a visited web page and DNS-rebinding reads.
        Host must be loopback (or COLLECTOR_ALLOWED_HOSTS); a non-GET request
        carrying an Origin header must match the Host."""
        host = (request.headers.get("host") or "").split(":")[0].lower()
        allowed = {"127.0.0.1", "localhost", "::1", "testserver", "testclient"}
        allowed |= {h.strip().lower() for h in os.environ.get("COLLECTOR_ALLOWED_HOSTS", "").split(",") if h.strip()}
        if host not in allowed:
            return JSONResponse(status_code=403, content={"detail": "host not allowed"})
        origin = request.headers.get("origin")
        if origin and request.method not in ("GET", "HEAD", "OPTIONS"):
            from urllib.parse import urlsplit
            if (urlsplit(origin).hostname or "").lower() != host:
                return JSONResponse(status_code=403, content={"detail": "cross-origin request refused"})
        return await call_next(request)

    @app.get("/api/jobs")
    def list_jobs(request: Request):
        return _registry(request).list()

    @app.post("/api/jobs")
    async def start_job(request: Request, body: JobIn):
        allowed = set(_languages())
        bad = [l for l in body.languages if l not in allowed]
        if bad:
            raise HTTPException(422, f"unknown language codes: {', '.join(bad)}")
        if len(set(body.languages)) != len(body.languages):
            raise HTTPException(422, "duplicate language codes")
        spec, title = _build_spec(body)
        col = get_collector(request)
        try:
            sid = await asyncio.to_thread(col.start, spec, title)
            console_url = col.console_url(sid)
        except HTTPException:
            raise
        except Exception as e:
            log.exception("could not start job"); raise HTTPException(502, "could not start job") from e
        job = {
            "session_id": sid, "title": title, "kind": body.kind, "languages": body.languages,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "console_url": console_url, "last_status": "running",
            "push": body.push, "max_docs": body.max_docs, "max_sentences": body.max_sentences,
        }
        _registry(request).add(job)
        return {"session_id": sid, "title": title, "console_url": console_url}

    @app.get("/api/jobs/{sid}/events")
    async def job_events(request: Request, sid: str):
        _job_or_404(request, sid)
        col = get_collector(request)
        reg = _registry(request)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        END = object()
        stop = threading.Event()
        streams = request.app.state.open_streams
        with request.app.state.collector_lock:
            if streams.get(sid, 0) >= MAX_STREAMS_PER_JOB:
                raise HTTPException(429, "too many open event streams for this job")
            streams[sid] = streams.get(sid, 0) + 1

        def put(item):
            fut = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            fut.result(timeout=60)

        def pump():
            try:
                for ev in col.events(sid, include_history=True):
                    if stop.is_set():
                        return
                    put(ev)
            except Exception as e:  # surface the error to the client, then end
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "text": str(e), "raw_type": "web.error"})
                except RuntimeError:
                    return
            try:
                loop.call_soon_threadsafe(queue.put_nowait, END)
            except RuntimeError:
                pass

        threading.Thread(target=pump, daemon=True, name=f"sse-{sid}").start()

        async def gen():
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"
                        continue
                    if item is END:
                        final = _final_status(col, sid)
                        reg.pending_asks.pop(sid, None)
                        reg.set_status(sid, final)
                        yield "event: end\ndata: {}\n\n"
                        return
                    reg.note_event(sid, item)
                    yield "data: " + json.dumps(item, default=str) + "\n\n"
            finally:
                stop.set()   # client went away: let the pump thread exit at its next event
                with request.app.state.collector_lock:
                    streams[sid] = max(0, streams.get(sid, 1) - 1)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/jobs/{sid}/confirm")
    async def confirm(request: Request, sid: str, body: ConfirmIn):
        _job_or_404(request, sid)
        col = get_collector(request)
        try:
            await asyncio.to_thread(col.confirm, sid, body.tool_event_id, body.allow, body.message)
        except Exception as e:
            raise HTTPException(502, f"confirm failed: {e}") from e
        _registry(request).resolve_ask(sid, body.tool_event_id)
        return {"ok": True}

    @app.post("/api/jobs/{sid}/interrupt")
    async def interrupt(request: Request, sid: str):
        _job_or_404(request, sid)
        col = get_collector(request)
        try:
            await asyncio.to_thread(col.interrupt, sid)
        except Exception as e:
            raise HTTPException(502, f"interrupt failed: {e}") from e
        _registry(request).set_status(sid, "interrupting")
        return {"ok": True}

    @app.get("/api/jobs/{sid}/status")
    async def status(request: Request, sid: str):
        job = _job_or_404(request, sid)
        col = get_collector(request)
        try:
            st = await asyncio.to_thread(col.status, sid)
        except Exception as e:
            st = {"id": sid, "status": job.get("last_status"), "outcome": None, "usage": None, "error": str(e)}
        out = dict(st or {})
        out.setdefault("id", sid)
        out["job"] = job
        out["pending_asks"] = _registry(request).asks(sid)
        return out

    @app.get("/api/jobs/{sid}/outputs")
    async def outputs(request: Request, sid: str):
        _job_or_404(request, sid)
        col = get_collector(request)
        try:
            return await asyncio.to_thread(col.outputs, sid)
        except Exception as e:
            log.exception("outputs failed"); raise HTTPException(502, "outputs failed") from e

    @app.get("/api/jobs/{sid}/outputs/{file_id}")
    async def download(request: Request, sid: str, file_id: str):
        _job_or_404(request, sid)
        col = get_collector(request)
        try:
            files = await asyncio.to_thread(col.outputs, sid)
        except Exception:
            files = []
        meta = next((f for f in files if f.get("id") == file_id), None)
        if meta is None:
            raise HTTPException(404, "unknown file")   # never fetch a file id the session didn't list
        filename = meta.get("filename") or file_id
        safe = filename.replace("\\", "/").replace("..", "_").lstrip("/")
        base = (_registry(request).state_dir / "downloads" / sid).resolve()
        dest = (base / safe).resolve()
        if base != dest and base not in dest.parents:
            raise HTTPException(400, "bad filename")
        try:
            path = await asyncio.to_thread(col.download, sid, file_id, dest)
        except Exception as e:
            log.exception("download failed"); raise HTTPException(404, "download failed") from e
        return FileResponse(Path(path), filename=Path(safe).name, media_type="application/octet-stream")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _final_status(col, sid: str) -> str:
    try:
        st = col.status(sid) or {}
        return str(st.get("status") or "ended")
    except Exception:
        return "ended"


app = create_app()
