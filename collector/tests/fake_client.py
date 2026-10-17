"""A scripted stand-in for `anthropic.Anthropic()` covering the surface the
collector runtime uses. No network.

Surface mimicked:
    client.beta.sessions.create(**kw)            -> session object (id, status)
    client.beta.sessions.retrieve(session_id=)   -> session object
    client.beta.sessions.events.send(session_id=, events=)
    client.beta.sessions.events.stream(session_id=)  (context manager -> iterable)
    client.beta.sessions.events.list(session_id=)    (iterable, has .data)
    client.beta.files.list(scope_id=, betas=)        (iterable, has .data)
    client.beta.files.download(file_id)              -> object with write_to_file()

Events are `types.SimpleNamespace` objects with the same attribute names as the
SDK's pydantic models (`type`, `id`, `processed_at`, `content`, `name`, `input`,
`evaluated_permission`, `stop_reason.type`, `result`, `explanation`,
`iteration`, `model_usage`, ...). Build them with the `ev_*` helpers below.

Usage (worker C):

    from collector.tests.fake_client import FakeClient, demo_script
    client = FakeClient(script=demo_script())
    collector = Collector(client, agent_id="agent_x", environment_id="env_x")
    sid = collector.start(spec)
    for ev in collector.events(sid): ...
    client.sent_events          # [(session_id, event_dict), ...]
    client.created_sessions     # [kwargs passed to sessions.create, ...]
"""
from __future__ import annotations

import contextlib
import itertools
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Iterable

_counter = itertools.count(1)


def _next_id(prefix: str = "sevt") -> str:
    return f"{prefix}_{next(_counter):04d}"


def _ts(n: int | None = None) -> str:
    n = next(_counter) if n is None else n
    return f"2026-10-17T10:{n % 60:02d}:00Z"


# ---------------------------------------------------------------------------
# Event builders (attribute names match the SDK types)
# ---------------------------------------------------------------------------


def ev_define_outcome(description: str = "Collect text", id: str | None = None) -> NS:
    return NS(
        type="user.define_outcome", id=id or _next_id(), processed_at=_ts(),
        description=description, rubric=NS(type="text", content="- criteria"),
        max_iterations=4, outcome_id="outc_demo",
    )


def ev_message(text: str, id: str | None = None) -> NS:
    return NS(
        type="agent.message", id=id or _next_id(), processed_at=_ts(),
        content=[NS(type="text", text=text)],
    )


def ev_tool_use(
    name: str = "bash", input: dict | None = None, permission: str = "allow",
    id: str | None = None, mcp_server_name: str | None = None,
) -> NS:
    ev = NS(
        type="agent.mcp_tool_use" if mcp_server_name else "agent.tool_use",
        id=id or _next_id(), processed_at=_ts(), name=name,
        input=input if input is not None else {"command": "ls /mnt/session/outputs"},
        evaluated_permission=permission,
        evaluation=NS(type="auto", reason_code=None),
    )
    if mcp_server_name:
        ev.mcp_server_name = mcp_server_name
    return ev


def ev_custom_tool_use(name: str, input: dict | None = None, id: str | None = None) -> NS:
    return NS(
        type="agent.custom_tool_use", id=id or _next_id(), processed_at=_ts(),
        name=name, input=input or {},
    )


def ev_tool_result(tool_use_id: str, text: str = "ok", id: str | None = None) -> NS:
    return NS(
        type="agent.tool_result", id=id or _next_id(), processed_at=_ts(),
        tool_use_id=tool_use_id, content=[NS(type="text", text=text)],
    )


def ev_idle(stop_reason: str = "end_turn", event_ids: list[str] | None = None, id: str | None = None) -> NS:
    stop = NS(type=stop_reason)
    if stop_reason == "requires_action":
        stop.event_ids = event_ids or []
    return NS(type="session.status_idle", id=id or _next_id(), processed_at=_ts(), stop_reason=stop, stop_details=None)


def ev_terminated(id: str | None = None) -> NS:
    return NS(type="session.status_terminated", id=id or _next_id(), processed_at=_ts())


def ev_outcome_end(result: str = "satisfied", explanation: str = "All criteria met.", iteration: int = 0, id: str | None = None) -> NS:
    return NS(
        type="span.outcome_evaluation_end", id=id or _next_id(), processed_at=_ts(),
        outcome_evaluation_start_id=_next_id(), outcome_id="outc_demo",
        result=result, explanation=explanation, iteration=iteration,
        usage=NS(input_tokens=2400, output_tokens=350, cache_creation_input_tokens=0, cache_read_input_tokens=1800),
    )


def ev_model_request_end(input_tokens: int = 3571, output_tokens: int = 727, id: str | None = None) -> NS:
    return NS(
        type="span.model_request_end", id=id or _next_id(), processed_at=_ts(),
        is_error=False, model_request_start_id=_next_id(),
        model_usage=NS(input_tokens=input_tokens, output_tokens=output_tokens,
                       cache_creation_input_tokens=0, cache_read_input_tokens=6656),
    )


def ev_other(type: str = "agent.thinking", id: str | None = None) -> NS:
    return NS(type=type, id=id or _next_id(), processed_at=_ts())


def fake_file(filename: str, size_bytes: int = 1024, id: str | None = None, content: bytes = b"") -> NS:
    return NS(id=id or _next_id("file"), filename=filename, size_bytes=size_bytes,
              mime_type="application/octet-stream", created_at=_ts(), _content=content or filename.encode())


def demo_script() -> list[NS]:
    """A realistic collect run: kickoff echo, work, one paused call, grader pass."""
    ask = ev_tool_use("bash", {"command": "git push origin main"}, permission="ask")
    return [
        ev_define_outcome("Collect Cebuano and Waray text into /mnt/session/outputs/"),
        ev_message("Starting collection for ceb, war. Checking the repo mount."),
        ev_tool_use("bash", {"command": "ls /mnt/repo"}, permission="allow"),
        ev_model_request_end(),
        ask,
        ev_idle("requires_action", event_ids=[ask.id]),
        ev_message("Shards staged under /mnt/session/outputs/text/. Writing report.md."),
        ev_outcome_end("satisfied", "All criteria met: shards, manifest and report present.", 0),
        ev_idle("end_turn"),
    ]


# ---------------------------------------------------------------------------
# Fake client
# ---------------------------------------------------------------------------


class _Page:
    """Iterable with a `.data` list, like the SDK's SyncPageCursor."""

    def __init__(self, data: list):
        self.data = list(data)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


class _Download:
    def __init__(self, content: bytes):
        self.content = content

    def write_to_file(self, path) -> None:
        Path(path).write_bytes(self.content)


class _Events:
    def __init__(self, client: "FakeClient"):
        self._c = client

    def send(self, *, session_id: str, events: Iterable[dict], **_: Any) -> NS:
        events = list(events)
        for ev in events:
            self._c.sent_events.append((session_id, ev))
        return NS(events=[NS(id=_next_id(), **ev) for ev in events])

    def list(self, *, session_id: str, **_: Any) -> _Page:
        self._c.calls.append(("events.list", session_id))
        return _Page(self._c.history)

    @contextlib.contextmanager
    def stream(self, *, session_id: str, **_: Any):
        self._c.calls.append(("events.stream", session_id))
        self._c.stream_opened += 1
        yield iter(list(self._c.script))


class _Sessions:
    def __init__(self, client: "FakeClient"):
        self._c = client
        self.events = _Events(client)

    def create(self, **kwargs: Any) -> NS:
        self._c.created_sessions.append(kwargs)
        sid = f"sesn_fake{len(self._c.created_sessions):03d}"
        self._c.session_ids.append(sid)
        return NS(id=sid, status="running", title=kwargs.get("title"))

    def retrieve(self, *, session_id: str, **_: Any) -> NS:
        self._c.calls.append(("sessions.retrieve", session_id))
        return NS(
            id=session_id, type="session", status=self._c.status,
            outcome_evaluations=[NS(outcome_id="outc_demo", result=r, iteration=i, explanation="x")
                                 for i, r in enumerate(self._c.outcomes)],
            usage=self._c.usage,
        )


class _Files:
    def __init__(self, client: "FakeClient"):
        self._c = client

    def list(self, *, scope_id: str | None = None, **_: Any) -> _Page:
        self._c.calls.append(("files.list", scope_id))
        self._c.files_list_calls += 1
        if self._c.files_list_calls <= self._c.files_empty_first:
            return _Page([])
        return _Page(self._c.files)

    def download(self, file_id: str, **_: Any) -> _Download:
        self._c.calls.append(("files.download", file_id))
        for f in self._c.files:
            if f.id == file_id:
                return _Download(getattr(f, "_content", b""))
        raise KeyError(f"unknown file {file_id}")


class FakeClient:
    """Scripted stand-in for `anthropic.Anthropic()`.

    script            events the live stream yields, in order
    history           events `events.list` returns (defaults to empty)
    files             fake file objects `files.list` returns
    files_empty_first `files.list` returns [] this many times first (indexing lag)
    status            value `sessions.retrieve().status` reports
    outcomes          outcome_evaluations results in order
    usage             object/dict returned as `session.usage`
    """

    def __init__(
        self,
        script: list | None = None,
        *,
        history: list | None = None,
        files: list | None = None,
        files_empty_first: int = 0,
        status: str = "idle",
        outcomes: list[str] | None = None,
        usage: Any = None,
    ) -> None:
        self.script = list(script or [])
        self.history = list(history or [])
        self.files = list(files or [])
        self.files_empty_first = files_empty_first
        self.files_list_calls = 0
        self.status = status
        self.outcomes = list(outcomes or [])
        self.usage = usage if usage is not None else NS(input_tokens=10, output_tokens=5)
        self.sent_events: list[tuple[str, dict]] = []
        self.created_sessions: list[dict] = []
        self.session_ids: list[str] = []
        self.calls: list[tuple] = []
        self.stream_opened = 0
        self.beta = NS(sessions=_Sessions(self), files=_Files(self))

    @property
    def confirmations(self) -> list[dict]:
        return [ev for _, ev in self.sent_events if ev.get("type") == "user.tool_confirmation"]
