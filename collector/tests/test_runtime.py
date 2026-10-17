"""Tests for collector.runtime.Collector against the scripted FakeClient (no network)."""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from collector import config, runtime
from collector.runtime import Collector
from collector.tests.fake_client import (
    FakeClient, demo_script, ev_define_outcome, ev_idle, ev_message, ev_model_request_end,
    ev_other, ev_outcome_end, ev_terminated, ev_tool_use, fake_file,
)
from types import SimpleNamespace as NS


# ---------------------------------------------------------------------------
# Minimal stand-in for collector.jobs (worker A's module may not exist yet)
# ---------------------------------------------------------------------------


@dataclass
class JobSpec:
    kind: str
    languages: list[str] = field(default_factory=list)
    max_docs: int = 300
    max_sentences: int = 500
    push: bool = False
    notes: str = ""
    max_iterations: int = 4


def _build_kickoff(spec: JobSpec) -> dict:
    return {
        "type": "user.define_outcome",
        "description": f"{spec.kind} for {','.join(spec.languages)}",
        "rubric": {"type": "text", "content": "- outputs exist"},
        "max_iterations": spec.max_iterations,
    }


def _job_title(spec: JobSpec) -> str:
    return f"{spec.kind} · {','.join(spec.languages)} · 2026-10-17"


@pytest.fixture
def fake_jobs(monkeypatch):
    mod = types.ModuleType("collector.jobs")
    mod.JobSpec = JobSpec
    mod.build_kickoff = _build_kickoff
    mod.job_title = _job_title
    mod.validate_spec = lambda spec: None
    monkeypatch.setitem(sys.modules, "collector.jobs", mod)
    return mod


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("HALO_AGENT_ID", "HALO_ENVIRONMENT_ID", "HALO_VAULT_ID", "GITHUB_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(runtime, "_sleep", lambda s: None)


def make(client, **kw) -> Collector:
    kw.setdefault("agent_id", "agent_test")
    kw.setdefault("environment_id", "env_test")
    return Collector(client, **kw)


SPEC = JobSpec(kind="collect", languages=["ceb", "war"], max_docs=50)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_load_ids_env_first(monkeypatch, tmp_path):
    lock = tmp_path / "claude-lock.json"
    lock.write_text('{"resources": {"./agents/halo-collector.md": {"id": "agent_lock"},'
                    ' "./environments/halo-collector.yaml": {"id": "env_lock"}}}')
    monkeypatch.setenv("HALO_AGENT_ID", "agent_env")
    ids = config.load_ids(lock)
    assert ids == {"agent_id": "agent_env", "environment_id": "env_lock", "vault_id": None}


def test_load_ids_missing_lock(tmp_path):
    ids = config.load_ids(tmp_path / "nope.json")
    assert ids == {"agent_id": None, "environment_id": None, "vault_id": None}


def test_repo_settings(monkeypatch):
    assert config.repo_settings() == {"url": "https://github.com/sapinsapin/halohalo", "branch": "main", "token": None}
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
    assert config.repo_settings()["token"] == "ghp_x"


# ---------------------------------------------------------------------------
# start()
# ---------------------------------------------------------------------------


def test_start_without_token_mounts_public_repo(fake_jobs, caplog):
    client = FakeClient()
    c = make(client, vault_ids=["vlt_1"])
    with caplog.at_level("WARNING", logger="collector.runtime"):
        sid = c.start(SPEC)
    assert sid == client.session_ids[0]
    kw = client.created_sessions[0]
    assert kw["agent"] == "agent_test"
    assert kw["environment_id"] == "env_test"
    assert kw["title"] == "collect · ceb,war · 2026-10-17"
    assert kw["vault_ids"] == ["vlt_1"]
    assert kw["resources"] == [{
        "type": "github_repository",
        "url": "https://github.com/sapinsapin/halohalo",
        "mount_path": "/mnt/repo",
        "checkout": {"type": "branch", "name": "main"},
    }]
    assert "without a token (public repo only)" in caplog.text
    assert len(kw["initial_events"]) == 1
    kickoff = kw["initial_events"][0]
    assert kickoff["type"] == "user.define_outcome"
    assert kickoff["rubric"]["type"] == "text"
    assert kickoff["max_iterations"] == 4


def test_start_with_token_mounts_repo(fake_jobs):
    client = FakeClient()
    c = make(client, repo_token="ghp_secret", repo_branch="main", vault_ids=[])
    c.start(SPEC, title="custom title")
    kw = client.created_sessions[0]
    assert kw["title"] == "custom title"
    assert "vault_ids" not in kw
    assert kw["resources"] == [{
        "type": "github_repository",
        "url": "https://github.com/sapinsapin/halohalo",
        "mount_path": "/mnt/repo",
        "authorization_token": "ghp_secret",
        "checkout": {"type": "branch", "name": "main"},
    }]


def test_start_requires_ids(fake_jobs):
    c = Collector(FakeClient(), agent_id=None, environment_id=None)
    with pytest.raises(RuntimeError):
        c.start(SPEC)


def test_constructor_defaults_from_env(monkeypatch):
    monkeypatch.setenv("HALO_AGENT_ID", "agent_env")
    monkeypatch.setenv("HALO_ENVIRONMENT_ID", "env_env")
    monkeypatch.setenv("HALO_VAULT_ID", "vlt_env")
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    c = Collector(FakeClient())
    assert (c.agent_id, c.environment_id, c.vault_ids, c.repo_token) == ("agent_env", "env_env", ["vlt_env"], "tok")


def test_constructor_never_builds_sdk_client():
    c = Collector()  # no client, no key: must not raise
    assert c._client is None


# ---------------------------------------------------------------------------
# events()
# ---------------------------------------------------------------------------


def test_events_normalizes_demo_script():
    client = FakeClient(script=demo_script())
    events = list(make(client).events("sesn_x"))
    types_ = [e["type"] for e in events]
    assert types_ == ["user_event", "message", "tool_use", "usage", "tool_ask", "idle",
                      "message", "outcome", "idle"]
    assert set(events[0]) == set(runtime._EVENT_KEYS)
    assert events[0]["raw_type"] == "user.define_outcome"
    assert events[1]["text"].startswith("Starting collection")
    assert events[2]["tool"] == "bash" and events[2]["permission"] == "allow"
    assert events[2]["input"] == {"command": "ls /mnt/repo"}
    assert events[3]["input"]["input_tokens"] == 3571 and events[3]["input"]["output_tokens"] == 727
    assert events[5]["stop_reason"] == "requires_action"
    assert events[7]["result"] == "satisfied" and events[7]["iteration"] == 0
    assert "All criteria" in events[7]["explanation"]
    assert events[8]["stop_reason"] == "end_turn"
    assert client.stream_opened == 1
    # stream-first: stream opened before history was listed
    assert client.calls.index(("events.stream", "sesn_x")) < client.calls.index(("events.list", "sesn_x"))


def test_events_surfaces_tool_ask_without_answering():
    ask = ev_tool_use("bash", {"command": "rm -rf /"}, permission="ask")
    client = FakeClient(script=[ask, ev_idle("requires_action", [ask.id]), ev_idle("end_turn")])
    events = list(make(client).events("sesn_x", include_history=False))
    asks = [e for e in events if e["type"] == "tool_ask"]
    assert len(asks) == 1
    assert asks[0]["id"] == ask.id and asks[0]["permission"] == "ask"
    assert client.sent_events == []  # not auto-answered
    # requires_action idle does not terminate the loop
    assert [e["type"] for e in events] == ["tool_ask", "idle", "idle"]


def _ev_tool_confirmation(tool_use_id: str, result: str = "deny") -> NS:
    return NS(type="user.tool_confirmation", id=f"sevt_conf_{tool_use_id}", processed_at=None,
              tool_use_id=tool_use_id, result=result, deny_message=None)


def test_tool_confirmation_echo_carries_result_and_tool():
    ev = runtime.normalize_event(_ev_tool_confirmation("sevt_ask1", "allow"))
    assert ev["type"] == "user_event" and ev["raw_type"] == "user.tool_confirmation"
    assert ev["result"] == "allow" and ev["tool"] == "sevt_ask1"
    assert ev["answered"] is False


def test_answered_history_asks_replay_as_tool_use():
    answered_ask = ev_tool_use("bash", {"command": "git push"}, permission="ask")
    open_ask = ev_tool_use("bash", {"command": "rm -rf x"}, permission="ask")
    history = [
        ev_message("start"),
        answered_ask,
        ev_idle("requires_action", [answered_ask.id]),
        _ev_tool_confirmation(answered_ask.id, "deny"),
        open_ask,
        ev_idle("requires_action", [open_ask.id]),
    ]
    client = FakeClient(history=history, script=[ev_idle("end_turn")])
    events = list(make(client).events("sesn_x"))
    by_id = {e["id"]: e for e in events}
    assert by_id[answered_ask.id]["type"] == "tool_use"
    assert by_id[answered_ask.id]["permission"] == "ask"
    assert by_id[answered_ask.id]["answered"] is True
    assert by_id[open_ask.id]["type"] == "tool_ask"
    assert by_id[open_ask.id]["answered"] is False
    assert [e["type"] for e in events] == [
        "message", "tool_use", "idle", "user_event", "tool_ask", "idle", "idle"]
    assert all("answered" in e for e in events)


class _ClosableStream:
    """Stream whose iterator blocks-then-raises once close() ran, like httpx."""

    def __init__(self, script):
        self._script = list(script)
        self.closed = False
        self.response = NS(is_closed=False)

    def close(self):
        self.closed = True
        self.response.is_closed = True

    def __iter__(self):
        for ev in self._script:
            if self.closed:
                raise RuntimeError("stream closed")
            yield ev
        if self.closed:
            raise RuntimeError("stream closed")


def _install_stream(client, stream):
    import contextlib

    @contextlib.contextmanager
    def fake_stream(*, session_id, **_):
        client.calls.append(("events.stream", session_id))
        client.stream_opened += 1
        try:
            yield stream
        finally:
            stream.close()

    client.beta.sessions.events.stream = fake_stream


def test_on_open_gets_stream_and_close_ends_cleanly():
    client = FakeClient()
    stream = _ClosableStream([ev_message("one"), ev_message("two"), ev_idle("end_turn")])
    _install_stream(client, stream)
    opened = []
    out = []
    for ev in make(client).events("sesn_x", include_history=False, on_open=opened.append):
        out.append(ev["type"])
        if len(out) == 1:
            opened[0].close()  # the web app's disconnect handler
    assert opened == [stream]
    assert out == ["message"]  # ended cleanly: no error, no raise, no further events
    assert stream.closed


def test_closed_stream_detected_via_sdk_response_flag():
    # The SDK Stream has no `closed` attribute: close() closes `stream.response`
    # (httpx), and the blocked iterator then raises. That must end the generator.
    class _SdkLike(_ClosableStream):
        def __iter__(self):
            yield ev_message("a")
            raise RuntimeError("stream closed")

    client = FakeClient()
    stream = _SdkLike([])
    _install_stream(client, stream)
    gen = make(client).events("sesn_x", include_history=False)
    assert next(gen)["type"] == "message"
    stream.response.is_closed = True     # only the response reports closed
    assert list(gen) == []               # raised inside the loop -> swallowed, no error


def test_unexpected_stream_error_still_raises():
    client = FakeClient()

    class _Boom(_ClosableStream):
        def __iter__(self):
            yield ev_message("x")
            raise RuntimeError("network broke")

    _install_stream(client, _Boom([]))
    gen = make(client).events("sesn_x", include_history=False)
    assert next(gen)["type"] == "message"
    with pytest.raises(RuntimeError, match="network broke"):
        next(gen)


def test_events_stops_at_terminal_gate_and_ignores_rest():
    client = FakeClient(script=[ev_message("a"), ev_idle("end_turn"), ev_message("never seen")])
    events = list(make(client).events("sesn_x"))
    assert [e["type"] for e in events] == ["message", "idle"]

    client = FakeClient(script=[ev_message("a"), ev_terminated(), ev_message("never")])
    events = list(make(client).events("sesn_x"))
    assert [e["type"] for e in events] == ["message", "terminated"]


def test_events_dedupes_history_vs_stream():
    m1, m2 = ev_message("one"), ev_message("two")
    done = ev_idle("end_turn")
    client = FakeClient(history=[m1, m2], script=[m1, m2, done])
    events = list(make(client).events("sesn_x"))
    assert [e["id"] for e in events] == [m1.id, m2.id, done.id]


def test_events_terminal_in_history_still_breaks():
    done = ev_idle("end_turn")
    client = FakeClient(history=[ev_message("x"), done], script=[ev_message("x"), done])
    events = list(make(client).events("sesn_x"))
    assert [e["type"] for e in events] == ["message", "idle"]


def test_events_terminal_only_in_history_dedup_still_gates():
    # Terminal event already seen via history, then seen again on the stream:
    # the gate must run even though the event itself is suppressed.
    done = ev_idle("end_turn")
    client = FakeClient(history=[], script=[ev_message("x"), done, done, ev_message("after")])
    events = list(make(client).events("sesn_x", include_history=False))
    assert [e["type"] for e in events] == ["message", "idle"]


def test_events_include_history_false_skips_list():
    client = FakeClient(history=[ev_message("old")], script=[ev_idle("end_turn")])
    events = list(make(client).events("sesn_x", include_history=False))
    assert [e["type"] for e in events] == ["idle"]
    assert ("events.list", "sesn_x") not in client.calls


def test_events_other_and_mcp_and_custom():
    script = [
        ev_other("agent.thinking"),
        ev_tool_use("search", {"q": "x"}, permission="allow", mcp_server_name="tavily"),
        ev_idle("retries_exhausted"),
    ]
    events = list(make(FakeClient(script=script)).events("s"))
    assert events[0]["type"] == "other" and events[0]["raw_type"] == "agent.thinking"
    assert events[1]["type"] == "tool_use" and events[1]["raw_type"] == "agent.mcp_tool_use"
    assert events[2]["type"] == "idle" and events[2]["stop_reason"] == "retries_exhausted"

    custom = runtime.normalize_event(types.SimpleNamespace(type="agent.custom_tool_use", id="e1", name="run", input={"a": 1}))
    assert custom["type"] == "custom_tool_use" and custom["tool"] == "run" and custom["ts"] is None


def test_normalize_tolerates_missing_attributes():
    ev = runtime.normalize_event(types.SimpleNamespace(type="session.status_idle", id="e"))
    assert ev["type"] == "idle" and ev["stop_reason"] is None
    ev = runtime.normalize_event(types.SimpleNamespace(type="agent.tool_use", id="e"))
    assert ev["type"] == "tool_use" and ev["tool"] is None and ev["input"] == {}


# ---------------------------------------------------------------------------
# confirm / interrupt / status / outputs / download / console_url
# ---------------------------------------------------------------------------


def test_confirm_payloads():
    client = FakeClient()
    c = make(client)
    c.confirm("sesn_x", "sevt_ask", True, message="ignored for allow")
    c.confirm("sesn_x", "sevt_ask2", False, message="not today")
    c.confirm("sesn_x", "sevt_ask3", False)
    assert client.sent_events == [
        ("sesn_x", {"type": "user.tool_confirmation", "tool_use_id": "sevt_ask", "result": "allow"}),
        ("sesn_x", {"type": "user.tool_confirmation", "tool_use_id": "sevt_ask2", "result": "deny",
                    "deny_message": "not today"}),
        ("sesn_x", {"type": "user.tool_confirmation", "tool_use_id": "sevt_ask3", "result": "deny"}),
    ]


def test_deny_all_and_interrupt():
    client = FakeClient()
    c = make(client)
    c.deny_all("s", ["a", "b"], "unattended")
    c.interrupt("s")
    assert [e["tool_use_id"] for e in client.confirmations] == ["a", "b"]
    assert all(e["result"] == "deny" and e["deny_message"] == "unattended" for e in client.confirmations)
    assert client.sent_events[-1] == ("s", {"type": "user.interrupt"})


def test_status_shape():
    client = FakeClient(status="idle", outcomes=["needs_revision", "satisfied"],
                        usage=types.SimpleNamespace(input_tokens=10, output_tokens=5))
    st = make(client).status("sesn_x")
    assert st == {"id": "sesn_x", "status": "idle", "outcome": "satisfied",
                  "usage": {"input_tokens": 10, "output_tokens": 5}}

    st = make(FakeClient(status="running", usage=None)).status("sesn_y")
    assert st["outcome"] is None and st["status"] == "running"


def test_outputs_retries_until_indexed():
    files = [fake_file("report.md", 12), fake_file("manifest.jsonl", 34)]
    client = FakeClient(files=files, files_empty_first=2)
    out = make(client).outputs("sesn_x")
    assert client.files_list_calls == 3
    assert out == [{"id": files[0].id, "filename": "report.md", "size_bytes": 12},
                   {"id": files[1].id, "filename": "manifest.jsonl", "size_bytes": 34}]
    assert ("files.list", "sesn_x") in client.calls


def test_outputs_gives_up_after_retries():
    client = FakeClient(files=[fake_file("late.md")], files_empty_first=10)
    assert make(client).outputs("sesn_x") == []
    assert client.files_list_calls == runtime.OUTPUT_LIST_RETRIES + 1


def test_download_and_console_url(tmp_path):
    f = fake_file("report.md", content=b"# hi")
    client = FakeClient(files=[f])
    c = make(client)
    dest = c.download("sesn_x", f.id, tmp_path / "out" / "report.md")
    assert dest == tmp_path / "out" / "report.md" and dest.read_bytes() == b"# hi"
    assert c.console_url("sesn_x") == "https://platform.claude.com/workspaces/default/sessions/sesn_x"


def test_full_flow_with_confirm(fake_jobs):
    client = FakeClient(script=demo_script(), files=[fake_file("report.md")], outcomes=["satisfied"])
    c = make(client)
    sid = c.start(SPEC)
    for ev in c.events(sid):
        if ev["type"] == "tool_ask":
            c.confirm(sid, ev["id"], False, "no pushes from the demo")
    assert len(client.confirmations) == 1 and client.confirmations[0]["result"] == "deny"
    assert c.status(sid)["outcome"] == "satisfied"
    assert [f["filename"] for f in c.outputs(sid)] == ["report.md"]
