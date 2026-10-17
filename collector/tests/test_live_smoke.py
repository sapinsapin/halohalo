"""Tests for collector.live_smoke against the scripted FakeClient (no network)."""
from __future__ import annotations

import io

import pytest

from collector import live_smoke, runtime, setup_sdk
from collector.tests.fake_client import (
    FakeClient, demo_script, ev_idle, ev_message, ev_outcome_end, ev_tool_use, fake_file,
)

REPORT = b"# Smoke report\n\n| lang | queries | accepted |\n| ceb | 3 | 20 |\n\n## Not done\nnothing\n"


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GITHUB_TOKEN", "HF_TOKEN",
                "TAVILY_API_KEY", "HALO_AGENT_ID", "HALO_ENVIRONMENT_ID", "HALO_VAULT_ID"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("COLLECTOR_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(runtime, "_sleep", lambda s: None)
    return tmp_path


def _ready(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("HALO_AGENT_ID", "agent_smoke")
    monkeypatch.setenv("HALO_ENVIRONMENT_ID", "env_smoke")
    monkeypatch.setenv("HALO_VAULT_ID", "vlt_smoke")


def _run(client, *argv, stdin=None):
    out = io.StringIO()
    code = live_smoke.run(list(argv), client=client, stdin=stdin or io.StringIO(), stdout=out)
    return code, out.getvalue()


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def test_preflight_without_key_exits_2_and_touches_no_client():
    client = FakeClient(script=demo_script())
    code, text = _run(client)
    assert code == live_smoke.EXIT_PREFLIGHT
    assert "ANTHROPIC_API_KEY=unset" in text
    assert "ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) is not set" in text
    assert "sk-ant" not in text                   # names only, never values
    assert client.created_sessions == [] and client.calls == [] and client.sent_events == []


def test_preflight_without_ids_exits_2_with_setup_commands(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "oauth-test")
    monkeypatch.setattr(live_smoke.config, "LOCK_FILE", live_smoke.config.LOCK_FILE.parent / "nope.json")
    client = FakeClient(script=demo_script())
    code, text = _run(client)
    assert code == live_smoke.EXIT_PREFLIGHT
    assert "python -m collector.setup_sdk" in text
    assert "python -m collector.live_smoke --setup" in text
    assert "oauth-test" not in text
    assert client.created_sessions == []


def test_preflight_warns_on_optional_vars(monkeypatch):
    _ready(monkeypatch)
    out = io.StringIO()
    assert live_smoke.preflight(out=out) == []
    text = out.getvalue()
    assert "GITHUB_TOKEN unset" in text
    assert "HF_TOKEN unset" in text and "TAVILY_API_KEY unset" in text
    assert "vault credential" in text


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_happy_path_exits_0_downloads_report_and_never_pushes(monkeypatch, tmp_path):
    _ready(monkeypatch)
    client = FakeClient(
        script=demo_script(),
        files=[fake_file("report.md", content=REPORT), fake_file("validated/ceb.jsonl", content=b"{}\n")],
        files_empty_first=1,
        outcomes=["satisfied"],
    )
    code, text = _run(client, "--kind", "validate", "--langs", "ceb", "--max-sentences", "20")
    assert code == live_smoke.EXIT_OK, text

    # one session, kickoff built from a push=False spec
    assert len(client.created_sessions) == 1
    created = client.created_sessions[0]
    assert created["agent"] == "agent_smoke" and created["environment_id"] == "env_smoke"
    assert created["vault_ids"] == ["vlt_smoke"]
    kickoff = created["initial_events"][0]
    assert kickoff["type"] == "user.define_outcome"
    assert "Hub push allowed: no" in kickoff["description"]
    assert "up to 20 judged sentences" in kickoff["description"]
    assert "sesn_fake001" in text and "platform.claude.com" in text

    # the paused call was denied (stdin is not a tty)
    assert client.confirmations == [{
        "type": "user.tool_confirmation", "tool_use_id": client.script[4].id,
        "result": "deny", "deny_message": live_smoke.DENY_MESSAGE,
    }]
    assert "denied (stdin is not a tty)" in text
    assert not any(ev.get("type") == "user.interrupt" for _, ev in client.sent_events)

    # status, outputs, report
    assert "status: {" in text and "'outcome': 'satisfied'" in text
    assert "report.md" in text and "validated/ceb.jsonl" in text
    dest = tmp_path / "state" / "smoke" / "sesn_fake001" / "report.md"
    assert dest.read_bytes() == REPORT
    assert "# Smoke report" in text and "## Not done" in text
    assert "result: SATISFIED" in text


def test_push_flag_is_not_accepted():
    with pytest.raises(SystemExit):
        live_smoke._build_parser().parse_args(["--push"])


def test_deny_asks_flag_denies_even_on_a_tty(monkeypatch):
    _ready(monkeypatch)

    class Tty(io.StringIO):
        def isatty(self):
            return True

    client = FakeClient(script=demo_script(), outcomes=["satisfied"])
    code, text = _run(client, "--deny-asks", stdin=Tty("y\n"))
    assert code == live_smoke.EXIT_OK
    assert client.confirmations[0]["result"] == "deny"
    assert "denied (--deny-asks)" in text


def test_tty_yes_allows(monkeypatch):
    _ready(monkeypatch)

    class Tty(io.StringIO):
        def isatty(self):
            return True

    client = FakeClient(script=demo_script(), outcomes=["satisfied"])
    code, text = _run(client, stdin=Tty("y\n"))
    assert code == live_smoke.EXIT_OK
    assert client.confirmations[0]["result"] == "allow"
    assert "deny_message" not in client.confirmations[0]


def test_unsatisfied_outcome_exits_1(monkeypatch):
    _ready(monkeypatch)
    script = [
        ev_message("Could not finish."),
        ev_outcome_end("unsatisfied", "report.md missing", 3),
        ev_idle("end_turn"),
    ]
    client = FakeClient(script=script, outcomes=["unsatisfied"])
    code, text = _run(client, "--deny-asks")
    assert code == live_smoke.EXIT_UNSATISFIED
    assert "report.md: not among the outputs" in text
    assert "not satisfied" in text


# ---------------------------------------------------------------------------
# timeout
# ---------------------------------------------------------------------------


def test_timeout_exits_3_and_interrupts(monkeypatch, tmp_path):
    _ready(monkeypatch)
    # the first clock read is the start; every later read is far past the budget
    ticks = iter([0.0] + [10 ** 6] * 50)
    monkeypatch.setattr(live_smoke, "_monotonic", lambda: next(ticks))

    script = [
        ev_message("Working..."),
        ev_tool_use("bash", {"command": "sleep 3600"}),
        ev_message("still going"),
        ev_outcome_end("satisfied", "n/a", 0),
        ev_idle("end_turn"),
    ]
    client = FakeClient(script=script, status="idle", outcomes=[],
                        files=[fake_file("report.md", content=REPORT)])
    code, text = _run(client, "--timeout-min", "5")
    assert code == live_smoke.EXIT_TIMEOUT
    interrupts = [ev for _, ev in client.sent_events if ev.get("type") == "user.interrupt"]
    assert len(interrupts) == 1
    assert "timeout: 5 min elapsed" in text
    # the loop stopped after the first event, before the later script entries
    assert "Working..." in text and "still going" not in text
    # status / outputs / report still reported after the interrupt
    assert "status: {" in text
    assert (tmp_path / "state" / "smoke" / "sesn_fake001" / "report.md").exists()
    assert "result: TIMEOUT" in text


# ---------------------------------------------------------------------------
# --setup
# ---------------------------------------------------------------------------


def test_setup_calls_apply_and_uses_its_ids(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(live_smoke.config, "LOCK_FILE", live_smoke.config.LOCK_FILE.parent / "nope.json")
    seen = {}

    def fake_apply(lock_path=setup_sdk.LOCK, *, deployment=False, client=None):
        seen["client"] = client
        seen["deployment"] = deployment
        return {"agent_id": "agent_new", "environment_id": "env_new", "vault_id": "vlt_new",
                "deployment_id": None}

    monkeypatch.setattr(setup_sdk, "apply", fake_apply)
    client = FakeClient(script=demo_script(), outcomes=["satisfied"])
    code, text = _run(client, "--setup", "--deny-asks")
    assert code == live_smoke.EXIT_OK, text
    assert seen == {"client": client, "deployment": False}
    created = client.created_sessions[0]
    assert created["agent"] == "agent_new" and created["environment_id"] == "env_new"
    assert created["vault_ids"] == ["vlt_new"]
    assert "setup: agent=agent_new" in text
