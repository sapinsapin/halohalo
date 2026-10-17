"""First live run of the halo-collector Managed Agent, as one command.

    python -m collector.live_smoke [--kind validate] [--langs ceb] [--max-sentences 20]
                                   [--setup] [--deny-asks] [--timeout-min 30]

Once credentials exist this is the smoke test: it checks the environment
without calling the API, optionally creates the resources (`--setup` runs
`collector.setup_sdk.apply`, idempotent via claude-lock.json), starts one
small job with Hub push forced off, streams the session to the terminal,
answers paused tool calls (deny unless a person says yes on a tty), enforces
a wall-clock budget, then prints the status, lists the outputs and downloads
`report.md` into `collector_state/smoke/<session id>/`.

Exit codes: 0 outcome satisfied; 1 any other outcome; 2 preflight failed
(nothing was started); 3 the budget ran out (the session was interrupted).

Everything that talks to the API goes through `collector.runtime.Collector`;
tests inject `collector.tests.fake_client.FakeClient` via `run(client=...)`.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO

from . import config, jobs
from .runtime import Collector, _format_event

log = logging.getLogger("collector.live_smoke")

EXIT_OK = 0
EXIT_UNSATISFIED = 1
EXIT_PREFLIGHT = 2
EXIT_TIMEOUT = 3

DENY_MESSAGE = "Denied by collector.live_smoke (smoke test, unattended)."
REPORT_HEAD_LINES = 40
SETUP_COMMANDS = (
    "python -m collector.setup_sdk            # or: ant apply agents/halo-collector.md "
    "environments/halo-collector.yaml vaults/halo-collector.yaml",
    "python -m collector.live_smoke --setup   # same thing, then run the smoke test",
)

# Patchable in tests.
_monotonic = time.monotonic


# ---------------------------------------------------------------------------
# preflight (no API calls)
# ---------------------------------------------------------------------------

def preflight(environ: dict[str, str] | None = None, *, out: TextIO = sys.stdout) -> list[str]:
    """Print a one-line table of which env vars are set (names only) and
    return the fatal problems (empty when the run may proceed)."""
    env = os.environ if environ is None else environ
    names = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GITHUB_TOKEN", "HF_TOKEN", "TAVILY_API_KEY")
    cells = [f"{n}={'set' if env.get(n) else 'unset'}" for n in names]
    print("preflight: " + "  ".join(cells), file=out)

    problems: list[str] = []
    if not (env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN")):
        problems.append("ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) is not set")
    if not env.get("GITHUB_TOKEN"):
        print("warning: GITHUB_TOKEN unset; the repo is mounted without a token "
              "(works only while sapinsapin/halohalo is public)", file=out)
    for name in ("HF_TOKEN", "TAVILY_API_KEY"):
        if not env.get(name):
            print(f"warning: {name} unset here; only --setup reads it. What matters at run "
                  f"time is the vault credential of the same name.", file=out)
    return problems


def _require_ids(out: TextIO) -> dict[str, str | None] | None:
    ids = config.load_ids()
    if ids.get("agent_id") and ids.get("environment_id"):
        return ids
    print("preflight: agent_id/environment_id unknown (HALO_AGENT_ID/HALO_ENVIRONMENT_ID "
          "unset and no claude-lock.json). Run one of:", file=out)
    for cmd in SETUP_COMMANDS:
        print(f"  {cmd}", file=out)
    return None


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m collector.live_smoke",
        description="First live run of the halo-collector agent: preflight, one small job "
                    "(push forced off), stream, status, report.md download.",
    )
    p.add_argument("--kind", default="validate", choices=list(jobs.JOB_KINDS))
    p.add_argument("--langs", default="ceb", help="comma-separated PLD codes (default: ceb)")
    p.add_argument("--max-docs", type=int, default=20, help="collect: docs per language (default 20)")
    p.add_argument("--max-sentences", type=int, default=20,
                   help="validate/instruct: rows per language (default 20)")
    p.add_argument("--notes", default="Smoke test: keep it small and finish quickly.")
    p.add_argument("--setup", action="store_true",
                   help="run collector.setup_sdk.apply() first (idempotent via claude-lock.json)")
    p.add_argument("--deny-asks", action="store_true",
                   help="deny paused tool calls instead of asking on the terminal")
    p.add_argument("--timeout-min", type=float, default=30.0,
                   help="wall-clock budget; on expiry the session is interrupted (exit 3)")
    p.add_argument("--state-dir", default=None,
                   help="where report.md lands (default $COLLECTOR_STATE_DIR or ./collector_state)")
    return p


def _answer_ask(ev: dict, *, deny_asks: bool, stdin: TextIO, out: TextIO) -> bool:
    if deny_asks:
        print("  -> denied (--deny-asks)", file=out)
        return False
    try:
        interactive = stdin.isatty()
    except (AttributeError, ValueError):
        interactive = False
    if not interactive:
        print("  -> denied (stdin is not a tty)", file=out)
        return False
    print(f"  allow {ev['tool']}? [y/N] ", end="", file=out, flush=True)
    answer = (stdin.readline() or "").strip().lower()
    allow = answer in ("y", "yes")
    print(f"  -> {'allowed' if allow else 'denied'}", file=out)
    return allow


def run(
    argv: list[str] | None = None,
    *,
    client: Any = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> int:
    """Everything `python -m collector.live_smoke` does; returns the exit code."""
    args = _build_parser().parse_args(argv)
    out = stdout or sys.stdout
    stdin = stdin or sys.stdin

    # 1. preflight: environment only, no API calls
    problems = preflight(out=out)
    if problems:
        for p in problems:
            print(f"preflight: {p}", file=out)
        return EXIT_PREFLIGHT

    # 2. resources
    if args.setup:
        from . import setup_sdk
        ids = setup_sdk.apply(client=client)
        print(f"setup: agent={ids['agent_id']} environment={ids['environment_id']} "
              f"vault={ids['vault_id']}", file=out)
    else:
        ids = _require_ids(out)
        if ids is None:
            return EXIT_PREFLIGHT

    spec = jobs.JobSpec(
        kind=args.kind,
        languages=[s.strip() for s in args.langs.split(",") if s.strip()],
        max_docs=args.max_docs,
        max_sentences=args.max_sentences,
        push=False,                      # a smoke test never pushes
        notes=args.notes,
    )
    try:
        jobs.validate_spec(spec)
    except ValueError as exc:
        print(f"preflight: bad job spec: {exc}", file=out)
        return EXIT_PREFLIGHT

    collector = Collector(
        client,
        agent_id=ids["agent_id"],
        environment_id=ids["environment_id"],
        vault_ids=[ids["vault_id"]] if ids.get("vault_id") else [],
    )

    # 3. start
    session_id = collector.start(spec)
    print(f"session {session_id}", file=out)
    print(f"console {collector.console_url(session_id)}", file=out)

    # 4. stream, with a wall-clock budget
    budget_s = args.timeout_min * 60.0
    started = _monotonic()
    timed_out = threading.Event()
    interrupt_lock = threading.Lock()

    def _expire() -> None:
        with interrupt_lock:
            if timed_out.is_set():
                return
            timed_out.set()
        print(f"\ntimeout: {args.timeout_min:g} min elapsed; interrupting {session_id}",
              file=out, flush=True)
        try:
            collector.interrupt(session_id)
        except Exception as exc:  # pragma: no cover - the report below still prints
            log.warning("interrupt failed: %s", exc)

    # A stream that is silent past the deadline still gets interrupted; the
    # interrupt makes the session idle, which ends the loop below.
    timer = threading.Timer(budget_s, _expire)
    timer.daemon = True
    timer.start()

    outcome: str | None = None
    stop_reason: str | None = None
    stream: Any = None

    def _on_open(s: Any) -> None:
        nonlocal stream
        stream = s

    try:
        for ev in collector.events(session_id, on_open=_on_open):
            line = _format_event(ev)
            if line is not None:
                print(line, file=out, flush=True)
            if ev["type"] == "outcome":
                outcome = ev["result"]
            elif ev["type"] == "idle":
                stop_reason = ev["stop_reason"]
            elif ev["type"] == "tool_ask":
                allow = _answer_ask(ev, deny_asks=args.deny_asks, stdin=stdin, out=out)
                collector.confirm(session_id, ev["id"], allow, None if allow else DENY_MESSAGE)
            if timed_out.is_set():
                break
            if _monotonic() - started >= budget_s:
                _expire()
                break
    finally:
        timer.cancel()

    # 5. wrap-up
    print(file=out)
    status = collector.status(session_id)
    print(f"status: {status}", file=out)
    if status.get("outcome"):
        outcome = status["outcome"]
    print(f"outcome: {outcome}  last stop_reason: {stop_reason}", file=out)

    files = collector.outputs(session_id)
    if not files:
        print("outputs: (none)", file=out)
    for f in files:
        print(f"output {f['id']}  {f['filename']}  {f['size_bytes']} bytes", file=out)

    report = next((f for f in files if (f.get("filename") or "").rsplit("/", 1)[-1] == "report.md"), None)
    if report is not None:
        state_dir = Path(args.state_dir or os.environ.get("COLLECTOR_STATE_DIR", "./collector_state"))
        dest = collector.download(session_id, report["id"], state_dir / "smoke" / session_id / "report.md")
        print(f"downloaded report.md -> {dest}", file=out)
        print(f"--- report.md (first {REPORT_HEAD_LINES} lines) ---", file=out)
        with open(dest, "r", encoding="utf-8", errors="replace") as fh:
            for i, text in enumerate(fh):
                if i >= REPORT_HEAD_LINES:
                    print("...", file=out)
                    break
                print(text.rstrip("\n"), file=out)
        print("--- end report.md ---", file=out)
    else:
        print("report.md: not among the outputs", file=out)

    if timed_out.is_set():
        print(f"result: TIMEOUT after {args.timeout_min:g} min (exit {EXIT_TIMEOUT})", file=out)
        return EXIT_TIMEOUT
    if outcome == "satisfied":
        print(f"result: SATISFIED (exit {EXIT_OK})", file=out)
        return EXIT_OK
    print(f"result: outcome {outcome!r} is not satisfied (exit {EXIT_UNSATISFIED})", file=out)
    return EXIT_UNSATISFIED


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
