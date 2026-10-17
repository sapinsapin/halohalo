"""halo-collector runtime: drive a Managed Agents session for one job.

`Collector` wraps the Anthropic Managed Agents SDK surface the collector needs:
start a session with a `user.define_outcome` kickoff, stream normalized events,
answer paused tool calls, interrupt, inspect status, and fetch outputs.

Every event yielded by `events()` is a plain JSON-serialisable dict (`Event`)
with the keys fixed in collector/SPEC.md:

    {"id", "type", "ts", "text", "tool", "input", "permission", "stop_reason",
     "result", "explanation", "iteration", "raw_type", "answered"}

`answered` (bool, default False) is True only on a historical `tool_use` whose
permission was "ask" but which a later `user.tool_confirmation` already
answered; such events are replayed as `tool_use`, never as `tool_ask`.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterator

from . import config

log = logging.getLogger("collector.runtime")

Event = dict

MANAGED_AGENTS_BETA = "managed-agents-2026-04-01"
REPO_MOUNT_PATH = "/mnt/repo"
CONSOLE_BASE = "https://platform.claude.com/workspaces/default/sessions"

OUTPUT_LIST_RETRIES = 2
OUTPUT_LIST_DELAY_S = 1.5

# Patchable in tests so the retry loop does not actually sleep.
_sleep = time.sleep


def _jobs():
    """Import collector.jobs lazily (worker A owns it; tests may stub it)."""
    from collector import jobs  # noqa: WPS433

    return jobs


# ---------------------------------------------------------------------------
# Event normalization
# ---------------------------------------------------------------------------

_EVENT_KEYS = (
    "id", "type", "ts", "text", "tool", "input", "permission", "stop_reason",
    "result", "explanation", "iteration", "raw_type", "answered",
)


def _blank(raw_type: str, event_id: str | None, ts: str | None) -> Event:
    ev: Event = {k: None for k in _EVENT_KEYS}
    ev["id"] = event_id
    ev["raw_type"] = raw_type
    ev["ts"] = ts
    ev["answered"] = False
    return ev


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover - defensive
            return str(value)
    return str(value)


def _as_dict(value: Any) -> dict | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return None


def _usage_tokens(usage: Any) -> dict:
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
    }


def normalize_event(event: Any) -> Event:
    """Turn an SDK session event (or a SimpleNamespace look-alike) into an Event."""
    raw_type = getattr(event, "type", None) or "unknown"
    ev = _blank(raw_type, getattr(event, "id", None), _iso(getattr(event, "processed_at", None)))

    if raw_type == "agent.message":
        parts = []
        for block in getattr(event, "content", None) or []:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", "") or "")
        ev["type"] = "message"
        ev["text"] = "".join(parts)

    elif raw_type in ("agent.tool_use", "agent.mcp_tool_use"):
        permission = getattr(event, "evaluated_permission", None)
        ev["type"] = "tool_ask" if permission == "ask" else "tool_use"
        ev["tool"] = getattr(event, "name", None)
        ev["input"] = _as_dict(getattr(event, "input", None)) or {}
        ev["permission"] = permission

    elif raw_type == "agent.custom_tool_use":
        ev["type"] = "custom_tool_use"
        ev["tool"] = getattr(event, "name", None)
        ev["input"] = _as_dict(getattr(event, "input", None)) or {}

    elif raw_type == "session.status_idle":
        ev["type"] = "idle"
        ev["stop_reason"] = getattr(getattr(event, "stop_reason", None), "type", None)

    elif raw_type == "session.status_terminated":
        ev["type"] = "terminated"

    elif raw_type == "span.outcome_evaluation_end":
        ev["type"] = "outcome"
        ev["result"] = getattr(event, "result", None)
        ev["explanation"] = getattr(event, "explanation", None)
        ev["iteration"] = getattr(event, "iteration", None)

    elif raw_type == "span.model_request_end":
        ev["type"] = "usage"
        ev["input"] = _usage_tokens(getattr(event, "model_usage", None))

    elif raw_type.startswith("user."):
        ev["type"] = "user_event"
        if raw_type == "user.define_outcome":
            ev["text"] = getattr(event, "description", None)
        elif raw_type == "user.message":
            parts = []
            for block in getattr(event, "content", None) or []:
                if getattr(block, "type", None) == "text":
                    parts.append(getattr(block, "text", "") or "")
            ev["text"] = "".join(parts) or None
        elif raw_type == "user.tool_confirmation":
            ev["result"] = getattr(event, "result", None)
            ev["tool"] = getattr(event, "tool_use_id", None)

    else:
        ev["type"] = "other"

    return ev


def _stream_closed(stream: Any) -> bool:
    """True once `stream.close()` ran (SDK Stream closes its httpx response)."""
    if getattr(stream, "closed", False):
        return True
    response = getattr(stream, "response", None)
    return bool(getattr(response, "is_closed", False))


def is_terminal(event: Any) -> bool:
    """Pattern 5 idle-break gate, evaluated on the raw SDK event."""
    raw_type = getattr(event, "type", None)
    if raw_type == "session.status_terminated":
        return True
    if raw_type == "session.status_idle":
        stop = getattr(getattr(event, "stop_reason", None), "type", None)
        return stop != "requires_action"
    return False


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class Collector:
    """Start and drive halo-collector jobs on Managed Agents."""

    def __init__(
        self,
        client=None,
        *,
        agent_id: str | None = None,
        environment_id: str | None = None,
        vault_ids: list[str] | None = None,
        repo_url: str | None = None,
        repo_branch: str = "main",
        repo_token: str | None = None,
    ) -> None:
        ids = config.load_ids()
        repo = config.repo_settings()
        self._client = client
        self.agent_id = agent_id or ids.get("agent_id")
        self.environment_id = environment_id or ids.get("environment_id")
        if vault_ids is None:
            vault_ids = [ids["vault_id"]] if ids.get("vault_id") else []
        self.vault_ids = list(vault_ids)
        self.repo_url = repo_url or repo["url"]
        self.repo_branch = repo_branch or repo["branch"] or "main"
        self.repo_token = repo_token if repo_token is not None else repo["token"]

    # -- client -------------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            import anthropic  # lazy: constructing a Collector never needs a key

            self._client = anthropic.Anthropic()
        return self._client

    # -- lifecycle ----------------------------------------------------------

    def _resources(self) -> list[dict]:
        resource: dict[str, Any] = {
            "type": "github_repository",
            "url": self.repo_url,
            "mount_path": REPO_MOUNT_PATH,
            "checkout": {"type": "branch", "name": self.repo_branch},
        }
        if self.repo_token:
            resource["authorization_token"] = self.repo_token
        else:
            log.warning("mounting %s without a token (public repo only)", self.repo_url)
        return [resource]

    def start(self, spec, title: str | None = None) -> str:
        """Create a session seeded with the job's `user.define_outcome`; return its id."""
        if not self.agent_id or not self.environment_id:
            raise RuntimeError(
                "agent_id and environment_id are required: set HALO_AGENT_ID / "
                "HALO_ENVIRONMENT_ID or run `ant apply` to produce claude-lock.json"
            )
        jobs = _jobs()
        kickoff = jobs.build_kickoff(spec)
        if title is None:
            title = jobs.job_title(spec)

        kwargs: dict[str, Any] = {
            "agent": self.agent_id,
            "environment_id": self.environment_id,
            "title": title,
            "initial_events": [kickoff],
        }
        resources = self._resources()
        if resources:
            kwargs["resources"] = resources
        if self.vault_ids:
            kwargs["vault_ids"] = list(self.vault_ids)

        session = self.client.beta.sessions.create(**kwargs)
        log.info("started session %s (%s): %s", session.id, title, self.console_url(session.id))
        return session.id

    def events(
        self, session_id: str, *, include_history: bool = True, on_open=None
    ) -> Iterator[Event]:
        """Yield normalized events until the terminal gate (pattern 1 + pattern 5).

        Opens the live stream first, then (optionally) replays history via
        `events.list`, deduping by id, then tails the stream. Paused tool calls
        (`evaluated_permission == "ask"`) are yielded as `tool_ask` and NOT
        answered here; the caller decides via `confirm()`. Historical asks that
        a later `user.tool_confirmation` already answered are replayed as
        `tool_use` with `answered=True`.

        `on_open(stream)` is called as soon as the live stream is open so the
        caller can `stream.close()` from another thread (client went away); the
        generator then ends cleanly instead of raising.
        """
        seen: set[str] = set()
        api = self.client.beta.sessions.events

        with api.stream(session_id=session_id) as stream:
            if on_open:
                on_open(stream)
            try:
                if include_history:
                    history = list(api.list(session_id=session_id))
                    answered = {
                        getattr(raw, "tool_use_id", None) for raw in history
                        if getattr(raw, "type", None) == "user.tool_confirmation"
                    }
                    for raw in history:
                        event_id = getattr(raw, "id", None)
                        if event_id:
                            if event_id in seen:
                                continue
                            seen.add(event_id)
                        ev = normalize_event(raw)
                        if ev["type"] == "tool_ask" and event_id in answered:
                            ev["type"] = "tool_use"
                            ev["answered"] = True
                        yield ev
                        if is_terminal(raw):
                            return

                for raw in stream:
                    raw_type = getattr(raw, "type", None)
                    if raw_type in ("event_start", "event_delta"):
                        continue  # live previews carry no id; not requested here
                    event_id = getattr(raw, "id", None)
                    fresh = not event_id or event_id not in seen
                    if event_id:
                        seen.add(event_id)
                    if fresh:
                        yield normalize_event(raw)
                    # Gate runs on every event, deduped ones included (pattern 1).
                    if is_terminal(raw):
                        return
            except Exception:
                if _stream_closed(stream):
                    log.info("event stream for %s closed by the caller", session_id)
                    return  # closed from outside: a normal end, not an error
                raise

    def confirm(
        self, session_id: str, tool_event_id: str, allow: bool, message: str | None = None
    ) -> None:
        """Answer a paused tool call. `tool_event_id` is the agent.tool_use event id."""
        payload: dict[str, Any] = {
            "type": "user.tool_confirmation",
            "tool_use_id": tool_event_id,
            "result": "allow" if allow else "deny",
        }
        if not allow and message:
            payload["deny_message"] = message
        self.client.beta.sessions.events.send(session_id=session_id, events=[payload])

    def deny_all(self, session_id: str, tool_event_ids: list[str], message: str | None = None) -> None:
        """Deny every pending paused call (unattended runs)."""
        for tool_event_id in tool_event_ids:
            self.confirm(session_id, tool_event_id, False, message)

    def interrupt(self, session_id: str) -> None:
        self.client.beta.sessions.events.send(
            session_id=session_id, events=[{"type": "user.interrupt"}]
        )

    # -- inspection ---------------------------------------------------------

    def status(self, session_id: str) -> dict:
        session = self.client.beta.sessions.retrieve(session_id=session_id)
        evaluations = getattr(session, "outcome_evaluations", None) or []
        outcome = getattr(evaluations[-1], "result", None) if evaluations else None
        return {
            "id": getattr(session, "id", session_id),
            "status": getattr(session, "status", None),
            "outcome": outcome,
            "usage": _as_dict(getattr(session, "usage", None)),
        }

    def outputs(self, session_id: str) -> list[dict]:
        """List files the agent wrote under /mnt/session/outputs/ (with indexing-lag retries)."""
        files: list[dict] = []
        for attempt in range(OUTPUT_LIST_RETRIES + 1):
            page = self.client.beta.files.list(scope_id=session_id, betas=[MANAGED_AGENTS_BETA])
            files = [
                {
                    "id": getattr(f, "id", None),
                    "filename": getattr(f, "filename", None),
                    "size_bytes": getattr(f, "size_bytes", None),
                }
                for f in page
            ]
            if files or attempt == OUTPUT_LIST_RETRIES:
                break
            _sleep(OUTPUT_LIST_DELAY_S)
        return files

    def download(self, session_id: str, file_id: str, dest: Path) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        response = self.client.beta.files.download(file_id)
        response.write_to_file(dest)
        return dest

    def console_url(self, session_id: str) -> str:
        return f"{CONSOLE_BASE}/{session_id}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_event(ev: Event) -> str | None:
    kind = ev["type"]
    if kind == "message":
        return ev["text"] or ""
    if kind in ("tool_use", "custom_tool_use"):
        return f"[tool] {ev['tool']} ({ev['permission'] or 'n/a'})"
    if kind == "tool_ask":
        return f"[PAUSED] {ev['tool']} asks permission: {ev['input']}"
    if kind == "outcome":
        return f"[outcome] iteration {ev['iteration']}: {ev['result']} - {ev['explanation']}"
    if kind == "idle":
        return f"[idle] stop_reason={ev['stop_reason']}"
    if kind == "terminated":
        return "[terminated]"
    if kind == "usage":
        return None
    if kind == "user_event":
        return f"[user] {ev['raw_type']}"
    return None


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m collector.runtime",
        description="Start a halo-collector job and stream it to the terminal.",
    )
    parser.add_argument("--kind", required=True, choices=["collect", "validate", "instruct", "voice_discover"])
    parser.add_argument("--langs", required=True, help="comma-separated PLD codes, e.g. ceb,war")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--push", action="store_true", help="allow the agent to push to the Hub")
    parser.add_argument("--notes", default="", help="free text appended to the task")
    parser.add_argument("--deny-asks", action="store_true", help="deny paused tool calls instead of asking")
    parser.add_argument("--no-history", action="store_true", help="do not replay history before tailing")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    jobs = _jobs()

    spec_kwargs: dict[str, Any] = {
        "kind": args.kind,
        "languages": [s.strip() for s in args.langs.split(",") if s.strip()],
        "push": args.push,
        "notes": args.notes,
    }
    if args.max_docs is not None:
        spec_kwargs["max_docs"] = args.max_docs
    if args.max_sentences is not None:
        spec_kwargs["max_sentences"] = args.max_sentences
    spec = jobs.JobSpec(**spec_kwargs)
    jobs.validate_spec(spec)

    collector = Collector()
    session_id = collector.start(spec)
    print(f"session {session_id}")
    print(f"console {collector.console_url(session_id)}")

    for ev in collector.events(session_id, include_history=not args.no_history):
        line = _format_event(ev)
        if line is not None:
            print(line, flush=True)
        if ev["type"] == "tool_ask":
            allow = False
            if args.deny_asks:
                print("  -> denied (--deny-asks)")
            else:
                answer = input(f"  allow {ev['tool']}? [y/N] ").strip().lower()
                allow = answer in ("y", "yes")
                print(f"  -> {'allowed' if allow else 'denied'}")
            collector.confirm(
                session_id, ev["id"], allow,
                None if allow else "Denied by the collector runtime.",
            )

    print()
    print("status:", collector.status(session_id))
    files = collector.outputs(session_id)
    if not files:
        print("outputs: (none)")
    for f in files:
        print(f"output {f['id']}  {f['filename']}  {f['size_bytes']} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
