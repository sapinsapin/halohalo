"""
ONE-TIME SETUP fallback for the halo-collector Managed Agent when `ant` is not
available. The preferred path is `ant apply` (docs/collector.md); this script
does the same four things with the Python SDK and writes claude-lock.json in
the same shape, so collector/config.py reads either:

  python -m collector.setup_sdk --dry-run        # print the payloads, call nothing
  python -m collector.setup_sdk                  # environment, agent, vault, credentials
  python -m collector.setup_sdk --deployment     # ... plus the nightly deployment

Resource definitions come from the files `ant apply` uses (agents/halo-collector.md,
environments/halo-collector.yaml, vaults/halo-collector.yaml,
deployments/halo-collector-nightly.md) so there is one source of truth.
Credentials come from the environment (HF_TOKEN, TAVILY_API_KEY) and are never
written to disk. Idempotent: an ID already in claude-lock.json is updated, not
recreated. Needs ANTHROPIC_API_KEY (or an `ant auth login` profile) unless --dry-run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "claude-lock.json"

AGENT_FILE = "./agents/halo-collector.md"
ENV_FILE = "./environments/halo-collector.yaml"
VAULT_FILE = "./vaults/halo-collector.yaml"
DEPLOYMENT_FILE = "./deployments/halo-collector-nightly.md"

REPO_URL = "https://github.com/sapinsapin/halohalo"
REPO_MOUNT = "/mnt/repo"

# secret env var -> hosts the vault may substitute it for (SPEC.md)
CREDENTIALS = {
    "HF_TOKEN": ["huggingface.co", "*.huggingface.co"],
    "TAVILY_API_KEY": ["api.tavily.com"],
}


# ----------------------------------------------------------------------------
# file formats
# ----------------------------------------------------------------------------

def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a `---` YAML frontmatter block from a Markdown body.

    Returns (frontmatter dict, body). A file without frontmatter is ({}, text).
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines(keepends=True)
    if lines[0].strip() != "---":
        return {}, text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            fm = yaml.safe_load("".join(lines[1:i])) or {}
            body = "".join(lines[i + 1:])
            return fm, body.lstrip("\n")
    raise ValueError("frontmatter opened with --- but never closed")


def load_agent(path: Path = ROOT / AGENT_FILE) -> dict:
    """agents.create body from agents/halo-collector.md (body -> `system`)."""
    fm, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    for key in ("name", "model"):
        if key not in fm:
            raise ValueError(f"{path}: frontmatter needs `{key}`")
    payload = dict(fm)
    payload["system"] = body.strip() + "\n"
    return payload


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_deployment(path: Path = ROOT / DEPLOYMENT_FILE) -> dict:
    """deployments.create body from the .md: frontmatter, body appended as a
    user.message only when non-empty (ours is empty: the outcome is in frontmatter)."""
    fm, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    payload = dict(fm)
    if body.strip():
        payload.setdefault("initial_events", []).append(
            {"type": "user.message", "content": [{"type": "text", "text": body.strip()}]})
    return payload


def _resolve_refs(payload: dict, ids: dict) -> dict:
    """Replace the file-path references a deployment file uses (`agent:
    ../agents/x.md`) with the IDs claude-lock.json holds, as `ant apply` does."""
    out = dict(payload)
    agent = out.get("agent")
    if isinstance(agent, str) and agent.endswith(".md"):
        out["agent"] = ids[AGENT_FILE]
    env = out.get("environment_id")
    if isinstance(env, str) and env.endswith((".yaml", ".yml")):
        out["environment_id"] = ids[ENV_FILE]
    vault_ids = []
    for v in out.get("vault_ids", []) or []:
        if isinstance(v, str) and v.endswith((".yaml", ".yml")):
            v = ids.get(VAULT_FILE)
        if v:
            vault_ids.append(v)
    if not vault_ids and ids.get(VAULT_FILE):
        # the file names no vault (or one we could not resolve): the run needs the
        # credentials, so fall back to the vault this setup created
        vault_ids = [ids[VAULT_FILE]]
    if vault_ids:
        out["vault_ids"] = vault_ids
    # the repo token must never sit in a file; inject it from the environment
    for res in out.get("resources", []) or []:
        if res.get("type") == "github_repository" and "authorization_token" not in res:
            tok = os.environ.get("GITHUB_TOKEN")
            if tok:
                res["authorization_token"] = tok
    return out


DEPLOYMENT_HEADER = """\
---
# deployments/halo-collector-nightly.md - OPTIONAL nightly flywheel deployment.
# Frontmatter = deployments.create body; the Markdown body is intentionally empty
# because the kickoff is the user.define_outcome below. Its text is generated from
# collector.jobs.build_kickoff(JobSpec("collect", LANGUAGES)) by
#   python -m collector.setup_sdk --write-deployment
# and collector/tests/test_jobs.py fails if the two drift. Do not hand-edit the event.
#
# Before applying, wire two things that cannot live in this file:
#   1. vault_ids: `python -m collector.setup_sdk --deployment` resolves the
#      ../vaults/halo-collector.yaml reference below to the vlt_... ID in
#      claude-lock.json (and falls back to that ID when the list is empty). For
#      `ant apply`, which sends vault_ids as written, paste the vlt_... ID instead.
#      Without the vault the run has no TAVILY_API_KEY/HF_TOKEN and collects nothing.
#   2. The repo token: `authorization_token` is omitted on the github_repository
#      resource. That is fine while sapinsapin/halohalo is public. If it is private,
#      do NOT paste the token here - run `python -m collector.setup_sdk --deployment`,
#      which injects GITHUB_TOKEN from the environment at create time.
# Fires 02:00 Asia/Manila (no DST, so the 1-3 AM cron caveat does not apply).
# Runs are unattended: paused tool calls are never answered, so bash stays on `auto`.
# Apply and pause in one command, then test with a manual run (docs/collector.md).
"""


class _Dumper(yaml.SafeDumper):
    pass


def _str_presenter(dumper, s):
    style = "|" if "\n" in s else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", s, style=style)


_Dumper.add_representer(str, _str_presenter)


def render_deployment() -> str:
    """The text of deployments/halo-collector-nightly.md, built from collector.jobs."""
    from .jobs import LANGUAGES, JobSpec, build_kickoff
    event = build_kickoff(JobSpec("collect", list(LANGUAGES), max_docs=300, push=False,
                                  max_iterations=4))
    body = {
        "name": "halo-collector nightly collect",
        "description": ("Nightly seeded scrape for all ten PLD languages; stages shards under "
                        "/mnt/session/outputs, never pushes."),
        "agent": "../agents/halo-collector.md",
        "environment_id": "../environments/halo-collector.yaml",
        "vault_ids": ["../vaults/halo-collector.yaml"],
        "resources": [{
            "type": "github_repository",
            "url": REPO_URL,
            "mount_path": REPO_MOUNT,
            "checkout": {"type": "branch", "name": "main"},
        }],
        "schedule": {"type": "cron", "expression": "0 2 * * *", "timezone": "Asia/Manila"},
        "initial_events": [event],
    }
    text = yaml.dump(body, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100)
    return DEPLOYMENT_HEADER + text + "---\n"


# ----------------------------------------------------------------------------
# lock file (same shape as `ant apply`)
# ----------------------------------------------------------------------------

def read_lock(path: Path = LOCK) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"version": 1, "resources": {}}


def write_lock(lock: dict, path: Path = LOCK) -> None:
    path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")


def lock_id(lock: dict, key: str) -> str | None:
    entry = lock.get("resources", {}).get(key)
    return entry.get("id") if isinstance(entry, dict) else None


def set_lock(lock: dict, key: str, kind: str, obj_id: str, **extra) -> None:
    lock.setdefault("resources", {})[key] = {"type": kind, "id": obj_id, **extra}


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def _print_payload(title: str, payload: dict) -> None:
    safe = json.loads(json.dumps(payload))
    for res in safe.get("resources", []) or []:
        if "authorization_token" in res:
            res["authorization_token"] = "<redacted>"
    if "auth" in safe and "secret_value" in safe["auth"]:
        safe["auth"]["secret_value"] = "<redacted>"
    print(f"\n== {title}\n{json.dumps(safe, indent=2, ensure_ascii=False)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="one-time setup of the halo-collector agent via the SDK")
    ap.add_argument("--dry-run", action="store_true", help="print payloads, call nothing")
    ap.add_argument("--deployment", action="store_true",
                    help="also create/update the nightly deployment from deployments/halo-collector-nightly.md")
    ap.add_argument("--lock", type=Path, default=LOCK, help="claude-lock.json path")
    ap.add_argument("--write-deployment", action="store_true",
                    help="regenerate deployments/halo-collector-nightly.md from collector.jobs and exit")
    args = ap.parse_args(argv)

    if args.write_deployment:
        path = ROOT / DEPLOYMENT_FILE
        path.write_text(render_deployment(), encoding="utf-8")
        print(f"wrote {path}")
        return 0

    env_payload = load_yaml(ROOT / ENV_FILE)
    agent_payload = load_agent()
    vault_payload = load_yaml(ROOT / VAULT_FILE)
    vault_payload.pop("type", None)          # `type: vault` is a file marker, not an API field
    lock = read_lock(args.lock)

    creds = []
    for name, hosts in CREDENTIALS.items():
        value = os.environ.get(name)
        if not value:
            print(f"warning: {name} is not set; skipping that credential "
                  f"(add it later with `ant beta:vaults:credentials create` or rerun with it set)",
                  file=sys.stderr)
            continue
        creds.append({
            "display_name": name,
            "auth": {
                "type": "environment_variable",
                "secret_name": name,
                "secret_value": value,
                "networking": {"type": "limited", "allowed_hosts": hosts},
                "injection_location": {"header": True, "body": True},
            },
        })

    if args.dry_run:
        _print_payload(f"environments.create  ({ENV_FILE})", env_payload)
        _print_payload(f"agents.create  ({AGENT_FILE})", agent_payload)
        _print_payload(f"vaults.create  ({VAULT_FILE})", vault_payload)
        for c in creds:
            _print_payload(f"vaults.credentials.create  ({c['display_name']})", c)
        if args.deployment:
            fake = {AGENT_FILE: "agent_...", ENV_FILE: "env_...", VAULT_FILE: "vlt_..."}
            fake.update({k: v["id"] for k, v in lock.get("resources", {}).items()})
            _print_payload(f"deployments.create  ({DEPLOYMENT_FILE})",
                           _resolve_refs(load_deployment(), fake))
        print("\n(dry run: nothing created; claude-lock.json untouched)")
        return 0

    import anthropic  # lazy: --dry-run must work without a key
    client = anthropic.Anthropic()

    # environment
    env_id = lock_id(lock, ENV_FILE)
    if env_id:
        env = client.beta.environments.update(env_id, **env_payload)
        print(f"updated environment {env.id}")
    else:
        env = client.beta.environments.create(**env_payload)
        print(f"created environment {env.id}")
    set_lock(lock, ENV_FILE, "environment", env.id)
    write_lock(lock, args.lock)

    # agent
    agent_id = lock_id(lock, AGENT_FILE)
    if agent_id:
        agent = client.beta.agents.update(agent_id, **agent_payload)
        print(f"updated agent {agent.id} -> version {agent.version}")
    else:
        agent = client.beta.agents.create(**agent_payload)
        print(f"created agent {agent.id} version {agent.version}")
    set_lock(lock, AGENT_FILE, "agent", agent.id, version=agent.version)
    write_lock(lock, args.lock)

    # vault
    vault_id = lock_id(lock, VAULT_FILE)
    if vault_id:
        vault = client.beta.vaults.update(vault_id, **vault_payload)
        print(f"updated vault {vault.id}")
    else:
        vault = client.beta.vaults.create(**vault_payload)
        print(f"created vault {vault.id}")
    set_lock(lock, VAULT_FILE, "vault", vault.id)
    write_lock(lock, args.lock)

    # credentials: secret_name is unique per vault, so update in place when present
    existing = {}
    for c in client.beta.vaults.credentials.list(vault.id):
        auth = getattr(c, "auth", None)
        secret_name = getattr(auth, "secret_name", None)
        if secret_name:
            existing[secret_name] = c.id
    for c in creds:
        name = c["auth"]["secret_name"]
        if name in existing:
            client.beta.vaults.credentials.update(
                existing[name], vault_id=vault.id,
                auth={"type": "environment_variable", "secret_value": c["auth"]["secret_value"]},
                display_name=c["display_name"])
            print(f"updated credential {name}")
        else:
            client.beta.vaults.credentials.create(vault.id, **c)
            print(f"created credential {name}")

    # deployment (optional)
    if args.deployment:
        ids = {AGENT_FILE: agent.id, ENV_FILE: env.id, VAULT_FILE: vault.id}
        dep_payload = _resolve_refs(load_deployment(), ids)
        if not os.environ.get("GITHUB_TOKEN"):
            print("warning: GITHUB_TOKEN not set; the deployment mounts the repo without a token "
                  "(works only if the repository is public)", file=sys.stderr)
        dep_id = lock_id(lock, DEPLOYMENT_FILE)
        if dep_id:
            dep = client.beta.deployments.update(dep_id, **dep_payload)
            print(f"updated deployment {dep.id}")
        else:
            dep = client.beta.deployments.create(**dep_payload)
            print(f"created deployment {dep.id}; pausing it until a manual run looks right")
            client.beta.deployments.pause(dep.id)
        set_lock(lock, DEPLOYMENT_FILE, "deployment", dep.id)
        write_lock(lock, args.lock)
        print(f"next runs: {getattr(dep.schedule, 'upcoming_runs_at', None)}")

    print(f"\nwrote {args.lock}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
