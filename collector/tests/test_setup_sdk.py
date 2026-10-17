"""Tests for collector.setup_sdk.apply against a recording fake client. No network."""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from collector import config, setup_sdk

ROOT = setup_sdk.ROOT
HF = "hf_secret_value_123"
TAVILY = "tvly_secret_value_456"
GH = "ghp_secret_repo_token_789"


# ---------------------------------------------------------------------------
# fake client: beta.environments / agents / vaults(.credentials) / deployments
# ---------------------------------------------------------------------------


class _Resource:
    """create/update recorder; returns objects with .id (and .version for agents)."""

    def __init__(self, client: "SetupFake", prefix: str, versioned: bool = False):
        self._c = client
        self._prefix = prefix
        self._versioned = versioned
        self.created: list[dict] = []
        self.updated: list[tuple[str, dict]] = []
        self._n = 0

    def _obj(self, obj_id: str, version: int) -> NS:
        obj = NS(id=obj_id)
        if self._versioned:
            obj.version = version
        return obj

    def create(self, **kwargs):
        self._n += 1
        self.created.append(kwargs)
        obj_id = f"{self._prefix}_{self._n:03d}"
        self._c.calls.append((f"{self._prefix}.create", obj_id))
        return self._obj(obj_id, 1)

    def update(self, obj_id: str, **kwargs):
        self.updated.append((obj_id, kwargs))
        self._c.calls.append((f"{self._prefix}.update", obj_id))
        return self._obj(obj_id, 2)


class _Credentials:
    def __init__(self, client: "SetupFake"):
        self._c = client
        self.created: list[tuple[str, dict]] = []
        self.updated: list[tuple[str, dict]] = []
        self.store: dict[str, list[NS]] = {}   # vault_id -> credential objects

    def list(self, vault_id: str, **_):
        self._c.calls.append(("credentials.list", vault_id))
        return list(self.store.get(vault_id, []))

    def create(self, vault_id: str, **kwargs):
        self.created.append((vault_id, kwargs))
        cid = f"cred_{len(self.created):03d}"
        self.store.setdefault(vault_id, []).append(
            NS(id=cid, display_name=kwargs["display_name"],
               auth=NS(secret_name=kwargs["auth"]["secret_name"])))
        self._c.calls.append(("credentials.create", cid))
        return NS(id=cid)

    def update(self, cred_id: str, **kwargs):
        self.updated.append((cred_id, kwargs))
        self._c.calls.append(("credentials.update", cred_id))
        return NS(id=cred_id)


class _Deployments(_Resource):
    def __init__(self, client):
        super().__init__(client, "dep")
        self.paused: list[str] = []

    def _obj(self, obj_id, version):
        return NS(id=obj_id, schedule=NS(upcoming_runs_at=["2026-10-18T02:00:00+08:00"]))

    def pause(self, dep_id: str, **_):
        self.paused.append(dep_id)
        self._c.calls.append(("dep.pause", dep_id))
        return NS(id=dep_id, status="paused")


class SetupFake:
    def __init__(self):
        self.calls: list[tuple] = []
        self.environments = _Resource(self, "env")
        self.agents = _Resource(self, "agent", versioned=True)
        self.vaults = _Resource(self, "vlt")
        self.vaults.credentials = _Credentials(self)
        self.deployments = _Deployments(self)
        self.beta = NS(environments=self.environments, agents=self.agents,
                       vaults=self.vaults, deployments=self.deployments)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("HF_TOKEN", "TAVILY_API_KEY", "GITHUB_TOKEN", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def secrets(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", HF)
    monkeypatch.setenv("TAVILY_API_KEY", TAVILY)


@pytest.fixture
def lock_path(tmp_path):
    return tmp_path / "claude-lock.json"


def _lock(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_fresh_lock_creates_everything(lock_path, secrets, capsys):
    fake = SetupFake()
    ids = setup_sdk.apply(lock_path, client=fake)

    assert len(fake.environments.created) == 1
    assert len(fake.agents.created) == 1
    assert len(fake.vaults.created) == 1
    assert not fake.environments.updated and not fake.agents.updated and not fake.vaults.updated
    assert not fake.deployments.created and not fake.deployments.updated

    # payloads come from the resource files
    assert fake.environments.created[0]["name"] == "halo-collector"
    agent_payload = fake.agents.created[0]
    assert agent_payload["name"] == "halo-collector"
    assert agent_payload["system"].startswith("You are halo-collector")
    assert fake.vaults.created[0] == {"display_name": "halo-collector"}
    assert "type" not in fake.vaults.created[0]

    # two credentials, both in the new vault, with the right hosts
    assert [v for v, _ in fake.vaults.credentials.created] == ["vlt_001", "vlt_001"]
    by_name = {c["auth"]["secret_name"]: c for _, c in fake.vaults.credentials.created}
    assert set(by_name) == {"HF_TOKEN", "TAVILY_API_KEY"}
    assert by_name["HF_TOKEN"]["auth"]["secret_value"] == HF
    assert by_name["HF_TOKEN"]["auth"]["networking"]["allowed_hosts"] == \
        ["huggingface.co", "*.huggingface.co"]
    assert by_name["TAVILY_API_KEY"]["auth"]["secret_value"] == TAVILY
    assert by_name["TAVILY_API_KEY"]["auth"]["networking"]["allowed_hosts"] == ["api.tavily.com"]
    assert not fake.vaults.credentials.updated

    # returned ids and lock file in the ant-apply shape
    assert ids == {"agent_id": "agent_001", "environment_id": "env_001",
                   "vault_id": "vlt_001", "deployment_id": None}
    lock = _lock(lock_path)
    assert lock["version"] == 1
    assert lock["resources"] == {
        "./environments/halo-collector.yaml": {"type": "environment", "id": "env_001"},
        "./agents/halo-collector.md": {"type": "agent", "id": "agent_001", "version": 1},
        "./vaults/halo-collector.yaml": {"type": "vault", "id": "vlt_001"},
    }
    # and collector.config reads it back
    assert config.load_ids(lock_path) == {
        "agent_id": "agent_001", "environment_id": "env_001", "vault_id": "vlt_001"}
    assert "warning" not in capsys.readouterr().err


def test_second_apply_updates_instead_of_creating(lock_path, secrets):
    fake = SetupFake()
    first = setup_sdk.apply(lock_path, client=fake)
    second = setup_sdk.apply(lock_path, client=fake)

    assert second == first
    assert len(fake.environments.created) == 1
    assert len(fake.agents.created) == 1
    assert len(fake.vaults.created) == 1
    assert [i for i, _ in fake.environments.updated] == ["env_001"]
    assert [i for i, _ in fake.agents.updated] == ["agent_001"]
    assert [i for i, _ in fake.vaults.updated] == ["vlt_001"]
    assert fake.environments.updated[0][1] == fake.environments.created[0]
    assert fake.agents.updated[0][1] == fake.agents.created[0]

    # credentials: updated in place, not duplicated
    assert len(fake.vaults.credentials.created) == 2
    assert len(fake.vaults.credentials.store["vlt_001"]) == 2
    updated = {cid: kw for cid, kw in fake.vaults.credentials.updated}
    assert set(updated) == {"cred_001", "cred_002"}
    for kw in updated.values():
        assert kw["vault_id"] == "vlt_001"
        assert kw["auth"]["type"] == "environment_variable"
        assert kw["auth"]["secret_value"] in (HF, TAVILY)
        assert kw["display_name"] in ("HF_TOKEN", "TAVILY_API_KEY")

    # agent version follows the update
    assert _lock(lock_path)["resources"]["./agents/halo-collector.md"]["version"] == 2


def test_unset_credential_is_skipped_with_warning(lock_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", HF)   # TAVILY_API_KEY stays unset
    fake = SetupFake()
    setup_sdk.apply(lock_path, client=fake)

    names = [c["auth"]["secret_name"] for _, c in fake.vaults.credentials.created]
    assert names == ["HF_TOKEN"]
    assert not fake.vaults.credentials.updated
    err = capsys.readouterr().err
    assert "warning: TAVILY_API_KEY is not set" in err
    assert "HF_TOKEN is not set" not in err
    # the rest of the setup still happened
    assert len(fake.vaults.created) == 1
    assert _lock(lock_path)["resources"]["./vaults/halo-collector.yaml"]["id"] == "vlt_001"


def test_deployment_create_resolves_refs_and_injects_github_token(lock_path, secrets, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", GH)
    fake = SetupFake()
    ids = setup_sdk.apply(lock_path, deployment=True, client=fake)

    assert ids["deployment_id"] == "dep_001"
    assert len(fake.deployments.created) == 1 and not fake.deployments.updated
    payload = fake.deployments.created[0]
    assert payload["agent"] == "agent_001"
    assert payload["environment_id"] == "env_001"
    assert payload["vault_ids"] == ["vlt_001"]
    assert payload["name"] == "halo-collector nightly collect"
    assert payload["schedule"]["type"] == "cron"
    assert payload["initial_events"][0]["type"] == "user.define_outcome"

    repo = [r for r in payload["resources"] if r["type"] == "github_repository"]
    assert len(repo) == 1
    assert repo[0]["url"] == setup_sdk.REPO_URL
    assert repo[0]["mount_path"] == setup_sdk.REPO_MOUNT
    assert repo[0]["authorization_token"] == GH

    # created paused, until a manual run looks right
    assert fake.deployments.paused == ["dep_001"]

    # the deployment is recorded in the lock, but the token is not
    lock = _lock(lock_path)
    assert lock["resources"]["./deployments/halo-collector-nightly.md"] == \
        {"type": "deployment", "id": "dep_001"}
    assert GH not in lock_path.read_text(encoding="utf-8")
    dep_md = (ROOT / "deployments" / "halo-collector-nightly.md").read_text(encoding="utf-8")
    assert GH not in dep_md
    fm, _ = setup_sdk.parse_frontmatter(dep_md)
    assert all("authorization_token" not in r for r in fm["resources"])


def test_deployment_second_apply_updates_and_does_not_pause_again(lock_path, secrets, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", GH)
    fake = SetupFake()
    setup_sdk.apply(lock_path, deployment=True, client=fake)
    ids = setup_sdk.apply(lock_path, deployment=True, client=fake)

    assert ids["deployment_id"] == "dep_001"
    assert len(fake.deployments.created) == 1
    assert [i for i, _ in fake.deployments.updated] == ["dep_001"]
    assert fake.deployments.updated[0][1]["vault_ids"] == ["vlt_001"]
    assert fake.deployments.updated[0][1]["resources"][0]["authorization_token"] == GH
    assert fake.deployments.paused == ["dep_001"]


def test_deployment_without_github_token_warns_and_omits_it(lock_path, secrets, capsys):
    fake = SetupFake()
    setup_sdk.apply(lock_path, deployment=True, client=fake)

    repo = fake.deployments.created[0]["resources"][0]
    assert repo["type"] == "github_repository"
    assert "authorization_token" not in repo
    assert "warning: GITHUB_TOKEN not set" in capsys.readouterr().err


def test_secrets_never_reach_the_lock_file(lock_path, secrets, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", GH)
    fake = SetupFake()
    setup_sdk.apply(lock_path, deployment=True, client=fake)
    setup_sdk.apply(lock_path, deployment=True, client=fake)

    text = lock_path.read_text(encoding="utf-8")
    for secret in (HF, TAVILY, GH):
        assert secret not in text
    for word in ("secret_value", "authorization_token", "HF_TOKEN", "TAVILY_API_KEY"):
        assert word not in text
    # only ids/types/versions live in the lock
    for entry in _lock(lock_path)["resources"].values():
        assert set(entry) <= {"type", "id", "version"}
