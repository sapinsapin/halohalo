"""Tests for collector.jobs and the resource files setup_sdk reads. No network."""

from datetime import date
from pathlib import Path

import pytest

from collector import setup_sdk
from collector.jobs import (
    JOB_KINDS, LANGUAGES, MAX_ITERATIONS_CAP, JobSpec, build_kickoff, job_title, validate_spec,
)

ROOT = Path(__file__).resolve().parents[2]


# -- validate_spec -----------------------------------------------------------

def test_languages_are_the_ten_pld_codes():
    assert LANGUAGES == ["bcl", "ceb", "eng", "fil", "hil", "ilo", "pag", "pam", "tsg", "war"]


@pytest.mark.parametrize("kind", JOB_KINDS)
def test_validate_spec_accepts_defaults_for_every_kind(kind):
    validate_spec(JobSpec(kind, ["ceb"]))


@pytest.mark.parametrize("spec, msg", [
    (JobSpec("scrape", ["ceb"]), "unknown job kind"),
    (JobSpec("collect", []), "non-empty"),
    (JobSpec("collect", ["xx"]), "unknown language"),
    (JobSpec("collect", ["ceb", "ceb"]), "duplicates"),
    (JobSpec("collect", ["ceb"], max_docs=0), "max_docs"),
    (JobSpec("validate", ["ceb"], max_sentences=-1), "max_sentences"),
    (JobSpec("collect", ["ceb"], max_iterations=0), "max_iterations"),
    (JobSpec("collect", ["ceb"], max_iterations=MAX_ITERATIONS_CAP + 1), "max_iterations"),
    (JobSpec("validate", ["ceb"], push=True), "push is only meaningful"),
    (JobSpec("collect", ["ceb"], push="yes"), "push must be a bool"),
])
def test_validate_spec_rejects(spec, msg):
    with pytest.raises(ValueError, match=msg):
        validate_spec(spec)


# -- build_kickoff -----------------------------------------------------------

@pytest.mark.parametrize("kind", JOB_KINDS)
def test_build_kickoff_shape(kind):
    ev = build_kickoff(JobSpec(kind, ["ceb", "war"], max_iterations=6))
    assert set(ev) == {"type", "description", "rubric", "max_iterations"}
    assert ev["type"] == "user.define_outcome"
    assert ev["rubric"]["type"] == "text"
    assert ev["rubric"]["content"].strip()
    assert ev["max_iterations"] == 6
    assert f"Job kind: {kind}" in ev["description"]
    assert "ceb, war" in ev["description"]


@pytest.mark.parametrize("kind", JOB_KINDS)
def test_rubric_has_6_to_10_numbered_criteria(kind):
    content = build_kickoff(JobSpec(kind, LANGUAGES))["rubric"]["content"]
    numbered = [l for l in content.splitlines() if l[:2].rstrip(".").isdigit() or l[:3].rstrip(".").isdigit()]
    assert 6 <= len(numbered) <= 10, numbered
    assert "report.md" in content
    assert "Not done" in content


def test_rubric_enforces_output_contract():
    collect = build_kickoff(JobSpec("collect", ["ceb"], max_docs=42))["rubric"]["content"]
    assert "text/{lang}/shard-*.parquet" in collect
    assert "manifest.jsonl" in collect
    assert "content_hash" in collect and "lid_detail" in collect
    assert "25%" in collect and "42 accepted" in collect
    assert "Nothing was pushed" in collect

    validate = build_kickoff(JobSpec("validate", ["pag"], max_sentences=7))["rubric"]["content"]
    assert "validated/{lang}.jsonl" in validate
    assert "[0, 1]" in validate and 'validator` is "model"' in validate
    assert "more than 7 rows" in validate and "pag, pam and tsg" in validate

    instruct = build_kickoff(JobSpec("instruct", ["ilo"]))["rubric"]["content"]
    assert "instruct/{lang}.jsonl" in instruct
    for t in ("qa", "summarize", "translate_en_x", "translate_x_en", "continue", "classify"):
        assert t in instruct
    assert "license" in instruct and "source_row" in instruct

    voice = build_kickoff(JobSpec("voice_discover", ["hil"]))["rubric"]["content"]
    assert "voice/{lang}.jsonl" in voice
    assert "Creative Commons" in voice and "duration_s" in voice
    assert "No audio file" in voice


def test_push_flag_changes_description_and_rubric():
    on = build_kickoff(JobSpec("collect", ["ceb"], push=True))
    off = build_kickoff(JobSpec("collect", ["ceb"], push=False))
    assert "Hub push allowed: yes" in on["description"]
    assert "Hub push allowed: no" in off["description"]
    assert "manifest shows the run finished" in on["rubric"]["content"]
    assert "Nothing was pushed" in off["rubric"]["content"]


def test_notes_are_appended():
    ev = build_kickoff(JobSpec("collect", ["ceb"], notes="use --queries-per-lang 5"))
    assert ev["description"].endswith("Notes from the operator: use --queries-per-lang 5")
    assert "Notes from the operator" not in build_kickoff(JobSpec("collect", ["ceb"]))["description"]


def test_build_kickoff_validates():
    with pytest.raises(ValueError):
        build_kickoff(JobSpec("collect", []))


# -- job_title ---------------------------------------------------------------

def test_job_title():
    assert job_title(JobSpec("collect", ["ceb", "war"]), today=date(2026, 10, 17)) == \
        "collect · ceb,war · 2026-10-17"
    assert job_title(JobSpec("validate", ["tsg"])).startswith("validate · tsg · ")


# -- resource files ----------------------------------------------------------

def test_parse_frontmatter_roundtrip():
    fm, body = setup_sdk.parse_frontmatter("---\nname: x\nmodel: {id: m}\n---\n\nhello\n")
    assert fm == {"name": "x", "model": {"id": "m"}}
    assert body == "hello\n"
    assert setup_sdk.parse_frontmatter("plain") == ({}, "plain")
    with pytest.raises(ValueError):
        setup_sdk.parse_frontmatter("---\nname: x\n")


def test_agent_file_disables_both_web_tools_and_uses_auto_policy():
    agent = setup_sdk.load_agent(ROOT / "agents" / "halo-collector.md")
    assert agent["name"] == "halo-collector"
    assert agent["model"] == {"id": "claude-opus-5-5", "effort": "high"}
    assert "mcp_servers" not in agent and "skills" not in agent
    (toolset,) = agent["tools"]
    assert toolset["type"] == "agent_toolset_20260401"
    assert toolset["default_config"]["permission_policy"] == {"type": "auto"}
    configs = {c["name"]: c for c in toolset["configs"]}
    assert configs["web_fetch"]["enabled"] is False
    assert configs["web_search"]["enabled"] is False
    system = agent["system"]
    assert "/mnt/repo" in system and "/mnt/session/outputs" in system
    for kind in JOB_KINDS:
        assert kind in system
    assert "--no-glotlid" in system and "--push" in system
    assert 80 <= system.count("\n") <= 160


def test_environment_file_matches_spec():
    env = setup_sdk.load_yaml(ROOT / "environments" / "halo-collector.yaml")
    assert env["name"] == "halo-collector"
    net = env["config"]["networking"]
    assert env["config"]["type"] == "cloud"
    assert net["type"] == "limited" and net["allow_package_managers"] is True
    assert net["allowed_hosts"] == ["huggingface.co", "*.huggingface.co", "api.tavily.com"]


def test_vault_file_is_container_only():
    vault = setup_sdk.load_yaml(ROOT / "vaults" / "halo-collector.yaml")
    assert vault == {"display_name": "halo-collector"}


def test_deployment_file_is_in_sync_with_jobs():
    path = ROOT / "deployments" / "halo-collector-nightly.md"
    assert path.read_text(encoding="utf-8") == setup_sdk.render_deployment(), \
        "run: python -m collector.setup_sdk --write-deployment"
    dep = setup_sdk.load_deployment(path)
    assert dep["agent"] == "../agents/halo-collector.md"
    assert dep["environment_id"] == "../environments/halo-collector.yaml"
    assert dep["schedule"] == {"type": "cron", "expression": "0 2 * * *", "timezone": "Asia/Manila"}
    (event,) = dep["initial_events"]
    assert event == build_kickoff(JobSpec("collect", LANGUAGES, max_docs=300, push=False, max_iterations=4))
    (repo,) = dep["resources"]
    assert repo["type"] == "github_repository" and repo["mount_path"] == "/mnt/repo"
    assert "authorization_token" not in repo
    assert "GITHUB_TOKEN" not in path.read_text(encoding="utf-8").split("---", 2)[2]


def test_resolve_refs_fills_ids_and_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    ids = {setup_sdk.AGENT_FILE: "agent_1", setup_sdk.ENV_FILE: "env_1", setup_sdk.VAULT_FILE: "vlt_1"}
    dep = setup_sdk.load_deployment()
    dep["vault_ids"] = ["../vaults/halo-collector.yaml"]
    out = setup_sdk._resolve_refs(dep, ids)
    assert out["agent"] == "agent_1" and out["environment_id"] == "env_1"
    assert out["vault_ids"] == ["vlt_1"]
    assert out["resources"][0]["authorization_token"] == "ghp_test"


def test_dry_run_needs_no_api_key(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert setup_sdk.main(["--dry-run", "--deployment", "--lock", str(tmp_path / "lock.json")]) == 0
    out = capsys.readouterr()
    assert "agents.create" in out.out and "deployments.create" in out.out
    assert "hf_secret_value" not in out.out
    assert "TAVILY_API_KEY is not set" in out.err
    assert not (tmp_path / "lock.json").exists()
