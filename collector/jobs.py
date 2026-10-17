"""
Job specs for the halo-collector Managed Agent.

A job is one `user.define_outcome` event: a description paragraph the agent
works toward and a rubric the outcome grader scores independently, criterion
by criterion. Pure Python, no SDK import - the web app, the CLI runtime and
the nightly deployment all build their kickoff from here so the output
contract (collector/SPEC.md) is written down exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, get_args

JobKind = Literal["collect", "validate", "instruct", "voice_discover"]
JOB_KINDS: tuple[str, ...] = get_args(JobKind)

# PLD codes; eng = Philippine English. Same order as halolib.lid.LANGS.
LANGUAGES: list[str] = ["bcl", "ceb", "eng", "fil", "hil", "ilo", "pag", "pam", "tsg", "war"]

OUTPUTS = "/mnt/session/outputs"
MAX_ITERATIONS_CAP = 20          # API maximum for user.define_outcome
HOST_SHARE_CAP = 0.25            # no single host may supply more than this share of a language's new rows


@dataclass
class JobSpec:
    kind: JobKind
    languages: list[str] = field(default_factory=list)   # subset of LANGUAGES, non-empty
    max_docs: int = 300             # collect: accepted docs per language
    max_sentences: int = 500        # validate / instruct: rows per language
    push: bool = False              # allow Hub push
    notes: str = ""                 # free text appended to the task description
    max_iterations: int = 4         # outcome grader iterations


def validate_spec(spec: JobSpec) -> None:
    """Raise ValueError on anything the agent could not act on."""
    if spec.kind not in JOB_KINDS:
        raise ValueError(f"unknown job kind {spec.kind!r}; one of {', '.join(JOB_KINDS)}")
    if not isinstance(spec.languages, (list, tuple)) or not spec.languages:
        raise ValueError("languages must be a non-empty list of PLD codes")
    bad = [l for l in spec.languages if l not in LANGUAGES]
    if bad:
        raise ValueError(f"unknown language code(s) {bad}; valid: {LANGUAGES}")
    if len(set(spec.languages)) != len(spec.languages):
        raise ValueError("languages contains duplicates")
    if not isinstance(spec.max_docs, int) or isinstance(spec.max_docs, bool) or spec.max_docs < 1:
        raise ValueError("max_docs must be a positive integer")
    if (not isinstance(spec.max_sentences, int) or isinstance(spec.max_sentences, bool)
            or spec.max_sentences < 1):
        raise ValueError("max_sentences must be a positive integer")
    if (not isinstance(spec.max_iterations, int) or isinstance(spec.max_iterations, bool)
            or not 1 <= spec.max_iterations <= MAX_ITERATIONS_CAP):
        raise ValueError(f"max_iterations must be an integer in 1..{MAX_ITERATIONS_CAP}")
    if not isinstance(spec.push, bool):
        raise ValueError("push must be a bool")
    if spec.push and spec.kind != "collect":
        raise ValueError("push is only meaningful for collect jobs (shards -> sapinsapin/halo-{lang})")
    if not isinstance(spec.notes, str):
        raise ValueError("notes must be a string")


def job_title(spec: JobSpec, today: date | None = None) -> str:
    """e.g. "collect · ceb,war · 2026-10-17"."""
    d = (today or date.today()).isoformat()
    return f"{spec.kind} · {','.join(spec.languages)} · {d}"


# ----------------------------------------------------------------------------
# descriptions
# ----------------------------------------------------------------------------

def _description(spec: JobSpec) -> str:
    langs = ", ".join(spec.languages)
    push = "yes" if spec.push else "no"
    head = {
        "collect": (
            f"Job kind: collect. Languages: {langs}. Target: up to {spec.max_docs} accepted "
            f"documents per language. Hub push allowed: {push}.\n\n"
            f"Run the seeded web scrape (`scrape_web.py`) from the halohalo repo mounted at "
            f"/mnt/repo for these languages and stage the result under {OUTPUTS}/: parquet shards "
            f"per language at text/{{lang}}/shard-*.parquet, a combined manifest.jsonl with one "
            f"line per URL examined (accepted or rejected, with the reason), and report.md. "
            f"Every accepted row must carry its url, content_hash and both LID verdicts (HaloLID "
            f"and GlotLID, or an explicit note in report.md that GlotLID was skipped). "
            + ("Push shards to sapinsapin/halo-{lang} only after the manifest shows the run "
               "finished for that language." if spec.push else
               "Do NOT push anything to the Hub.")
        ),
        "validate": (
            f"Job kind: validate. Languages: {langs}. Target: up to {spec.max_sentences} judged "
            f"sentences per language. Hub push allowed: {push}.\n\n"
            f"Read sentences from the newest scrape shards available (the mount's scrape dir, "
            f"else `sapinsapin/halo-{{lang}}` via `datasets`) and judge each one yourself on "
            f"language identity, naturalness (not machine-translated), completeness, and "
            f"safety/PII. Write validated/{{lang}}.jsonl under {OUTPUTS}/ with one row per "
            f"sentence: text, language, url, content_hash, verdict (ok|reject), reason, "
            f"confidence (0-1), validator \"model\". Then write report.md."
        ),
        "instruct": (
            f"Job kind: instruct. Languages: {langs}. Target: up to {spec.max_sentences} "
            f"instruction pairs per language. Hub push allowed: {push}.\n\n"
            f"Generate in-language instruction-tuning pairs from validated ok rows only "
            f"(validated/{{lang}}.jsonl in the mount or a previous job's outputs; if none exist, "
            f"validate first and say so). Cover the task types qa, summarize, translate_en_x, "
            f"translate_x_en, continue and classify. Write instruct/{{lang}}.jsonl under "
            f"{OUTPUTS}/ with rows: instruction, input, output, language, task_type, source_row, "
            f"license. Then write report.md."
        ),
        "voice_discover": (
            f"Job kind: voice_discover. Languages: {langs}. Hub push allowed: {push}.\n\n"
            f"Run `scrape_web.py --voice` (no --download-audio) for these languages and write "
            f"voice/{{lang}}.jsonl under {OUTPUTS}/ listing only Creative-Commons-licensed "
            f"YouTube videos whose spoken language matched: video_id, url, title, license, "
            f"duration_s, channel. Then write report.md. No audio is downloaded and no ASR "
            f"runs in this job."
        ),
    }[spec.kind]
    if spec.notes.strip():
        head += f"\n\nNotes from the operator: {spec.notes.strip()}"
    return head


# ----------------------------------------------------------------------------
# rubrics - explicit, independently gradeable criteria
# ----------------------------------------------------------------------------

def _common_tail(spec: JobSpec) -> list[str]:
    return [
        f"`{OUTPUTS}/report.md` exists with (a) a per-language table of per-stage counts "
        f"(queries, hits, fetched/raw_fallback, accepted, rejected), (b) rejects broken down by "
        f"reason, (c) the top hosts per language with counts, and (d) a section titled "
        f"\"Not done\" that lists "
        f"anything the task asked for that was not completed with the concrete reason "
        f"(missing credential, blocked host, empty source, time), or states \"nothing\".",
        "No file under `/mnt/session/outputs/` contains an API key, token or other secret, "
        "and no row was fabricated: every url/video_id in the outputs came from the pipeline "
        "or the named source dataset.",
    ]


def _rubric(spec: JobSpec) -> str:
    langs = ", ".join(spec.languages)
    pct = int(HOST_SHARE_CAP * 100)
    if spec.kind == "collect":
        push_rule = (
            "Shards were pushed to `sapinsapin/halo-{lang}` only for languages whose "
            "manifest shows the run finished, and report.md names which languages were pushed "
            "and how many new rows each."
            if spec.push else
            "Nothing was pushed to the Hugging Face Hub; report.md states that push was not "
            "allowed for this job."
        )
        items = [
            f"For every requested language ({langs}) at least one "
            f"`{OUTPUTS}/text/{{lang}}/shard-*.parquet` file exists, or report.md explains "
            f"per language why zero documents were accepted.",
            "Every shard row has non-empty `text`, `url`, `content_hash`, `language` and "
            "`lid_detail` columns, and `language` equals the directory's language code.",
            "Every row's `lid_detail` carries the HaloLID verdict and, unless report.md says "
            "GlotLID was skipped (`--no-glotlid`), the GlotLID verdict as well.",
            f"`{OUTPUTS}/manifest.jsonl` exists, is valid JSON lines, and has one line per URL "
            f"examined with its `status` (accepted / lid_reject / fetch_fail / too_short / "
            f"host_excluded / host_capped / dup_url / dup_hash).",
            f"No accepted document is shorter than 30 words, and no language has more than "
            f"{spec.max_docs} accepted documents in this run.",
            f"No single host supplies more than {pct}% of a language's newly accepted rows, "
            f"or report.md names the host, its share, and why it was kept.",
            push_rule,
        ]
    elif spec.kind == "validate":
        items = [
            f"For every requested language ({langs}) `{OUTPUTS}/validated/{{lang}}.jsonl` exists "
            f"and is valid JSON lines (one object per line, UTF-8).",
            "Every row has exactly the keys text, language, url, content_hash, verdict, reason, "
            "confidence, validator; `verdict` is \"ok\" or \"reject\"; `validator` is \"model\".",
            "Every `confidence` is a number in [0, 1], and every `reason` is a non-empty string "
            "that names the criterion applied (language identity, naturalness/MT, completeness, "
            "safety/PII, or a positive note for ok rows).",
            f"No language file has more than {spec.max_sentences} rows, and no `text` value is "
            f"repeated within a file.",
            "Every row's `url` and `content_hash` trace back to a source shard row or Hub row "
            "(no empty or invented provenance).",
            "report.md states where the sentences came from (shard path or Hub dataset + "
            "revision) and notes that model confidence is weaker for pag, pam and tsg when any "
            "of those languages were judged.",
        ]
    elif spec.kind == "instruct":
        items = [
            f"For every requested language ({langs}) `{OUTPUTS}/instruct/{{lang}}.jsonl` exists "
            f"and is valid JSON lines.",
            "Every row has exactly the keys instruction, input, output, language, task_type, "
            "source_row, license; `instruction` and `output` are non-empty.",
            "`task_type` is one of qa, summarize, translate_en_x, translate_x_en, continue, "
            "classify, and each language file uses at least three distinct task types.",
            "`instruction` and (for all task types other than translate_x_en) `output` are "
            "written in the row's language, not English, except where the task type requires "
            "English on one side.",
            "Every `source_row` points at a row whose verdict was \"ok\" in a validated file, "
            "and `license` is copied from that source (never left empty or guessed).",
            f"No language file has more than {spec.max_sentences} rows and no two rows share the "
            f"same (instruction, input) pair.",
            "report.md gives per-language counts by task_type and the path of the validated "
            "file(s) used.",
        ]
    else:  # voice_discover
        items = [
            f"For every requested language ({langs}) `{OUTPUTS}/voice/{{lang}}.jsonl` exists "
            f"(possibly empty) and is valid JSON lines.",
            "Every row has exactly the keys video_id, url, title, license, duration_s, channel; "
            "`url` is `https://www.youtube.com/watch?v=` + video_id.",
            "Every row's `license` names a Creative Commons licence; no row with an unknown, "
            "empty or Standard YouTube licence is included.",
            "Every `duration_s` is a positive number and every `channel` is a non-empty string.",
            "No audio file (.wav, .m4a, .mp3, .webm) exists under `/mnt/session/outputs/`.",
            "report.md gives per-language counts of candidates found, LID-matched, CC-licensed, "
            "and total CC hours.",
        ]
    lines = [f"# Rubric for job kind `{spec.kind}`", ""]
    lines += [f"{i}. {c}" for i, c in enumerate(items + _common_tail(spec), 1)]
    return "\n".join(lines) + "\n"


def build_kickoff(spec: JobSpec) -> dict:
    """The `user.define_outcome` event dict for `initial_events`."""
    validate_spec(spec)
    return {
        "type": "user.define_outcome",
        "description": _description(spec),
        "rubric": {"type": "text", "content": _rubric(spec)},
        "max_iterations": spec.max_iterations,
    }
