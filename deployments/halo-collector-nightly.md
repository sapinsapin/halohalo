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
name: halo-collector nightly collect
description: Nightly seeded scrape for all ten PLD languages; stages shards under /mnt/session/outputs,
  never pushes.
agent: ../agents/halo-collector.md
environment_id: ../environments/halo-collector.yaml
vault_ids:
- ../vaults/halo-collector.yaml
resources:
- type: github_repository
  url: https://github.com/sapinsapin/halohalo
  mount_path: /mnt/repo
  checkout:
    type: branch
    name: main
schedule:
  type: cron
  expression: 0 2 * * *
  timezone: Asia/Manila
initial_events:
- type: user.define_outcome
  description: |-
    Job kind: collect. Languages: bcl, ceb, eng, fil, hil, ilo, pag, pam, tsg, war. Target: up to 300 accepted documents per language. Hub push allowed: no.

    Run the seeded web scrape (`scrape_web.py`) from the halohalo repo mounted at /mnt/repo for these languages and stage the result under /mnt/session/outputs/: parquet shards per language at text/{lang}/shard-*.parquet, a combined manifest.jsonl with one line per URL examined (accepted or rejected, with the reason), and report.md. Every accepted row must carry its url, content_hash and both LID verdicts (HaloLID and GlotLID, or an explicit note in report.md that GlotLID was skipped). Do NOT push anything to the Hub.
  rubric:
    type: text
    content: |
      # Rubric for job kind `collect`

      1. For every requested language (bcl, ceb, eng, fil, hil, ilo, pag, pam, tsg, war) at least one `/mnt/session/outputs/text/{lang}/shard-*.parquet` file exists, or report.md explains per language why zero documents were accepted.
      2. Every shard row has non-empty `text`, `url`, `content_hash`, `language` and `lid_detail` columns, and `language` equals the directory's language code.
      3. Every row's `lid_detail` carries the HaloLID verdict and, unless report.md says GlotLID was skipped (`--no-glotlid`), the GlotLID verdict as well.
      4. `/mnt/session/outputs/manifest.jsonl` exists, is valid JSON lines, and has one line per URL examined with its `status` (accepted / lid_reject / fetch_fail / too_short / host_excluded / host_capped / dup_url / dup_hash).
      5. No accepted document is shorter than 30 words, and no language has more than 300 accepted documents in this run.
      6. No single host supplies more than 25% of a language's newly accepted rows, or report.md names the host, its share, and why it was kept.
      7. Nothing was pushed to the Hugging Face Hub; report.md states that push was not allowed for this job.
      8. `/mnt/session/outputs/report.md` exists with (a) a per-language table of per-stage counts (queries, hits, fetched/raw_fallback, accepted, rejected), (b) rejects broken down by reason, (c) the top hosts per language with counts, and (d) a section titled "Not done" that lists anything the task asked for that was not completed with the concrete reason (missing credential, blocked host, empty source, time), or states "nothing".
      9. No file under `/mnt/session/outputs/` contains an API key, token or other secret, and no row was fabricated: every url/video_id in the outputs came from the pipeline or the named source dataset.
  max_iterations: 4
---
