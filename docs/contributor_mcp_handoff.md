# Contributor skill and MCP handoff

User request (2026-11-14): spin this work into a separate session; turn the curation workflow into a skill and an MCP available through the sapinsapin-voice website. Contributors should use their own ChatGPT or Claude transcription capabilities, not Boses ASR, and upload easily to huggingface.co/sapinsapin through the web application.

## Product scope

Implement a reusable contributor skill, MCP service and website contribution path. Keep acquisition/transcription/linguistic text review/audio verification distinct. Preserve natural code-switching and source questions outside transcripts. Support Filipino/Taglish, Hiligaynon, Cebuano, Bikolano (explicit varieties), and Ilocano. Collection target is 250 retained/transcribed hours each; this is not a claim of current completion.

Verify current ChatGPT/Claude client capabilities using official documentation before promising that MCP can invoke native ASR. MCP tools can accept a transcript produced in the contributor's client, but availability of native transcription, attachments or audio tool results must be tested. Never silently substitute Boses or an extra paid provider. Explain and implement an honest supported fallback if native ASR cannot be invoked programmatically.

Use a server-held, least-privilege Hugging Face credential limited to the intended dataset. Contributors receive application-scoped authorization or bounded upload sessions, never the organization write token. Authenticate contributors, bind uploaded media/transcript to checksums, preserve consent/source/rights evidence, support interrupted/repeated uploads and controlled ingestion. Do not publish unreviewed submissions into an approved training split. The existing Chika dataset is private; no public release is authorized by this handoff alone.

## Existing implementation and repositories

- Curation: `/Users/timsan/halohalo/repo`, branch `chika-curation-workflow`, remote `git@github.com:sapinsapin/halohalo.git`.
- Playbook: `docs/dataset_curation.md`; canonical schema `curation/schema.py`; registry `curation/targets.json`; card `docs/cards/chika.md`.
- Reusable components: collect/import_media/prepare/asr/review/attach_reviews/cli export/publish/hub/restore/evict/supervise/media_backup. `publish --verify-existing` provides read-only recovery. Normal publisher refuses occupied prefixes; CAS prevents concurrent commit races.
- Tracking: https://github.com/sapinsapin/halohalo/issues/8.
- Dataset: private https://huggingface.co/datasets/sapinsapin/chika. PLD v2 has229 recordings/2.9522h,39 reserved test. New collected pilot has3 Hiligaynon windows/90s and2 Filipino windows/60s; unverified drafts. Pilot commit db224906b9ca13c8eb14b382443f9080c99c62d1 passed9 remote checksums,5 decoded rows and both streaming configurations.
- Website source: `git@github.com:aineolab/sapinsapin-voice.git`. Read-only clone inspected at `curation/outputs/serving-source`, commit9f341a84a22e3af04f45adc5d8c9fee8cf457af1. Read its AGENTS.md. Existing contribution flow is a distinct consented pilot; do not repurpose production data without examining its schema and authorization.
- Saved Codex project **Sapinsapin Voice**, `/Users/timsan/Documents/ChatGPT/Sapinsapin Voice`, may be a better implementation checkout: inspect identity, status and AGENTS before editing. Do not reset or discard user edits.
- Public app/docs: https://boses.sapinsapin.ai/ and /quickstart. Source points to Lovable project da17800d-aabd-435d-9d85-ea15ce9d4e18. Prefer connected tools where sufficient.

## Current workers and ownership

Collection workers remain active in the original curation workspace; do not duplicate or stop them to build the contribution product. Inspect actual process/locks before any recovery. Original HIL RSS PID13731, Taglish13792, HIL archives14584; new Cebuano29776, Bikolano RSS29761, Ilocano30164. Storage supervisor was safely upgraded to five targets as PID30225. Bikol DZGB archive worker may be launched; check outputs/bikolano/dzgb worker metadata. Radio backup PID25197 with recovery monitor25917 was still in progress, no verified receipt yet at handoff preparation. These PIDs are historical hints, not liveness evidence.

Boses direct ASR small pilots worked, but larger batch returned repeated finalize_timeout and was terminated. User explicitly prioritizes acquisition over ASR problems. Do not route new languages to fil/hil models. Native-client contribution ASR is a separate product requirement from this paused bulk ASR stage.

All credentials stay in private external files/environment or server secret storage. No credential values belong in the skill, MCP payloads, repository, client JavaScript, or chat. No need to read or transfer the old Boses key for this new contributor workflow.

## Skill authoring

Use `/Users/timsan/.codex/skills/.system/skill-creator/SKILL.md` and validate with its quick_validate.py. Package a concise distributable skill with protocol/schema references only as needed; test realistic contributor scenarios independently. Keep provider ASR capability assertions and live deployment status evidence-based.
