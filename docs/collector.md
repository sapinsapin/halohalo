# halo-collector — setup and operation

A Claude Managed Agent that runs the halohalo scrape pipeline for the ten PLD
languages (`bcl ceb eng fil hil ilo pag pam tsg war`) inside an Anthropic cloud
sandbox, plus a small web app and CLI that drive it. The shared contract is
`collector/SPEC.md`; this page is the how-to.

```
agents/halo-collector.md              # agent config (frontmatter) + system prompt (body)
environments/halo-collector.yaml      # cloud sandbox, limited networking
vaults/halo-collector.yaml            # vault container (credentials are added separately)
deployments/halo-collector-nightly.md # optional nightly collect, 02:00 Asia/Manila
collector/jobs.py                     # JobSpec -> user.define_outcome (description + rubric)
collector/setup_sdk.py                # SDK fallback for ant apply; writes claude-lock.json
collector/runtime.py                  # Collector: start / events / confirm / outputs
collector/live_smoke.py               # first live run: preflight + one small job + report.md
collector/web/app.py                  # FastAPI single-page UI
claude-lock.json                      # resource IDs (written by ant apply or setup_sdk)
```

## First live run

Once `ANTHROPIC_API_KEY` is set and the vault credentials exist, one command does the
whole first run end to end (no Hub push, ever):

```sh
python -m collector.live_smoke                       # validate · ceb · 20 sentences, 30 min budget
python -m collector.live_smoke --setup --deny-asks   # also create env/agent/vault first (idempotent)
python -m collector.live_smoke --kind collect --langs war --max-docs 10 --timeout-min 15
```

It checks the environment first without calling the API (`ANTHROPIC_API_KEY` or
`ANTHROPIC_AUTH_TOKEN` required; `GITHUB_TOKEN`, `HF_TOKEN`, `TAVILY_API_KEY` only warned
about — at run time the vault credentials are what count), exits 2 with the two setup
commands if `claude-lock.json`/`HALO_*_ID` are missing, starts the job, streams it to the
terminal, denies paused tool calls unless you answer `y` on a tty (`--deny-asks` never asks),
interrupts the session when the budget runs out (exit 3), then prints the status, lists the
outputs and saves `report.md` to `collector_state/smoke/<session id>/`. Exit 0 means the
outcome grader said `satisfied`; 1 is anything else.

## 1. Create the resources

### Path A — `ant apply` (recommended)

Needs `ant` >= 1.34 and `ANTHROPIC_API_KEY` (or `ant auth login`). Run from the repo root;
the first run writes `claude-lock.json` there — commit it.

```sh
ant apply --dry-run -v agents/halo-collector.md environments/halo-collector.yaml vaults/halo-collector.yaml
ant apply agents/halo-collector.md environments/halo-collector.yaml vaults/halo-collector.yaml
```

Name the files; do not pass `.` (the repo has other `deployments/`-like names that a walk
would pick up). To change the agent, edit the file and apply again; the agent gets a new
version and running sessions keep the old one.

### Path B — the SDK (no `ant`)

```sh
pip install "anthropic>=1.15" pyyaml
export ANTHROPIC_API_KEY=...  HF_TOKEN=...  TAVILY_API_KEY=...
python -m collector.setup_sdk --dry-run     # print every payload, call nothing, needs no key
python -m collector.setup_sdk               # environment + agent + vault + both credentials
```

It reads the same four files `ant apply` uses (the agent body becomes `system`), writes
`claude-lock.json` in the same shape (`resources["./agents/halo-collector.md"].id` etc.),
and is idempotent: an ID already in the lock file is updated, not recreated. A credential
whose env var is unset is skipped with a warning. Do not mix the two paths for the same
resource — `ant apply` cannot adopt an SDK-created agent and would create a second one.

## 2. Credentials

The vault holds two `environment_variable` credentials. Inside the sandbox they appear as
opaque placeholders; the real value is substituted only on requests to the listed hosts.

| secret_name | allowed_hosts | injection |
|---|---|---|
| `HF_TOKEN` | `huggingface.co`, `*.huggingface.co` | header + body |
| `TAVILY_API_KEY` | `api.tavily.com` | header + body (Tavily takes the key in the JSON body) |

`setup_sdk.py` adds them from the environment. By hand (one block at a time, the secret typed
at a silent prompt so it never sits on a command line or in a file):

```sh
VAULT_ID=$(jq -r '.resources["./vaults/halo-collector.yaml"].id' claude-lock.json)
read -rs SECRET && ant beta:vaults:credentials create --vault-id "$VAULT_ID" <<YAML
display_name: TAVILY_API_KEY
auth:
  type: environment_variable
  secret_name: TAVILY_API_KEY
  secret_value: "$SECRET"
  networking: {type: limited, allowed_hosts: [api.tavily.com]}
  injection_location: {header: true, body: true}
YAML
```

Same for `HF_TOKEN` with `allowed_hosts: [huggingface.co, "*.huggingface.co"]`. The repo
token (`GITHUB_TOKEN`, fine-grained PAT, Contents: Read) is passed on each session create as
the `github_repository` resource's `authorization_token`; it is never stored in a file or a
vault. `.env` is git-ignored; `claude-lock.json` holds IDs only and is safe to commit.

## 3. Run a job

IDs are resolved by `collector/config.py`: env `HALO_AGENT_ID` / `HALO_ENVIRONMENT_ID` /
`HALO_VAULT_ID`, else `claude-lock.json`. Also set `GITHUB_TOKEN` so the repo can be mounted
at `/mnt/repo`.

**From the CLI** — see `collector/runtime.py` for the exact flags; the shape is

```sh
python -m collector.runtime --kind collect --langs ceb,war --max-docs 200
python -m collector.runtime --kind validate --langs pag,pam,tsg --max-sentences 300
python -m collector.runtime --kind instruct --langs ilo
python -m collector.runtime --kind voice_discover --langs hil
```

`--push` is accepted only for `collect` and lets the agent append finished shards to
`sapinsapin/halo-{lang}`; without it everything stays staged under `/mnt/session/outputs/`.
Every run prints its Console URL (`https://platform.claude.com/workspaces/default/sessions/<id>`).
`--notes` appends free text to the task, `--deny-asks` denies paused tool calls without
prompting, `--no-history` skips the history replay before tailing.

**From the web app**

```sh
pip install fastapi uvicorn                 # both are in requirements.txt; SSE needs no extra package
uvicorn collector.web.app:app --reload      # http://127.0.0.1:8000
```

The app has **no authentication**. Keep it on `127.0.0.1` (uvicorn's default):
never run it with `--host 0.0.0.0` or behind a reverse proxy without adding
auth, because it can start jobs with Hub push, approve `bash` calls inside the
sandbox, interrupt runs and serve output files. Requests whose `Host` is not
loopback, or whose `Origin` differs from the host on a write, are refused with
403 (set `COLLECTOR_ALLOWED_HOSTS=myhost` to allow another name). Don't run it
with `ANTHROPIC_LOG=debug` on a shared machine: the SDK's debug log prints
request bodies, which include the repo token sent on session creation.

Run it as a **single process**: the job registry, pending tool asks and open SDK
streams live in memory, so `--workers` must stay 1 (`create_app` refuses
`WEB_CONCURRENCY>1`). A tool ask nobody answers is denied after
`COLLECTOR_ASK_TIMEOUT_S` seconds (default 900; `0` disables) and the feed shows a
`web.auto_deny` line. Asks answered from the Console disappear from the UI when
the `user.tool_confirmation` echo arrives, and reloading a page whose session is
`idle` with a pending ask reconnects to the stream instead of treating it as done.

```sh
DEMO=1 uvicorn collector.web.app:app        # canned event script, no credentials needed
DEMO=1 DEMO_DELAY=0 uvicorn collector.web.app:app   # DEMO_DELAY = seconds between canned events (default 0.4)
```

The app builds its `Collector` from `collector.config.load_ids()` plus `GITHUB_TOKEN`,
`HALO_REPO_URL` (default `https://github.com/sapinsapin/halohalo`) and `HALO_REPO_BRANCH`
(default `main`); the CLI uses the same IDs but always mounts the default repo and branch.

Flow: pick a job kind → languages and options → start → watch the event stream → answer any
paused tool call (bash runs under the `auto` policy; a call the server cannot decide on
pauses the session until someone clicks allow/deny — unattended runs deny) → download the
outputs. Started jobs are listed in `jobs.json` under `COLLECTOR_STATE_DIR`
(default `collector_state/`, git-ignored; `live_smoke --state-dir` overrides it for `report.md`).
`POST /api/jobs` also accepts `max_iterations` (default 4, API cap 20).

**What a job produces** (all under `/mnt/session/outputs/`, downloadable via the Files API):

| kind | files |
|---|---|
| `collect` | `text/{lang}/shard-*.parquet`, `manifest.jsonl`, `report.md` |
| `validate` | `validated/{lang}.jsonl` (`text, language, url, content_hash, verdict, reason, confidence, validator`), `report.md` |
| `instruct` | `instruct/{lang}.jsonl` (`instruction, input, output, language, task_type, source_row, license`), `report.md` |
| `voice_discover` | `voice/{lang}.jsonl` (`video_id, url, title, license, duration_s, channel`, CC-only), `report.md` |

Each job is a `user.define_outcome` whose rubric (`collector/jobs.py`) the outcome grader
scores per criterion for up to `max_iterations` (default 4) rounds. `report.md` always ends
with a **Not done** section; read it before trusting the counts.

## 4. The nightly deployment (optional)

`deployments/halo-collector-nightly.md` runs `collect` for all ten languages at 02:00
Asia/Manila with push disabled. Its `user.define_outcome` is generated from `jobs.py`
(`python -m collector.setup_sdk --write-deployment`; the test suite fails if it drifts).
Two things must be wired before it works:

1. **Vault** — `vault_ids` takes IDs, not paths. Apply the vault first, paste its `vlt_...`
   into the file, then apply the deployment.
2. **Repo token** — the file mounts `https://github.com/sapinsapin/halohalo` at `/mnt/repo`
   without an `authorization_token`, which is enough while the repo is public. If it is
   private, do not put the token in the file: `python -m collector.setup_sdk --deployment`
   injects `GITHUB_TOKEN` from the environment at create time.

```sh
ant apply deployments/halo-collector-nightly.md &&
  DEP=$(jq -r '.resources["./deployments/halo-collector-nightly.md"].id' claude-lock.json) &&
  ant beta:deployments pause --deployment-id "$DEP"       # live from creation: pause first
ant beta:deployments run --deployment-id "$DEP"            # manual test run (works while paused)
ant beta:deployment-runs list --deployment-id "$DEP" --max-items 1
ant beta:deployments unpause --deployment-id "$DEP"        # once a run looks right
```

(`setup_sdk.py --deployment` creates it paused.) Nobody is attached when a scheduled run
fires, so a paused tool call waits indefinitely; keep `bash` on `auto`, and check runs with
`ant beta:deployment-runs list --has-error true`.

## 5. Limits

- **No GPU.** No ASR, no alignment, no fine-tuning in the sandbox; `requirements.txt`'s GPU
  stages are not installed. GlotLID (1.7 GB) is skipped by default (`--no-glotlid`), so rows
  carry the HaloLID verdict only unless a job asks otherwise.
- **Voice = discovery only.** `voice_discover` lists CC-licensed YouTube videos; nothing is
  downloaded. The environment allows `youtube.com`, `www.youtube.com`, `*.googlevideo.com`
  and `*.ytimg.com` for yt-dlp's search and licence probes only. The pipeline
  licence-probes the first 40 candidates per language, so later rows carry
  `license: unknown` and are counted separately in `report.md`.
- **Networking is deny-by-default.** Only `huggingface.co`, `*.huggingface.co`,
  `api.tavily.com`, the YouTube hosts above and package indexes are reachable. Tavily returns page text with its
  results, so the scrape works, but the live-fetch step fails and the pipeline falls back
  to the search dump; `report.md` records the fetch failures.
- **Pushes are gated twice**: the job must be started with push allowed, and the agent pushes
  only languages whose manifest shows a finished run.
- **The agent judges validation itself** (`validator: "model"`); confidence on `pag`, `pam`
  and `tsg` is lower and the report says so. Treat `validate` output as a first pass, not
  ground truth.
- Tests never touch the network: `python -m pytest collector/tests -q`.
