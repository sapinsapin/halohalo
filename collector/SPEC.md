# halo-collector — MVP spec (shared contract for all workers)

A Claude Managed Agent that collects and validates text data for the ten PLD
languages, plus a small web app that drives it. Lives in `halohalo/` on the
branch `claude/youthful-brown-6a5zgf`. Python 3.11, `anthropic>=1.15`.

## Layout

```
agents/halo-collector.md              # ant apply: YAML frontmatter = agent config, body = system prompt
environments/halo-collector.yaml      # ant apply: environment create body
vaults/halo-collector.yaml            # ant apply: vault container (display_name only, never a credential)
deployments/halo-collector-nightly.md # ant apply: nightly flywheel deployment (frontmatter = create body)
collector/
  __init__.py
  config.py        # resolves HALO_AGENT_ID / HALO_ENVIRONMENT_ID / HALO_VAULT_ID (env vars, else claude-lock.json)
  jobs.py          # JobSpec, LANGUAGES, build_kickoff() -> user.define_outcome event
  runtime.py       # Collector: start / events / confirm / interrupt / outputs / download / status
  live_smoke.py    # first live run: preflight, one small job (push off), stream, report.md download
  setup_sdk.py     # ONE-TIME SETUP fallback when `ant` is unavailable (creates env, agent, vault, credentials)
  web/app.py       # FastAPI app; static/ is the single-page UI
  web/demo.py      # DemoCollector for DEMO=1 (pacing: DEMO_DELAY seconds, default 0.4)
  web/static/{index.html,app.js,style.css}
  tests/           # pytest, no network: fake anthropic client
docs/collector.md  # how to set up and run
```

## Decisions already made (do not re-open)

- Jobs are kicked off on demand from the web app or CLI; a nightly *deployment*
  file exists for the `collect` job but is optional.
- The agent **stages** shards under `/mnt/session/outputs/` and pushes to the
  Hub (`sapinsapin/halo-{lang}`) only when the job was started with `push=True`.
- Voice in v1 = CC-licensed YouTube discovery only (`scrape_web.py --voice`,
  no download). No ASR in the sandbox (no GPU).
- Model `claude-opus-5-5`, effort high. Toolset `agent_toolset_20260401`,
  `default_config: {permission_policy: {type: auto}}`, `web_search` and
  `web_fetch` both `enabled: false` (Tavily inside the pipeline searches).
  `bash` stays on `auto`; the run is unattended most of the time, so the
  runtime denies paused calls unless a person answers from the web app.
- Environment: `cloud`, `limited`, `allow_package_managers: true`,
  `allowed_hosts: [huggingface.co, "*.huggingface.co", api.tavily.com, youtube.com, www.youtube.com, "*.googlevideo.com", "*.ytimg.com"]` (the YouTube hosts exist only for `voice_discover`).
- Repo mount: `https://github.com/sapinsapin/halohalo` at `/mnt/repo`,
  branch `main`, token from `GITHUB_TOKEN` (never stored in files).
- Vault credentials (`environment_variable`): `HF_TOKEN` (hosts
  `huggingface.co`, `*.huggingface.co`), `TAVILY_API_KEY` (host
  `api.tavily.com`). Both `injection_location: {header: true, body: true}`
  (Tavily takes the key in the JSON body).
- Kickoff is always `user.define_outcome` in `initial_events` (one round trip).

## Languages

`LANGUAGES = ["bcl", "ceb", "eng", "fil", "hil", "ilo", "pag", "pam", "tsg", "war"]`
(PLD codes; `eng` = Philippine English).

## `collector/jobs.py` (owner: worker A)

```python
JobKind = Literal["collect", "validate", "instruct", "voice_discover"]

@dataclass
class JobSpec:
    kind: JobKind
    languages: list[str]            # subset of LANGUAGES, non-empty
    max_docs: int = 300             # collect: accepted docs per language
    max_sentences: int = 500        # validate / instruct: rows per language
    push: bool = False              # allow Hub push
    notes: str = ""                 # free text appended to the task description
    max_iterations: int = 4         # outcome grader iterations

def validate_spec(spec: JobSpec) -> None        # raises ValueError
def build_kickoff(spec: JobSpec) -> dict        # {"type": "user.define_outcome", description, rubric: {type: text, content}, max_iterations}
def job_title(spec: JobSpec) -> str             # e.g. "collect · ceb,war · 2026-10-17"
```

Each kind has a description paragraph and a rubric (5–10 gradeable criteria).
Output contract the rubric enforces (all under `/mnt/session/outputs/`):

- `collect`:  `text/{lang}/shard-*.parquet` + `manifest.jsonl` + `report.md`
- `validate`: `validated/{lang}.jsonl`, rows `{text, language, url, content_hash, verdict: ok|reject, reason, confidence: 0-1, validator: "model"}` + `report.md`
- `instruct`: `instruct/{lang}.jsonl`, rows `{instruction, input, output, language, task_type, source_row, license}` + `report.md`
- `voice_discover`: `voice/{lang}.jsonl`, rows `{video_id, url, title, license, duration_s, channel}` (CC-licensed only) + `report.md`

## `collector/runtime.py` (owner: worker B)

```python
Event = dict   # normalized, JSON-serialisable:
# {"id": str, "type": str, "ts": str|None, "text": str|None, "tool": str|None,
#  "input": dict|None, "permission": str|None, "stop_reason": str|None,
#  "result": str|None, "explanation": str|None, "iteration": int|None, "raw_type": str,
#  "answered": bool}   # a historical tool_ask already answered replays as tool_use with answered=True

class Collector:
    def __init__(self, client=None, *, agent_id=None, environment_id=None,
                 vault_ids=None, repo_url=None, repo_branch="main", repo_token=None): ...
    def start(self, spec: JobSpec, title: str | None = None) -> str   # session id
    def events(self, session_id: str, *, include_history=True, on_open=None) -> Iterator[Event]
        # on_open(stream) gets the SDK stream so the caller can close() it on client disconnect
        # stream-first; history via events.list deduped by id; yields until terminal gate
        # (session.status_terminated, or status_idle with stop_reason != requires_action)
        # paused tool calls (evaluated_permission == "ask") are yielded with type "tool_ask"
        # and NOT auto-answered — the caller decides (web UI) or calls deny_all()
    def confirm(self, session_id: str, tool_event_id: str, allow: bool, message: str | None = None) -> None
    def interrupt(self, session_id: str) -> None
    def status(self, session_id: str) -> dict        # {id, status, outcome: result|None, usage}
    def outputs(self, session_id: str) -> list[dict] # [{id, filename, size_bytes}]
    def download(self, session_id: str, file_id: str, dest: Path) -> Path
    def console_url(self, session_id: str) -> str
```

`collector/config.py`: `load_ids() -> dict(agent_id, environment_id, vault_id)`
from env `HALO_AGENT_ID` / `HALO_ENVIRONMENT_ID` / `HALO_VAULT_ID`, falling
back to `claude-lock.json` (`resources["./agents/halo-collector.md"].id`, etc.).

## `collector/web/app.py` (owner: worker C)

FastAPI, single-page UI with the same flow as the agent proposal:
**pick a job → languages/options → start → watch the stream → answer paused tool
calls → download outputs**.

Endpoints:
- `GET  /`                                   → static index
- `GET  /api/languages`                       → LANGUAGES
- `GET  /api/jobs`                            → list of started jobs (in-memory registry, persisted to `jobs.json` under `COLLECTOR_STATE_DIR`)
- `POST /api/jobs` `{kind, languages, max_docs?, max_sentences?, push?, notes?, max_iterations?}` → `{session_id, title, console_url}`
- `GET  /api/jobs/{sid}/events`               → SSE of normalized Events (one `data:` JSON per event; `event: end` at terminal)
- `POST /api/jobs/{sid}/confirm` `{tool_event_id, allow, message?}`
- `POST /api/jobs/{sid}/interrupt`
- `GET  /api/jobs/{sid}/status`             → `{..., pending_asks: [...]}`
- `GET  /api/jobs/{sid}/outputs`              → files list
- `GET  /api/jobs/{sid}/outputs/{file_id}`     → file download

The app constructs one `Collector` from `collector.config.load_ids()` and env
(`GITHUB_TOKEN`, `HALO_REPO_URL`, `HALO_REPO_BRANCH`). Tests inject a fake Collector
via `app.state.collector`. `DEMO=1` env → a built-in fake Collector that replays a
canned event script so the UI can be demoed without credentials.

Other env vars: `COLLECTOR_STATE_DIR` (registry dir, default `./collector_state`),
`COLLECTOR_ALLOWED_HOSTS` (comma-separated extra `Host` values besides loopback;
other hosts and cross-origin writes get 403), `COLLECTOR_ASK_TIMEOUT_S` (auto-deny an
unanswered tool ask, default 900, `0` disables), `WEB_CONCURRENCY` (>1 is refused:
single process only). CLI flags: `runtime.py` `--kind --langs --max-docs --max-sentences
--push --notes --deny-asks --no-history`; `live_smoke.py` `--kind --langs --max-docs
--max-sentences --notes --setup --deny-asks --timeout-min --state-dir` (exit 0 satisfied,
1 unsatisfied, 2 preflight, 3 timeout); `setup_sdk.py` `--dry-run --deployment
--write-deployment --lock`.

## Testing rule

No test touches the network. `collector/tests/fake_client.py` (worker B
writes it, C reuses it) is a minimal stand-in for `anthropic.Anthropic()`
exposing `beta.sessions.{create,retrieve}`, `beta.sessions.events.{send,stream,list}`,
`beta.files.{list,download}` with scripted events.
