# Working alongside other agents (Claude Code, Antigravity, people)

Several agents run Chika curation jobs at the same time on **dwnd-01** (`ssh dwnd`, user `iotim`, workspace
`~/chika-curation`) and publish to **`sapinsapin/chika`** on Hugging Face. To avoid two agents using the GPU, writing
the same output directory or racing commits to the dataset, everyone uses one claim board on dwnd-01.

## Before you start anything long-running or shared

```sh
ssh dwnd 'cd ~/chika-curation && python3 agent_sync.py status'          # who holds what
ssh dwnd 'cd ~/chika-curation && python3 agent_sync.py log -n 30'        # what happened recently
```

Then claim what you will use. `claim` fails (exit 1) if someone else holds it — pick other work or wait; never
work around a claim.

```sh
ssh dwnd 'cd ~/chika-curation && python3 agent_sync.py claim gpu:dwnd --agent antigravity --ttl 6 --note "what and where"'
```

| Resource | Claim it before you… |
| --- | --- |
| `gpu:dwnd` | run batch ASR, evaluations or anything else on the GB10 GPU |
| `hub:sapinsapin/chika` | upload, register a card config, or change settings on the dataset |
| `data:<output-dir>` | create or write a directory under `curation/outputs/` |
| `lang:<code>` | select the next batch of a language's untranscribed audio (prevents two agents picking the same clips) |
| `job:<name>` | anything else long-running |

- Use `--agent antigravity`, `--agent claude`, or your own name. TTL is in hours; `renew` with the same arguments
  before it runs out if the job is still going. Expired claims are free for anyone.
- When done (or abandoned): `release <resource> --agent <you> --note "result, output dir, Hub revision"`.
- Record anything others should know: `note --agent <you> "…"` (published configs, failures, things not to replay).

## House rules that still apply

- Outputs are immutable: write a **new** directory, never modify another agent's output or a published prefix.
- Every published row stays `draft` / `unassigned`; rights stay `unknown` unless there is evidence.
- Before publishing, check clip IDs and `audio_sha256` against every already-published config of that language.
- Never put tokens or keys in files, commands shown to users, or this board. Credentials live in
  `~/.config/hf/` on dwnd-01 and `~/gemini-key` on the Mac.
- Contributor uploads go to the private `sapinsapin/chika-contributions`, never to `sapinsapin/chika`.

Playbook: `docs/dataset_curation.md`. Release record: `docs/dataset_releases.md`.
