# Adding the remaining dataset languages to ASR

The authenticated live observation in `curation/ASR_DEPLOYED_ROUTES_CHECKPOINT.json` advertises Filipino and Hiligaynon only. The local legacy registry (`nyo-serve/modules/asr/registry.py`) names Cebuano, Central Bikol and Ilocano models, but the faster-whisper registry (`modules/asr_fw/models.yaml`) has no entries for these three. A legacy model name is not proof of conversion, loading or public routing.

This is an implementation checklist, not a deployment receipt. No serving changes have been made by this task. NYO requested `us.kimi-k3` and returned `global.moonshotai.kimi-k3.1` for the evidence-boundary review; the steps below were checked against local `deploy/asr_fw.md` and registry code.

## Required evidence and implementation order

1. Identify the actual host, running checkout and engine behind the public gateway. The operator guide's jc ports and scratch deployments must not be assumed to be production. Read loaded models, default model, memory cap and host free disk/GPU capacity before preparing changes.
2. For each candidate `sapinsapin/whisper-large-v3-pld-{ceb,bcl,ilo}-norm`, verify the repository, immutable revision, applicable license, training language/token, architecture and evaluation data. These are candidate names from the legacy registry, not validated deployment selections. Establish the Bikol source variety before mapping generic `bik` recordings to `bcl`.
3. Add explicit entries to the faster-whisper registry in an isolated checkout: pinned source, supported quantizations, serving quantization, NRSP-to-Whisper language mapping, mel count, beam/interim settings and a language-matched quality gate. Do not copy Filipino/Hiligaynon gate scores or assume a new baseline. The local legacy registry says these fine-tunes reuse Whisper's `tl` token; verify that against each selected checkpoint before setting its mapping.
4. Convert and verify on the serving/conversion host, not the nearly full dataset laptop. The documented modules are `modules.asr_fw.convert` and `modules.asr_fw.gate`; use their current help and the operator guide for arguments. Preserve conversion manifests and pinned source hashes. An English decode smoke test proves loading, not quality in the target language. Establish a same-corpus float16 baseline and evaluate each proposed quantization with the language-matched gate.
5. Test the converted models on an isolated engine/gateway before touching the public service. Verify language routing, final transcript events, code-switching and empty/error handling on a bounded consented or appropriately licensed sample. Keep questions out of ASR inputs. Record failures and uncertain requests without automatic replay.
6. The documented engine supports additive `/admin/load` on loopback, registry reload and a manifest-based memory cap. Verify those capabilities on the actual host before using them. Preserve existing models/default and sessions; insufficient capacity is a reason to choose a separate worker or schedule, not to evict a live model implicitly. Admin endpoints are not publicly routed.
7. Re-read authenticated public `/v1/models`, then validate the exact public route with a bounded sample. Record deployment revision, model revision, quantization, protocol and results. Only then enable the corresponding bulk-ASR queue with its storage floor and request accounting. Inventory alone does not prove transcription quality or availability under load.

## Current unresolved items

Production host/checkout mapping, candidate revisions/licenses, variety-matched gate data, remote capacity, conversion receipts, model quality gates, public route registration and bounded transcription results remain unverified for the three languages. Collection continues independently. Existing Filipino and Hiligaynon serving and running dataset jobs remain unchanged.

## Candidate metadata verified on 2026-11-15

Pinned model cards, architecture configs and generation configs are recorded with file hashes in `curation/ASR_CANDIDATE_MODELS_CHECKPOINT.json`:

| Language | Candidate revision |
|---|---|
| Cebuano (`ceb`) | `109c7f42e40bfa071d53270115734dc2d88fb521` |
| Central Bikol (`bcl`) | `a35847d1db75e5c72409be4cc7f8b03775b45fb7` |
| Ilocano (`ilo`) | `00f3c4812849d7abfc0bb3519741b466457368ab` |

All three advertise `cc-by-nc-4.0`; commercial permission is not established by these cards. All three configs identify Whisper conditional generation with 128 mel bins. Their cards explicitly describe using Whisper's `tl` language token for these languages; generation configs specify `tagalog` and `transcribe`. This verifies training-token evidence, not target-language quality. The cards describe read-speech training and warn that conversational and code-switching performance may degrade. Therefore a spontaneous, code-switched evaluation set remains necessary. The user’s valid code-switching convention is unchanged; mixed-language speech is not an error by itself.

Only small metadata files were downloaded. No weights were converted, deployed or evaluated. These observations resolve candidate existence, immutable revisions, architecture and token evidence; production mapping, commercial permission, source variety, quality gates and capacity remain open.
