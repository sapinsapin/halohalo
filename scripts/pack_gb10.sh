#!/usr/bin/env bash
# Pack what a fresh clone on the GB10 cannot rebuild (docs/prd_gb10.md):
#
#   bash scripts/pack_gb10.sh      # -> finetune_runs/port/gb10/halohalo-gb10-inputs.tar (~50 MB)
#
#   - splits/pld_*.json: the frozen speaker- and prompt-disjoint split specs.
#     Without them the dataset loader falls back to the published random
#     split without failing, and every train and test number silently
#     changes meaning (the fleet retrain exists because of exactly that).
#   - the ceb and pam evalpacks, so the GB10 checks the very clips the
#     workstation checked (the other eight languages it builds itself)
#   - every result file so far, including the PyTorch reference transcripts
#     that parity is measured against
#   - the TTS sentence manifest the Orpheus check synthesises from
#
# Split specs, clips and transcripts are PLD (CC-BY-NC, research only): the
# tar is private. Copy it by hand; never commit, upload or paste it.
# Paths are relative to the repo root, so it untars straight into the clone.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=finetune_runs/port/gb10
mkdir -p "$OUT"
E=finetune_runs/port/evalpack
tar -cf "$OUT/halohalo-gb10-inputs.tar" \
    splits/pld_*.json \
    $E/ceb.npz $E/ceb_calib.npz $E/pam.npz $E/pam_calib.npz $E/meta.json \
    finetune_runs/tts_eval/manifest.json \
    $(find finetune_runs/port/results -type f -name '*.json')
ls -la "$OUT/halohalo-gb10-inputs.tar"
tar -tf "$OUT/halohalo-gb10-inputs.tar" | awk -F/ '{print $1"/"$2}' | sort | uniq -c
echo "on the GB10, in the clone's root: tar xf halohalo-gb10-inputs.tar"
