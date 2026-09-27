#!/usr/bin/env bash
# Pack what the Mac half of the porting pipeline needs into one file, so the
# Mac needs no clone of this repo, no .env and no dataset access:
#
#   bash scripts/port_bundle_mac.sh        # -> finetune_runs/port/mac/halohalo-mac.tar (~4.5 GB)
#
# Inside: the porting code and the few repo modules the Orpheus check imports,
# the Apple-bound artefacts (MLX, Core ML, whisper.cpp models, Orpheus GGUF),
# the evalpack clips, the PyTorch reference transcripts and the TTS sentence
# manifest. The evalpack holds PLD test audio and text (CC-BY-NC, research
# only): keep the bundle private. Then, on the Mac: see scripts/port_mac.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=finetune_runs/port/mac
mkdir -p "$OUT"
A=finetune_runs/port/artefacts
W=$A/whisper-small-pld-ceb
C=$A/omniASR_W2V_1B_SSL-ctc-char-pld_ceb-norm
O=$A/orpheus-3b-0.1-pretrained-char-pld-ceb
LIST=$(mktemp)
{
  find porting -type f \( -name '*.py' -o -name '*.sh' -o -name '*.mjs' -o -name '*.html' -o -name '*.json' \) \
       -not -path '*/node_modules/*' -not -path '*/__pycache__/*'
  echo scripts/port_mac.sh
  echo scripts/tts_eval.py
  echo finetune_orpheus.py
  find halolib -type f -name '*.py' -not -path '*/__pycache__/*'
  echo docs/porting_pipeline.md
  echo docs/porting_report.md
  echo finetune_runs/port/evalpack/ceb.npz
  echo finetune_runs/port/evalpack/meta.json
  find finetune_runs/port/results -type f -name '*.json'
  echo finetune_runs/tts_eval/manifest.json
  find $W/mlx $W/coreml $W/hf-json -type f
  echo $W/ggml/ggml-model-f16.bin
  echo $W/ggml/ggml-model-q5_0.bin
  echo $W/ggml/halohalo.json
  find $C/coreml $C/hf-json -type f
  find $A/snac_24khz/coreml -type f
  echo $O/gguf/model-q4_k_m.gguf
} | grep -v '\.safetensors$' > "$LIST.all"
# MLX weights are safetensors: put them back
find $W/mlx -name '*.safetensors' >> "$LIST.all"
sort -u "$LIST.all" > "$LIST"
tar -cf "$OUT/halohalo-mac.tar" --transform 's,^,halohalo-mac/,' -T "$LIST"
rm -f "$LIST" "$LIST.all"
ls -la "$OUT/halohalo-mac.tar"
echo "copy it to the Mac, then: tar xf halohalo-mac.tar && cd halohalo-mac && bash scripts/port_mac.sh all"
