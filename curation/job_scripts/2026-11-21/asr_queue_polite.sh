#!/bin/bash
# Polite GPU queue for dwnd-01: run ready batch-ASR selections one at a time, only when the GPU is free of other
# people's jobs. Before each batch: no foreign GPU compute processes, MemAvailable >= 45 GiB, and the agent_sync
# claim on gpu:dwnd succeeds. The claim is renewed every chunk-check and released after each batch, so another agent
# can take the GPU between batches. Stops cleanly when the queue is done or a batch fails (never retries a failure).
#
# Usage (dwnd-01): nohup curation/job_scripts/2026-11-21/asr_queue_polite.sh > curation/outputs/asr_queue_polite.log 2>&1 &
set -u
cd /home/iotim/chika-curation
O=curation/outputs
SYNC="python3 agent_sync.py"
PY=.venv-batchasr/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=.
# dir | model | revision | language
QUEUE="
remote-ceb-asr-v5-v1|sapinsapin/whisper-large-v3-pld-ceb-norm|109c7f42e40bfa071d53270115734dc2d88fb521|ceb
remote-ilo-asr-v5-v1|sapinsapin/whisper-large-v3-pld-ilo-norm|00f3c4812849d7abfc0bb3519741b466457368ab|ilo
remote-ceb-asr-mac-podcasts-v1|sapinsapin/whisper-large-v3-pld-ceb-norm|109c7f42e40bfa071d53270115734dc2d88fb521|ceb
remote-ceb-asr-mac-archive-v1|sapinsapin/whisper-large-v3-pld-ceb-norm|109c7f42e40bfa071d53270115734dc2d88fb521|ceb
remote-ilo-asr-mac-podcasts-v1|sapinsapin/whisper-large-v3-pld-ilo-norm|00f3c4812849d7abfc0bb3519741b466457368ab|ilo
remote-ilo-asr-mac-archive-v1|sapinsapin/whisper-large-v3-pld-ilo-norm|00f3c4812849d7abfc0bb3519741b466457368ab|ilo
remote-bik-asr-dzgb-v1|sapinsapin/whisper-large-v3-pld-bcl-norm|a35847d1db75e5c72409be4cc7f8b03775b45fb7|bik
"
gpu_free() {
  local mine foreign mem
  foreign=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  mem=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  [ "$foreign" -eq 0 ] && [ "$mem" -ge 45 ]
}
for item in $QUEUE; do
  IFS='|' read -r dir model rev lang <<< "$item"
  D=$O/$dir
  if [ -f $D/asr/summary.json ]; then echo "$(date -u +%FT%TZ) $dir already complete"; continue; fi
  echo "$(date -u +%FT%TZ) waiting for a free GPU for $dir"
  while true; do
    if gpu_free && $SYNC claim gpu:dwnd --agent claude --ttl 3 --note "batch ASR $dir (asr_queue_polite)" >/dev/null; then break; fi
    sleep 300
  done
  $SYNC claim data:$dir --agent claude --ttl 12 --note "batch ASR output $dir/asr" >/dev/null || { echo "data:$dir held by another agent; skipping"; $SYNC release gpu:dwnd --agent claude; continue; }
  echo "$(date -u +%FT%TZ) start $dir"
  ( while sleep 1200; do $SYNC renew gpu:dwnd --agent claude --ttl 3 --note "batch ASR $dir (asr_queue_polite)" >/dev/null; done ) &
  renewer=$!
  nice -n 10 $PY -m curation.asr_batch --manifest $D/input.jsonl --output $D/asr --batch-size 16 --chunk-rows 256 \
      --model $model --revision $rev --language $lang
  rc=$?
  kill $renewer 2>/dev/null
  if [ $rc -ne 0 ]; then
    $SYNC release gpu:dwnd --agent claude --note "$dir FAILED rc=$rc; see asr/worker.json; queue stopped"
    $SYNC release data:$dir --agent claude --note "failed rc=$rc"
    echo "$(date -u +%FT%TZ) $dir failed rc=$rc; stopping"; exit 1
  fi
  $SYNC release gpu:dwnd --agent claude --note "$dir complete"
  $SYNC release data:$dir --agent claude --note "$dir/asr complete; needs merge + overlap check + publish"
  $SYNC note --agent claude "ASR complete: $dir ($lang). Merge with merge_export_batch.py, check overlap vs published $lang configs, then publish." >/dev/null
  echo "$(date -u +%FT%TZ) done $dir"
done
echo QUEUE_DONE
