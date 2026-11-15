set -e
cd ~/chika-batchasr
export HF_HUB_OFFLINE=1 PYTHONPATH=.
M="--model sapinsapin/whisper-large-v3-pld-fil-norm --revision ee5b4f0cd1af616b38f742ae05b644d6783b52fe --language fil --min-mem-gib 8 --audio-path-map /home/iotim/chika-curation/curation/outputs/=/home/carrot/chika-mirror/"
for bs in 1 8 32; do .venv/bin/python -m curation.asr_batch --manifest /home/carrot/chika-mirror/batch-asr-bench-v2/fil-input-256.jsonl --output bench3070/fil-bs$bs --batch-size $bs $M; done
.venv/bin/python -m curation.asr_batch --manifest /home/carrot/chika-mirror/batch-asr-bench-v2/fil-input.jsonl --output bench3070/fil-bs16 --batch-size 16 $M
.venv/bin/python -m curation.asr_batch --manifest /home/carrot/chika-mirror/batch-asr-bench-v2/fil-input.jsonl --output bench3070/fil-bs16 --batch-size 16 --audit $M
