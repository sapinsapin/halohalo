set -e
cd /home/iotim/chika-curation
export HF_HUB_OFFLINE=1 PYTHONPATH=.
.venv-batchasr/bin/python -m curation.asr_batch --manifest curation/outputs/batch-asr-bench-v2/fil-input.jsonl --output curation/outputs/batch-asr-bench-v2/fil-beam5-bs8 --batch-size 8 --num-beams 5 --model sapinsapin/whisper-large-v3-pld-fil-norm --revision ee5b4f0cd1af616b38f742ae05b644d6783b52fe --language fil
.venv-batchasr/bin/python -m curation.asr_batch --manifest curation/outputs/batch-asr-bench-v2/hil-input.jsonl --output curation/outputs/batch-asr-bench-v2/hil-beam5-bs8 --batch-size 8 --num-beams 5 --model sapinsapin/whisper-large-v3-pld-hil-norm --revision 7a695b0f737b3e7cec482c29f0d12c2126cb4f4c --language hil
