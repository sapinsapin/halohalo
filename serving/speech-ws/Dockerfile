# SapinSapin speech server for the GB10 (Dell Pro Max with GB10 / DGX Spark).
#
# NVIDIA's PyTorch container is the DGX Spark playbooks' path: it is the one
# image with a torch built for sm_121 on aarch64, so nothing is compiled here.
# The same Dockerfile builds on x86 with the matching nvcr tag (multi-arch).
#
#   docker build -t sapin/speech-ws serving/speech-ws
#   docker run --rm --gpus all -p 8000:8000 -v /var/lib/nyo/models:/models \
#     -e HF_HOME=/models/hf -e HF_TOKEN=... -e NYO_SERVING_KEY=... sapin/speech-ws
#
# nyo-node builds it as nyo/speech-ws:local (nyo-node build-runtime speech)
# and runs it as a Deployment with /models on the node's weight cache.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.09-py3
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf HF_HUB_CACHE=/models/hf/hub \
    OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

# ffmpeg: WebM/Opus and M4A uploads to /v1/audio/transcriptions; libsndfile
# handles WAV/FLAC/OGG without it.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Weights are not baked in: they land in /models (a hostPath shared by every
# model on the node) on first start. Set HF_TOKEN for the private repos.
VOLUME ["/models"]
EXPOSE 8000

# nyo-node overrides the command with the catalog recipe (served name, port,
# serving key); this default runs the same server open, for a hand-run test.
CMD ["python3", "serve.py", "--host", "0.0.0.0", "--port", "8000"]
