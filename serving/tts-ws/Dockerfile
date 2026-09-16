FROM python:3.12-slim

# Thread caps must be environment, set before torch is imported: adjusting
# them from Python afterwards is ignored by OpenMP. Two, to match the free
# CPU Space's 2 vCPU rather than oversubscribe it.
ENV OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Hugging Face Spaces runs the container as uid 1000. A cache written by root
# at build time would be unreadable at runtime, and the app would silently
# re-download every weight inside the first request.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

COPY --chown=user requirements.txt .
USER user
# Torch first, from the CPU index explicitly. requirements.txt only adds it as
# an --extra-index-url, which leaves PyPI primary — and PyPI's linux torch is
# the CUDA build, which drags in ~2.5 GB of GPU libraries this CPU-only server
# never loads. Installing it here means the -r below finds it satisfied.
RUN pip install --no-cache-dir torch==2.11.0       --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Languages served. Baked into the image, so changing this needs a rebuild —
# deliberately, because an unpinned language would download at runtime.
ENV SERVE_LANGS=fil,ceb,hil \
    DEFAULT_LANG=fil

# Weights land in the image, under the same HF_HOME the app reads at runtime.
# Copied as their own layer so a code change does not re-download 2 GB.
COPY --chown=user pins.py prefetch.py ./
RUN python prefetch.py

COPY --chown=user . .

# Every weight is in the image, so the runtime must never reach for the Hub.
# Set after the prefetch, and it does more than save a request: without it,
# transformers still tries the network first and retries with backoff before
# falling back to cache, which turns a 7-second warmup into minutes whenever
# the Hub is slow or unreachable. Unpinned languages now fail fast instead of
# downloading 600 MB mid-request, which is the behaviour we want.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 7860

# --workers 1 is required, not a default: a second worker would hold its own
# copy of the weights and break the single-inflight semaphore.
# --ws websockets is explicit because uvicorn's automatic choice between
# websockets and wsproto changes behaviour.
# --ws-ping-timeout 60 because uvicorn's default of 20 disconnects a minimal
# client that does not auto-pong, mid-synthesis.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", \
     "--ws", "websockets", "--ws-ping-interval", "20", "--ws-ping-timeout", "60", \
     "--timeout-keep-alive", "75", "--workers", "1", "--log-level", "info"]
