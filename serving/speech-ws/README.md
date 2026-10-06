# SapinSapin speech server (GB10)

Streaming text-to-speech and speech recognition for ten Philippine languages,
on the wire contract the Agora hardware integration uses, served from the
GB10 and brokered by NYO. The best published models, not the baselines:

| | Model | Measured |
|---|---|---|
| TTS | Orpheus 3B + `sapinsapin/orpheus-3b-0.1-pretrained-char-pld-<lang>` (SNAC codec) | round-trip CER 4-12 %, against 30-300 % for the SpeechT5 baselines |
| ASR | `sapinsapin/whisper-large-v3-pld-<lang>-norm` | the lowest CER per language on the speaker-disjoint split |

```
ws(s)://<host>/tts     {"text":"…"} in   -> PCM S16LE 16 kHz mono frames, then {"type":"end"}
ws(s)://<host>/asr     PCM frames in     -> {"type":"partial"} … {"type":"final"} … {"type":"end"}
POST /v1/audio/speech            OpenAI-shaped; streams PCM or returns WAV
POST /v1/audio/transcriptions    multipart file -> {"text": …}
GET  /v1/models  /v1/voices  ·  POST /v1/tickets  ·  GET /health /healthz
```

The `/tts` contract is [tts-ws's](../tts-ws/docs/AGORA_INTEGRATION.md), byte
for byte; `/asr` is [docs/ASR_INTEGRATION.md](docs/ASR_INTEGRATION.md). How to
run it on the GB10 and reach it through NYO: [docs/GB10.md](docs/GB10.md).

## How it streams

Orpheus emits SNAC codec tokens, seven to a frame. As they are generated the
server decodes four frames at a time and emits one (`orpheus_tokens.FrameWindow`,
Canopy's streaming decoder, without dropping the first and last frames),
resamples 24 → 16 kHz without seams (`pcm.StreamResampler`) and sends 100 ms
PCM frames. Time to first audio is the first 28 tokens, not the sentence.
Text is still cut into sentence-sized segments (`textseg.py`, from tts-ws),
so a long paragraph starts as early as a short one.

Whisper is not a streaming model, so `/asr` endpoints: an energy VAD buffers a
phrase while it is spoken, a partial transcript goes out every 1.5 s, and a
pause of 0.7 s finalises the phrase (`asr_stream.Endpointer`).

## Running

```bash
# on the GB10, with the NGC PyTorch image (torch for sm_121 included)
docker build -t sapin/speech-ws . && docker run --rm --gpus all -p 8000:8000 \
  -v /var/lib/nyo/models:/models -e HF_TOKEN=… -e NYO_SERVING_KEY=… sapin/speech-ws

# anywhere with a CUDA torch already installed
pip install -r requirements.txt
python serve.py --tts-langs fil,ceb,hil --asr-langs fil,ceb,hil --api-key …

# the protocol without any model (tests, or proving a tunnel before the weights are down)
TTS_ENGINE=fake ASR_ENGINE=fake python serve.py
```

| Env / flag | Default | Why change it |
|---|---|---|
| `TTS_ENGINE` | `orpheus` | `speecht5` for all ten languages on a small card; `fake` for plumbing tests |
| `TTS_LANGS`, `ASR_LANGS` | `fil,ceb,hil` | Which adapters and ASR models to load. Orpheus has nine (no `pag`). |
| `ORPHEUS_BACKEND` | `transformers` | `llamacpp` for a 2-3× faster generator (docs/GB10.md §5) |
| `ASR_MAX_RESIDENT` | `3` | How many ~3 GB Whisper models stay loaded |
| `SPEECH_API_KEY` / `NYO_SERVING_KEY` | unset | The bearer for `/v1/*`; with it set, sockets need a ticket or `?key=` |
| `SPEECH_WS_PUBLIC` | `0` | `1` leaves the sockets open even with a key (the first demo's behaviour) |
| `CHUNK_BYTES`, `PACE_FACTOR`, `MAX_TEXT_CHARS`, … | as tts-ws | The device-tuning knobs, unchanged |

## Verifying

```bash
python -m pytest tests                                       # protocol, tickets, resampler, endpointer, the server with fake engines
python tools/ws_client.py ws://localhost:8000/tts "Magandang umaga po." --expect-streaming
python tools/ws_protocol_test.py ws://localhost:8000/tts
python tools/asr_client.py ws://localhost:8000/asr clip.wav --lang ceb
```

## Layout

| File | Contents |
|---|---|
| `app.py` | The server: both sockets, the HTTP routes, auth, the executor |
| `protocol.py` | Both wire contracts; the frozen terminal frames |
| `tickets.py` | HMAC socket tickets NYO mints with the serving key |
| `orpheus_tokens.py` | Token layout, `char` frontend, streaming window arithmetic (no torch) |
| `pcm.py` | float ↔ S16LE, the streaming resampler, WAV, trimming |
| `asr_stream.py` | Endpointing for streaming recognition |
| `engines/` | `orpheus.py`, `speecht5.py`, `whisper_asr.py`, `fake.py` |
| `voices.py`, `assets.json`, `xvectors/` | Languages and voices (copies of `dashboard/`'s; regenerate there) |
| `serve.py` | Entrypoint with the flags nyo-node's recipe passes |
| `tools/` | Reference clients and the protocol conformance suite |

`assets.json`, `xvectors/` and `textseg.py` are copies of the ones under
`dashboard/` and `../tts-ws/`, byte-identical, because the Docker build
context has to be this directory alone.
