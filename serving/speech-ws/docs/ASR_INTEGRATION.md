# SapinSapin speech recognition — WebSocket integration

The streaming-recognition side of the speech server. Text-to-speech is the
contract in [`../../tts-ws/docs/AGORA_INTEGRATION.md`](../../tts-ws/docs/AGORA_INTEGRATION.md),
unchanged; this endpoint is shaped after it so a device that already speaks
that one has nothing new to learn: JSON text frames for control and results,
binary frames for audio, exactly one terminal frame per request.

## 1. Endpoint

    ws(s)://<host>/asr

Same host and handshake rules as `/tts`: an HTTP/1.1 WebSocket upgrade,
`Sec-WebSocket-Version: 13`. When the server has a serving key, the URL
carries a ticket (`/asr?ticket=…`, from `POST /v1/tickets`, which NYO's
gateway mints for you) or `?key=…`; without a key the socket is open.

`GET /healthz` returns `{"ready":true,…,"asr_langs":[…]}` once the models are
resident; check it before a scheduled test.

## 2. Request

Optionally, one text frame first:

```json
{"type":"start","lang":"ceb","sample_rate":16000,"partials":true}
```

| Field | Default | Meaning |
|---|---|---|
| `lang` | the server default (`fil`) | `fil`, `ceb`, `hil`, … or the display name. Unknown values are an error naming the served list. |
| `sample_rate` | `16000` | `16000`, `48000` or `8000`. Anything else is an error: a silently wrong rate gives a plausible, wrong transcript. |
| `partials` | `true` | `false` to receive finals only. |

Then **binary frames of PCM S16LE, mono**, at the declared rate. Any frame
length works as long as it is even; 3200 bytes (100 ms at 16 kHz) is a good
default, and no frame may exceed 65536 bytes. Send them at the pace you
record them, or faster from a file.

To end: `{"type":"stop"}`, or close the socket. Without `stop`, the server
treats 60 s of silence on the wire as the end.

## 3. Response

Text frames only. While a phrase is being spoken, roughly every 1.5 s:

```json
{"type":"partial","text":"maayong buntag sa","start":0.42}
```

When the speaker pauses (about 0.7 s of silence), the phrase is transcribed
once more as a whole and sent as:

```json
{"type":"final","text":"maayong buntag sa tanan","start":0.42,"end":2.91}
```

`start` and `end` are seconds since the first audio frame. A `final` always
follows the last `partial` of its phrase; a phrase under 250 ms is treated as
noise and produces nothing. Phrases longer than 25 s are cut at 25 s (Whisper
sees 30 s at a time) and continue as the next phrase.

After `stop` (or close), any open phrase is finalised and exactly one of:

```
{"type":"end"}
{"type":"error","message":"..."}
```

`end` is the same 14 bytes as on `/tts`, byte for byte.

## 4. Errors

| Message | Cause |
|---|---|
| `invalid JSON`, `expected a JSON object` | A text frame that did not parse. |
| `unknown 'type': …` | A text frame other than `start` or `stop`. |
| `unsupported 'sample_rate'; …` | See §2. |
| `unknown 'lang'; served: …` | The language is not loaded on this instance. |
| `audio frame has an odd byte length` | An int16 sample straddled two frames. |
| `audio frame too large` | Over 65536 bytes in one frame. |
| `connection audio limit reached` | More than 600 s of audio on one connection. |
| `model still loading, retry` | The instance is warming up; `/healthz` says when it is ready. |
| `transcription failed: <ExceptionType>` | Internal failure; the type name is included. |
| `server busy, retry` | Too many connections. |
| `unauthorized` | No valid ticket or key. |

An `error` is always the last frame. The close code is always `1000`.

## 5. What the transcript looks like

The models are `sapinsapin/whisper-large-v3-pld-<lang>-norm`: trained on
**normalised** text, so transcripts are lower case with no punctuation
(`maayong buntag sa tanan`). They were trained on read, prompted speech in
quiet rooms; expect worse on spontaneous or noisy audio. Per-language error
rates are in the dataset card and `dashboard/asr_registry.json` (frozen,
speaker-disjoint split: 5 % CER on Filipino to 16 % on Pangasinan).

## 6. Receive loop

```python
import asyncio, json, websockets

async def main(pcm_frames):
    async with websockets.connect("wss://<host>/asr?ticket=...", max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "lang": "ceb"}))
        async def send():
            for f in pcm_frames:              # bytes, PCM S16LE 16 kHz mono
                await ws.send(f)
            await ws.send(json.dumps({"type": "stop"}))
        asyncio.create_task(send())
        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "partial":  print("…", msg["text"])
            elif msg["type"] == "final":  print(msg["start"], msg["end"], msg["text"])
            else:                          break            # end or error
```

`tools/asr_client.py` is this loop with the checks, fed from a WAV file.

## 7. The HTTP alternative

For a recorded clip, `POST /v1/audio/transcriptions` (multipart `file`,
optional `language`, `response_format` `json` | `text` | `verbose_json`)
returns `{"text": "..."}` in one go; WAV, FLAC and OGG always, WebM/Opus and
M4A when ffmpeg is on the host. NYO's gateway proxies it at
`/api/public/v1/audio/transcriptions` with your NYO key.
