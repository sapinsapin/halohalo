# SapinSapin TTS — WebSocket integration

Everything needed to point a device at the SapinSapin text-to-speech demo.
One endpoint, one request message, raw PCM back. No authentication.

## 1. Endpoint

    ws://104.208.87.149:8765/tts

**Plain `ws://`, no TLS.** Standard HTTP/1.1 WebSocket upgrade
(`Sec-WebSocket-Key`, version 13). No authentication. Static IP, so the address
will not move; the host is in Azure East Asia (Hong Kong), roughly 30 ms from
Manila.

Being unencrypted is deliberate for this first test, so your client needs no TLS
stack, no CA bundle and no SNI. It does mean the audio and text cross the public
internet in cleartext — fine for test sentences, worth revisiting before anything
real. We can add TLS on request.

**The handshake must be HTTP/1.1.** A WebSocket upgrade sent over HTTP/2 is
meaningless, and a TLS front end that negotiated h2 will answer it with **404**
— which reads like "the endpoint does not exist" rather than "wrong protocol
version". Verified against this endpoint with curl:

```
curl -so /dev/null -w '%{http_code} over %{http_version}\n' \
  -H 'Connection: Upgrade' -H 'Upgrade: websocket' \
  -H 'Sec-WebSocket-Version: 13' \
  -H 'Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==' \
  http://104.208.87.149:8765/tts
# 404 over 2      <- curl negotiated HTTP/2
# 101 over 1.1    <- same request with --http1.1
```

If you get a 404, check the negotiated HTTP version before assuming the path is
wrong. Note that curl can complete the *handshake* but cannot speak the
protocol afterwards, so this is a reachability smoke test only.

A plain `GET /` on the same host returns what that instance is serving, and
`GET /healthz` returns `{"ready":true,...}` once the models are resident. Check
`/healthz` before a scheduled test.

## 2. Request

One JSON message per utterance, as a WebSocket text frame:

```json
{"text": "Magandang umaga po sa inyong lahat."}
```

A text payload sent with the binary opcode is also accepted.

Optional fields. Omit them and you get Filipino in the default voice; adding
them changes nothing else about the exchange:

| Field | Type | Meaning |
|---|---|---|
| `lang` | string | `fil`, `ceb`, `hil`, or the display name (`Filipino`). Case-insensitive. Unknown values are an error. |
| `voice` | string | Speaker id (`FIL_0003`), a gender (`female`), or an index (`0`). Unknown values fall back to the default voice silently. |
| `meta` | bool | Send a `start` frame before the audio. Off by default — see §6. |

Unknown fields are ignored, so adding your own is safe.

Limits: `text` must be a non-empty string of at most **500 characters**, and the
whole message at most **8192 bytes**. Over-length text is rejected, never
truncated.

## 3. Response — audio

Binary WebSocket frames, each a chunk of the waveform. Concatenate them in
order; that is the complete audio, with nothing to strip.

| | |
|---|---|
| Encoding | PCM S16LE — signed 16-bit, little-endian |
| Sample rate | 16000 Hz |
| Channels | 1 (mono) |
| Container | none. No WAV header, no length prefix, no sequence numbers |
| Frame size | **3200 bytes** = 1600 samples = 100 ms. The last frame may be shorter. |

Every frame has an even length, so an int16 sample never straddles two frames.

If 3200 bytes is too large for your receive buffer, say so — it is a one-line
configuration change on our side, and 1024 is ready to go.

Frames are sent as fast as the connection drains them, not paced to realtime,
so buffer them. If your firmware needs a realtime drip instead, tell us and we
will enable pacing.

## 4. Response — completion

After the last audio frame, exactly one text frame:

```
{"type":"end"}
```

Those **14 bytes, byte for byte**. No whitespace, no extra fields. Comparing
the literal string is safe, and we treat changing it as a breaking change.

`end` always means at least one audio frame was sent. If nothing could be
synthesized you get an `error` instead, never an empty `end`.

## 5. Response — errors

```
{"type":"error","message":"<description>"}
```

One terminal frame per request, always: you receive `end` **or** `error`, never
both and never two of either. After an `error` the server sends nothing further
and closes.

The complete set of messages:

| Message | Cause |
|---|---|
| `invalid JSON` | The payload did not parse. |
| `expected a JSON object` | Valid JSON, but not an object. |
| `missing 'text'` | No `text` key, or it was not a string. |
| `'text' is empty` | `text` was empty or only whitespace. |
| `'text' exceeds 500 characters` | Too long. Split it and send two requests. |
| `message exceeds 8192 bytes` | The whole JSON message was too large. |
| `unknown 'lang'; served: ...` | `lang` is not one this instance loaded. The served list is included. |
| `server busy, retry` | Too many connections, or the synthesis queue did not free up within 15 s. Retry. |
| `model still loading, retry` | The instance is warming up. Retry in a few seconds; `/healthz` says when it is ready. |
| `synthesis produced no audio` | The model returned nothing usable for that text. |
| `synthesis failed: <ExceptionType>` | Internal failure. The type name is included; details are in our logs. |
| `unauthorized` | Only if token auth is enabled, which it is not for this demo. |

The close code is always `1000`, on success and on failure alike. Do not read
meaning into it — the JSON frame is the channel that carries meaning.

## 6. Frame ordering

By default the **first frame you receive after sending a request is audio**.
Nothing precedes it. If you would rather have metadata up front, send
`{"text": "...", "meta": true}` and the first frame becomes:

```json
{"type":"start","sample_rate":16000,"channels":1,"format":"pcm_s16le",
 "lang":"fil","voice":"FIL_0003","segments":2}
```

This is opt-in precisely so that a client expecting audio first is never
surprised.

## 7. Cancellation

Close the WebSocket and synthesis stops. No cancel message is needed.

Long text is split into sentence-sized pieces and synthesized one at a time, so
a close takes effect **within one sentence** — typically under 2 seconds,
bounded by roughly 5. A piece already being synthesized when you disconnect
runs to completion internally and is discarded; nothing further is sent.

One caveat, measured rather than assumed. If you perform a **graceful** close
(send a close frame and wait for ours) while audio is still streaming, your
`close()` may block for up to your own close timeout — we measured about 10
seconds. The synthesis itself stops on time; only the handshake is slow, because
the host is CPU-constrained and the event loop is slow to answer while a
synthesis is running.

If you simply drop the socket, which is what most firmware does, you never see
this. Tell us if graceful-close latency matters and we will look at it.

Reusing the socket is fine: after `end` you may send another `{"text": ...}` on
the same connection. One request at a time, though — wait for `end` or `error`
before sending the next, since the protocol has no request id to match audio
against. An idle socket is closed after 60 s, and any connection after 300 s.

## 8. Latency

Measured on the live host, Filipino, warm instance:

| Text | Time to first audio | Audio produced |
|---|---|---|
| 35 characters, one sentence | 3.9 s | 2.06 s |
| 63 characters, one sentence | 8.4 s | 3.84 s |
| 128 characters, two clauses | 5.5 s | 5.52 s |

Synthesis runs at roughly **2x slower than realtime**, so audio does *not* arrive
faster than it plays. **Buffer a whole sentence before playing it** rather than
playing bytes as they arrive, or you will underrun mid-word.

Time to first audio is set by the first sentence, which we keep deliberately
short — note the 128-character case is *faster* to first audio than the
63-character one, because it splits at its comma and the opening segment is
smaller.

The host is a burstable instance, so **please pace your tests** rather than
looping them. Sustained hammering throttles the CPU and these numbers get worse.

Two cold-start cases to plan around:

- A newly started instance takes ~40 s to load models. `/healthz` returns 503
  until it is ready, and `{"ready":true,...}` once it is.
- The service restarts automatically on reboot, but if `/healthz` is
  unreachable, tell us — it is a demo host, not a production cluster.

## 9. Known limits

- **Write numbers as words.** "limang daan", not "500". The training text has
  no verbalized numerals and the tokenizer silently discards characters it
  cannot encode, so digits and `₱` vanish from the audio rather than being read
  out. This is a model limitation, not a transport one.
- Each sentence is synthesized independently, so intonation does not carry
  across sentence boundaries. Expect a slight reset at each full stop.
- These are read-speech baselines, not production voices.
- Three languages are served here — Filipino (`fil`), Cebuano (`ceb`) and
  Hiligaynon (`hil`). Filipino is the default.
- The host is a **burstable free-tier instance**. It is sized for an integration
  test, not for load. Please space your requests out.
- One synthesis runs at a time per instance. Concurrent requests queue for up
  to 15 s, then get `server busy, retry`.

## 10. Receive loop

Pseudocode for the whole exchange:

```c
ws = ws_connect("ws://104.208.87.149:8765/tts");   // plain ws, no TLS
ws_send_text(ws, "{\"text\":\"Magandang umaga po.\"}");

for (;;) {
    frame = ws_recv(ws);

    if (frame.is_binary) {
        // PCM S16LE, 16 kHz, mono. Hand straight to the DAC or buffer it.
        // frame.len is always even; do not assume 3200 for the last one.
        audio_write(frame.data, frame.len);
        continue;
    }

    // Text frame: this request is over, exactly one of these arrives.
    if (memcmp(frame.data, "{\"type\":\"end\"}", 14) == 0) {
        audio_flush();
        break;
    }
    log_error(frame.data);      // {"type":"error","message":"..."}
    break;
}

// To stop early, just close. No cancel message exists.
ws_close(ws, 1000);
```

Python equivalent, which is also the reference client in `tools/ws_client.py`:

```python
import json, websockets

async with websockets.connect(URL, max_size=None) as ws:
    await ws.send(json.dumps({"text": "Magandang umaga po."}))
    pcm = b""
    while True:
        msg = await ws.recv()
        if isinstance(msg, bytes):
            pcm += msg
            continue
        print(json.loads(msg))       # {"type": "end"} or an error
        break
# pcm is now raw PCM S16LE @ 16 kHz mono
```

## 11. Questions we need answered

TLS is settled — this endpoint is plain `ws://`, so that question is gone. Four
left, and they only affect defaults we can change on our side:

1. Is the handshake a standard HTTP/1.1 upgrade with `Sec-WebSocket-Key`?
2. How large is your receive buffer — is 3200 bytes per frame fine, or should we
   ship 1024?
3. Does your client reply to WebSocket ping frames automatically? We ping every
   20 s and allow 60 s to respond.
4. Do you want frames paced to realtime, or will you buffer a burst? Right now we
   send as fast as the connection drains.

## 12. Testing from the command line

`curl` cannot do this — it has no `ws`/`wss` protocol in most builds, including
Apple's, and its CLI cannot send a message then read frames. Use `websocat`
(`brew install websocat`, or your distro's package).

Full exchange, showing the audio frame count and the terminal message:

```bash
echo '{"text":"Hello, this is a Sapin TTS test."}' \
 | timeout 30 websocat -t -n --base64 --binary-prefix 'AUDIO ' --text-prefix 'JSON ' \
   ws://104.208.87.149:8765/tts \
 | awk '/^AUDIO/{n++} /^JSON/{printf "%d audio frames, then %s\n", n, substr($0,6); fflush(); exit}'
```

```
16 audio frames, then {"type":"end"}
```

Two flags are load-bearing. **`-n`** (`--no-close`) stops websocat sending a
close frame when stdin ends — without it the server correctly treats EOF as a
cancel and you get nothing. **`timeout`** is needed because we keep the socket
open for 60 s after `end` so you can send another request; the output appears
immediately, only the shell waits.

Save the raw PCM exactly as the device receives it:

```bash
echo '{"text":"Hello, this is a Sapin TTS test."}' \
 | timeout 30 websocat -t -n --base64 --binary-prefix 'AUDIO ' --text-prefix 'JSON ' \
   ws://104.208.87.149:8765/tts \
 | awk '/^AUDIO/{print substr($0,7)} /^JSON/{exit}' | base64 -d > sapin.pcm

ffplay -f s16le -ar 16000 -ac 1 -i sapin.pcm
```

Those `ffplay` flags are required precisely because there is no WAV header.

Error cases return on the first frame, so they need no tricks:

```bash
echo '{"text":""}'     | websocat -t -1 ws://104.208.87.149:8765/tts
echo 'not json'        | websocat -t -1 ws://104.208.87.149:8765/tts
```
