// The wav2vec2 CTC model's frontend and decoder, in plain JavaScript. Shared
// by the Node validator and the browser page, and identical to
// porting/frontends.py: normalise the real audio, optionally cut it into
// fixed windows, argmax each frame, collapse repeats, drop blanks, '|' -> ' '.
// No tokenizer library needed: the vocabulary is 32 characters.

export const SR = 16000;
export const STRIDE = 320;          // one frame per 20 ms
const BLANK = 0;

export function normalise(audio, doNormalize = true) {
  const x = Float32Array.from(audio);
  if (!doNormalize) return x;
  let mean = 0;
  for (const v of x) mean += v;
  mean /= x.length || 1;
  let varr = 0;
  for (const v of x) varr += (v - mean) ** 2;
  varr /= x.length || 1;
  const s = Math.sqrt(varr + 1e-7);
  for (let i = 0; i < x.length; i++) x[i] = (x[i] - mean) / s;
  return x;
}

export function id2unit(vocabJson) {
  let v = vocabJson;
  const first = Object.values(v)[0];
  if (first && typeof first === "object") v = first;       // {lang: {...}} form
  const m = new Map();
  for (const [u, i] of Object.entries(v)) m.set(i, u);
  return m;
}

export function greedy(frameIds, units) {
  let out = "", prev = null;
  for (const i of frameIds) {
    if (i !== prev && i !== BLANK) out += units.get(i) ?? "";
    prev = i;
  }
  return out.replaceAll("|", " ").trim();
}

function argmaxFrames(data, T, V, valid) {
  const ids = [];
  for (let t = 0; t < Math.min(T, valid ?? T); t++) {
    let best = 0, bv = -Infinity;
    for (let k = 0; k < V; k++) {
      const v = data[t * V + k];
      if (v > bv) { bv = v; best = k; }
    }
    ids.push(best);
  }
  return ids;
}

// run(Float32Array input) -> {data, dims: [1, T, V]}; windowS null = one pass
export async function transcribe(run, audio, units, { doNormalize = true, windowS = null } = {}) {
  const x = normalise(audio, doNormalize);
  const ids = [];
  if (!windowS) {
    const { data, dims } = await run(x);
    ids.push(...argmaxFrames(data, dims[1], dims[2]));
  } else {
    const n = windowS * SR;
    for (let s = 0; s < Math.max(x.length, 1); s += n) {
      const chunk = x.subarray(s, s + n);
      const w = new Float32Array(n);
      w.set(chunk);
      const { data, dims } = await run(w);
      ids.push(...argmaxFrames(data, dims[1], dims[2], Math.max(1, Math.ceil(chunk.length / STRIDE))));
    }
  }
  return greedy(ids, units);
}
