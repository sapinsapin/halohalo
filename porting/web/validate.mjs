// Node validation of the onnx-web folder, loaded exactly as a page loads it.
//   node validate.mjs job.json      (written by porting/validate.py)
// Whisper: Transformers.js pipeline. wav2vec2 CTC: onnxruntime-node + ctc.mjs.
// Prints one JSON line {"hyps": [...], "load_ms", "run_ms"}.
import fs from "node:fs";
import path from "node:path";
import { transcribe, id2unit } from "./ctc.mjs";

const job = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const buf = fs.readFileSync(job.audio);
const flat = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const clips = [];
let off = 0;
for (const n of job.lengths) { clips.push(flat.slice(off, off + n)); off += n; }

const TJS_DTYPE = { fp32: "fp32", int8: "q8", fp16: "fp16" };
const SUFFIX = { fp32: "", int8: "_quantized", fp16: "_fp16" };
const hyps = [];
const t0 = performance.now();
let t1 = null;          // set once the model is loaded

if (job.family === "whisper") {
  const { pipeline, env } = await import("@huggingface/transformers");
  env.allowRemoteModels = false;
  env.localModelPath = path.dirname(job.model_dir) + path.sep;
  const asr = await pipeline("automatic-speech-recognition", path.basename(job.model_dir), {
    dtype: TJS_DTYPE[job.dtype], device: "cpu",
    session_options: { intraOpNumThreads: job.threads },
  });
  t1 = performance.now();
  for (const a of clips) {
    // 225 new tokens, as the Python reference generates
    const r = await asr(a, { language: job.language, task: "transcribe", return_timestamps: false,
                             max_new_tokens: 225 });
    hyps.push(r.text.trim());
  }
} else {
  const ort = await import("onnxruntime-node");
  const graph = path.join(job.model_dir, "onnx", `model${SUFFIX[job.dtype]}.onnx`);
  const sess = await ort.InferenceSession.create(graph, {
    intraOpNumThreads: job.threads, graphOptimizationLevel: "all" });
  const pre = JSON.parse(fs.readFileSync(path.join(job.model_dir, "preprocessor_config.json"), "utf8"));
  const units = id2unit(JSON.parse(fs.readFileSync(path.join(job.model_dir, "vocab.json"), "utf8")));
  const input = sess.inputNames[0], output = sess.outputNames[0];
  const run = async (x) => {
    const out = await sess.run({ [input]: new ort.Tensor("float32", x, [1, x.length]) });
    return { data: out[output].data, dims: out[output].dims };
  };
  t1 = performance.now();
  for (const a of clips)
    hyps.push(await transcribe(run, a, units, { doNormalize: pre.do_normalize ?? true }));
}
const t2 = performance.now();
console.log(JSON.stringify({ hyps, load_ms: t1 - t0, run_ms: t2 - t1 }));
