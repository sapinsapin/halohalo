// fastText language ID in WebAssembly (fasttext.wasm.js: the fastText C++
// library compiled to wasm; the same build runs in browsers and workers).
//   node lid.mjs job.json   job = {"model": "/path/model.ftz", "texts": [...]}
// Prints one JSON line {"labels": [...], "probs": [...]}.
import fs from "node:fs";
import { pathToFileURL } from "node:url";
import { getFastTextClass, getFastTextModule } from "fasttext.wasm.js";

const job = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const FastText = await getFastTextClass({ getFastTextModule });
const model = await new FastText().loadModel(pathToFileURL(job.model).href);
const labels = [], probs = [];
for (const t of job.texts) {
  const v = model.predict(t.replaceAll("\n", " "), 1, 0.0);
  if (v.size() > 0) {
    const [p, l] = v.get(0);
    labels.push(l); probs.push(p);
  } else { labels.push(null); probs.push(0); }
}
console.log(JSON.stringify({ labels, probs }));
