"""Encode a frozen Filipino waveform export (``curation.tts_export``) into MOSS codes with reference voices (GPU).

    python -m training.fil_moss.encode --export /abs/new-fil-run/data --recipe training/fil_moss/recipe.json \
        --output /abs/new-fil-run/codes

For every split: re-verify audio hashes, encode 16 codebooks, and attach ``ref_audio_codes`` from a different
clip of the same speaker in the same split (deterministic choice), so training is multi-speaker without baking a
voice into the weights and no reference ever crosses a split. Rows whose speaker has one clip in the split keep
no reference. The first clips of each split are decoded back and checked (finite, not silent, duration).
The output directory must be new.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from . import runtime as R
from .scoring import silence_fraction


def choose_references(rows, seed):
    """{row id: reference row id or None}: another clip by the same speaker in the same split."""
    by_speaker = {}
    for r in rows:
        by_speaker.setdefault(r["speaker_id"], []).append(r["id"])
    out = {}
    for r in rows:
        others = sorted((i for i in by_speaker[r["speaker_id"]] if i != r["id"]),
                        key=lambda i: hashlib.sha256(("%s|%s|%s" % (seed, r["id"], i)).encode()).hexdigest())
        out[r["id"]] = others[0] if others else None
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--export", required=True)
    p.add_argument("--recipe", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)
    recipe = json.loads(Path(a.recipe).read_text())
    enc_cfg, seed = recipe["encode"], recipe["baseline"]["seed"]
    export = Path(a.export)
    report = json.loads((export / "export_report.json").read_text())
    if report.get("codec_encoded") is not False or report.get("training_launched") is not False:
        raise SystemExit("export report is not a fresh waveform export")
    out = R.new_output_dir(a.output)
    R.setup()
    codec = R.load_codec()
    summary = {"export": str(export), "export_manifests_sha256": report["manifests_sha256"], "splits": {},
               "codec_revision": recipe["codec"]["revision"], "reference_rule": enc_cfg["reference_conditioning"]}
    seen_ids = {}
    for split in ("train", "validation", "test"):
        manifest = export / ("%s.jsonl" % split)
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != report["manifests_sha256"][split]:
            raise SystemExit("%s manifest changed since export" % split)
        rows = []
        for rec in R.read_jsonl(manifest):
            if rec["id"] in seen_ids:
                raise SystemExit("id %s appears in %s and %s" % (rec["id"], seen_ids[rec["id"]], split))
            seen_ids[rec["id"]] = split
            turn = rec["conversations"][0]
            path = Path(turn["wav"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != rec["provenance"]["audio_sha256"]:
                raise SystemExit("audio changed since export: %s" % path)
            wav, sr = sf.read(path, dtype="float32")
            if sr != enc_cfg["sample_rate"] or wav.ndim != 1:
                raise SystemExit("unexpected waveform format for %s" % path)
            rows.append({"id": rec["id"], "speaker_id": rec["provenance"]["canonical_row"]["speaker_id"],
                         "text": turn["text"], "wav": str(path), "seconds": len(wav) / sr,
                         "codes": R.encode_wav(codec, wav, enc_cfg["n_vq"]), "provenance": rec["provenance"]})
        refs = choose_references(rows, seed)
        by_id = {r["id"]: r for r in rows}
        checks = []
        for r in rows[:enc_cfg["reconstruction_check_clips_per_split"]]:
            audio = R.decode(codec, r["codes"])
            secs = len(audio) / enc_cfg["sample_rate"]
            ok = (bool(np.isfinite(audio).all()) and silence_fraction(audio.tolist(), enc_cfg["sample_rate"]) < 0.95
                  and abs(secs - r["seconds"]) <= enc_cfg["reconstruction_duration_tolerance_seconds"])
            checks.append({"id": r["id"], "seconds": r["seconds"], "decoded_seconds": secs, "ok": ok})
            if not ok:
                raise SystemExit("codec reconstruction check failed: %s" % checks[-1])
        lines = []
        for r in rows:
            ref = by_id[refs[r["id"]]] if refs[r["id"]] else None
            lines.append(json.dumps({
                "id": r["id"], "speaker_id": r["speaker_id"], "split": split,
                "conversations": [{"role": "assistant", "text": r["text"], "audio_codes": r["codes"].tolist(),
                                   "wav": r["wav"]}],
                "seconds": r["seconds"],
                "ref_audio_codes": None if ref is None else ref["codes"].tolist(),
                "ref_id": None if ref is None else ref["id"], "ref_wav": None if ref is None else ref["wav"],
                "audio_sha256": r["provenance"]["audio_sha256"]}, ensure_ascii=False))
        (out / ("%s_codes.jsonl" % split)).write_text("\n".join(lines) + "\n")
        summary["splits"][split] = {"rows": len(rows), "with_reference": sum(1 for v in refs.values() if v),
                                    "speakers": len({r["speaker_id"] for r in rows}),
                                    "hours": sum(r["seconds"] for r in rows) / 3600, "reconstruction": checks}
        print(json.dumps({split: {k: v for k, v in summary["splits"][split].items() if k != "reconstruction"}}),
              flush=True)
    R.write_json(out / "encode_report.json", summary)
    print("ENCODING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
