"""
Dataset statistics for the livestream corpus — reads manifests, not audio.
Run after any stage; reports whatever columns exist so far.

  python stats_livestream.py
"""

import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from halolib import manifest as hm
from halolib.qc import ASR_GATE, TTS_GATE, passes_gate

load_dotenv(Path(__file__).parent / ".env")

LIVESTREAM_DIR = Path(os.environ["LIVESTREAM_DIR"])
MANIFEST_DIR   = LIVESTREAM_DIR.parent / "manifests"


def histogram(values, bins):
    counts = defaultdict(int)
    for v in values:
        for lo, hi in bins:
            if lo <= v < hi:
                counts[f"{lo}-{hi}"] += 1
                break
    return dict(counts)


def percentile(values, p):
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def pstats(label, values, fmt="{:.2f}"):
    if not values:
        print(f"  {label}: (no data)")
        return
    print(f"  {label}: mean {fmt.format(sum(values)/len(values))} | "
          f"p25 {fmt.format(percentile(values, 25))} | "
          f"p50 {fmt.format(percentile(values, 50))} | "
          f"p75 {fmt.format(percentile(values, 75))} | "
          f"p95 {fmt.format(percentile(values, 95))}")


rows = []
for mpath in sorted(MANIFEST_DIR.glob("*.jsonl")):
    spath = hm.status_path(MANIFEST_DIR, mpath.stem)
    entry = hm.get_stage(spath, "parse")
    if entry is None or entry.get("duplicate_of"):
        continue
    rows.extend(hm.read_manifest(mpath))

if not rows:
    raise SystemExit("No manifests found — run the parse stage first.")

total = len(rows)
files = {r["file_id"] for r in rows}
speakers = {r["speaker_id"] for r in rows}
durations = [r["duration"] for r in rows]
total_hours = sum(durations) / 3600

by_alignment = defaultdict(int)
for r in rows:
    by_alignment[r.get("alignment", "?")] += 1

print("=" * 60)
print("LIVESTREAM CORPUS — DATASET STATISTICS")
print("=" * 60)
print(f"\nSource files       : {len(files)}")
print(f"Total segments     : {total:,}")
print(f"Total duration     : {total_hours:.2f} hours")
print(f"Total speakers     : {len(speakers)}")

print("\n--- Alignment status ---")
for k, v in sorted(by_alignment.items()):
    bar = "█" * (v * 40 // total)
    print(f"  {k:14s} : {v:6,}  ({v/total*100:.1f}%)  {bar}")

scores = [r["align_score"] for r in rows if r.get("align_score") is not None]
if scores:
    print("\n--- Forced-alignment score ---")
    pstats("align_score", scores)

print("\n--- Duration (seconds) ---")
pstats("duration", durations, "{:.2f}s")
dur_bins = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 15), (15, 30), (30, 999)]
for bucket, count in histogram(durations, dur_bins).items():
    bar = "█" * (count * 40 // total)
    print(f"    {bucket:8s}s : {count:6,}  {bar}")

cers = [r["asr_cer"] for r in rows if r.get("asr_cer") is not None]
if cers:
    print("\n--- ASR round-trip CER ---")
    pstats("asr_cer (all)", cers, "{:.3f}")
    forced = [r["asr_cer"] for r in rows
              if r.get("asr_cer") is not None and r.get("alignment") == "forced"]
    interp = [r["asr_cer"] for r in rows
              if r.get("asr_cer") is not None and r.get("alignment") == "interpolated"]
    pstats("asr_cer (forced)", forced, "{:.3f}")
    pstats("asr_cer (interpolated)", interp, "{:.3f}")

ratios = [r["speech_ratio"] for r in rows if r.get("speech_ratio") is not None]
if ratios:
    print("\n--- Speech ratio (VAD coverage) ---")
    pstats("speech_ratio", ratios, "{:.3f}")

snrs = [r["snr_db"] for r in rows if r.get("snr_db") is not None]
if snrs:
    print("\n--- SNR proxy (dB) ---")
    pstats("snr_db", snrs, "{:.1f}")

overlaps = sum(1 for r in rows if r.get("overlap"))
if any("overlap" in r for r in rows):
    print(f"\nOverlap-flagged    : {overlaps:,}  ({overlaps/total*100:.1f}%)")

print("\n--- Speakers ---")
by_speaker = defaultdict(float)
for r in rows:
    by_speaker[r["speaker_id"]] += r["duration"]
for spk, dur in sorted(by_speaker.items(), key=lambda kv: -kv[1])[:15]:
    print(f"  {spk[-20:]:22s} : {dur/60:6.1f} min")
if len(by_speaker) > 15:
    print(f"  ... and {len(by_speaker) - 15} more")

print("\n--- Split ---")
by_split = defaultdict(int)
for r in rows:
    by_split[r["split"]] += 1
for k, v in sorted(by_split.items()):
    print(f"  {k:6s} : {v:6,}  ({v/total*100:.1f}%)")

print("\n--- Export gates (current thresholds) ---")
n_asr = sum(1 for r in rows if passes_gate(
    {**r, "speech_type": "spontaneous", "source": r["file_id"]}, ASR_GATE))
n_tts = sum(1 for r in rows if passes_gate(
    {**r, "speech_type": "spontaneous", "source": r["file_id"]}, TTS_GATE))
print(f"  ASR set : {n_asr:6,}  ({n_asr/total*100:.1f}%)")
print(f"  TTS set : {n_tts:6,}  ({n_tts/total*100:.1f}%)")
print("=" * 60)
