"""
Compute dataset statistics for the UP-DSP Philippine Language Dataset (PLD).

Reads session .log files and WAV headers directly from the extracted corpus —
run before or after processing. Durations come from WAV headers (sf.info), so
no audio is decoded.

Usage:
  python stats_pld.py
  python stats_pld.py --languages BIK,CEB
"""

import argparse
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf
from dotenv import load_dotenv

from halolib.pld import index_corpus

load_dotenv(Path(__file__).parent / ".env")

PLD_DIR = Path(os.environ.get(
    "PLD_DIR", "/mnt/d/backup/dsp_bkp/Speech_Corpora/PLD_raw/PLD"))


def histogram(values, bins):
    counts = defaultdict(int)
    for v in values:
        for lo, hi in bins:
            if lo <= v < hi:
                counts[f"{lo}-{hi}"] += 1
                break
    return counts


def percentile(values, p):
    s = sorted(values)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


def bar(count, total, width=40):
    return "█" * (count * width // total) if total else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--languages", help="comma-separated dir names, e.g. BIK,CEB")
    ap.add_argument("--workers", type=int, default=(os.cpu_count() or 4) * 4)
    args = ap.parse_args()

    langs = args.languages.split(",") if args.languages else None

    print(f"Corpus: {PLD_DIR}")
    print("Indexing session logs...")
    entries, counts = index_corpus(PLD_DIR, langs)
    print(f"  sessions: {counts['sessions']:,} | utterances: {counts['rows']:,} "
          f"| missing wav: {counts['missing_wav']:,}")

    if not entries:
        print("No entries found — is the corpus extracted?")
        return

    print("Reading WAV headers...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        infos = list(pool.map(lambda e: sf.info(e["wav_path"]), entries))
    for e, i in zip(entries, infos):
        e["duration"] = i.duration
        e["samplerate"] = i.samplerate

    durations = [e["duration"] for e in entries]
    total = len(durations)
    hours = sum(durations) / 3600

    by_lang = defaultdict(list)
    by_type = defaultdict(list)
    by_cat = defaultdict(int)
    speakers = defaultdict(set)
    gender = defaultdict(int)
    ages = []
    srs = defaultdict(int)

    for e in entries:
        by_lang[e["language_name"]].append(e["duration"])
        by_type[e["speech_type"]].append(e["duration"])
        by_cat[e["prompt_category"]] += 1
        speakers[e["language_name"]].add(e["speaker_id"])
        gender[e["gender"]] += 1
        srs[e["samplerate"]] += 1
        if e["age"] > 0:
            ages.append(e["age"])

    all_speakers = {s for v in speakers.values() for s in v}
    usable = [e for e in entries if not e["text_is_prompt"]]

    print("\n" + "=" * 62)
    print("PHILIPPINE LANGUAGE DATASET (PLD) — STATISTICS")
    print("=" * 62)
    print(f"\nTotal utterances   : {total:,}")
    print(f"Total duration     : {hours:,.2f} hours")
    print(f"Total speakers     : {len(all_speakers):,}")
    print(f"Languages          : {len(by_lang)}")
    print(f"Sample rates       : {dict(srs)}")
    print(f"\nUsable (audio,text) pairs : {len(usable):,} "
          f"({sum(e['duration'] for e in usable)/3600:,.2f} h)")
    print(f"Prompt-only rows (spontaneous; text is the elicitation question,")
    print(f"  not a transcript)       : {total - len(usable):,} "
          f"({(sum(durations) - sum(e['duration'] for e in usable))/3600:,.2f} h)")

    print(f"\n--- Per Language ---")
    print(f"  {'language':16s} {'utts':>9s} {'hours':>9s} {'spk':>6s}")
    for name, durs in sorted(by_lang.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {name:16s} {len(durs):9,} {sum(durs)/3600:9.2f} "
              f"{len(speakers[name]):6,}")

    print(f"\n--- Speech Type ---")
    print(f"  {'type':14s} {'utts':>9s} {'hours':>9s} {'mean':>8s}")
    for t, durs in sorted(by_type.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {t:14s} {len(durs):9,} {sum(durs)/3600:9.2f} "
              f"{sum(durs)/len(durs):7.2f}s")

    print(f"\n--- Duration (seconds) ---")
    print(f"  Mean {sum(durations)/total:.2f}s | Min {min(durations):.2f}s | "
          f"Max {max(durations):.2f}s")
    print(f"  p25 {percentile(durations,25):.2f}s | Median {percentile(durations,50):.2f}s | "
          f"p75 {percentile(durations,75):.2f}s | p95 {percentile(durations,95):.2f}s")
    dur_bins = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 30), (30, 120)]
    print(f"\n  Distribution:")
    for bucket, count in histogram(durations, dur_bins).items():
        print(f"    {bucket:8s}s : {count:8,}  {bar(count, total)}")

    words = [e["num_words"] for e in usable]
    if words:
        print(f"\n--- Words per Utterance (usable rows) ---")
        print(f"  Mean {sum(words)/len(words):.1f} | Median {percentile(words,50)} | "
              f"p95 {percentile(words,95)} | Max {max(words)}")

    print(f"\n--- Gender (utterances) ---")
    for k, v in sorted(gender.items(), key=lambda kv: -kv[1]):
        print(f"  {k:8s} : {v:8,}  ({v/total*100:5.1f}%)")

    if ages:
        print(f"\n--- Speaker Age ---")
        print(f"  Mean {sum(ages)/len(ages):.1f} | Min {min(ages)} | Max {max(ages)}")

    print(f"\n--- Top Prompt Categories ---")
    for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {cat:24s} : {n:8,}")
    print("=" * 62)


if __name__ == "__main__":
    main()
