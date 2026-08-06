"""
Diarized Livestream Corpus → TTS/ASR-ready datasets (staged pipeline driver)

Stages (each incremental — files already processed with the same config skip):
  parse   — unzip, parse JSON, decode 16kHz WAV cache, fingerprint-dedup,
            baseline (interpolated) per-turn timings → manifests/{id}.jsonl
  align   — forced alignment (MMS CTC) per block window + silero VAD snapping;
            upgrades timings in the manifest              [GPU ~1.6GB]
  qc      — faster-whisper round-trip CER, loudness/clipping/SNR, overlap flag
                                                          [GPU ~4GB]
  export  — apply ASR/TTS gates, write audiofolder (16k asr/, 24k tts/) and,
            with --push, stream parquet shards to the Hub

Usage:
  python process_livestream.py --stages parse,align,qc,export
  python process_livestream.py --stages export --push
  python process_livestream.py --stages align --force align --file-id <ID>

Manifests live in {LIVESTREAM_DIR}/../manifests; audiofolder output in
{LIVESTREAM_OUTPUT_DIR}/asr and /tts. See docs/livestream_pipeline.md.

Requires ffmpeg on PATH. GPU stages additionally require the alignment/QC
stack (torch, ctc-forced-aligner, faster-whisper, silero-vad — see
requirements.txt).
"""

import argparse
import hashlib
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from halolib import audio as ha
from halolib import manifest as hm
from halolib.sources import livestream as ls

load_dotenv(Path(__file__).parent / ".env")

LIVESTREAM_DIR = Path(os.environ["LIVESTREAM_DIR"])
OUTPUT_DIR     = Path(os.environ["LIVESTREAM_OUTPUT_DIR"])
MANIFEST_DIR   = LIVESTREAM_DIR.parent / "manifests"
SHARD_WORKDIR  = LIVESTREAM_DIR.parent / "shards"
DEFAULT_REPO   = os.environ.get("LIVESTREAM_HF_REPO", "sapinsapin/halo-livestream")

SPLIT_MOD   = 10        # md5(file_id) % 10 == 0 -> test  (file-level, speaker-disjoint)
SHARD_SIZE  = 2000
NUM_WORKERS = os.cpu_count() or 4

PARSE_CFG = {"version": 2, "min_chars": ls.MIN_CHARS, "split_mod": SPLIT_MOD}

log = logging.getLogger("livestream")


def setup_logging():
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(MANIFEST_DIR / "run.log"),
            logging.StreamHandler(),
        ],
    )


def assign_split(file_id: str) -> str:
    digest = int(hashlib.md5(file_id.encode()).hexdigest(), 16)
    return "test" if digest % SPLIT_MOD == 0 else "train"


def wav16_path(audio_src: Path, file_id: str) -> Path:
    return audio_src.with_name(f"{file_id}_16k.wav")


def wav24_path(audio_src: Path, file_id: str) -> Path:
    return audio_src.with_name(f"{file_id}_24k.wav")


# ---------------------------------------------------------------- parse stage

def _parse_one(task: tuple) -> tuple[str, str, list[dict] | None]:
    """Decode + parse one source file. Returns (file_id, fingerprint, rows)."""
    file_id, json_path, audio_path = task
    doc = ls.parse_doc(json_path)

    wav16 = wav16_path(audio_path, file_id)
    if not ha.decode_audio(audio_path, wav16, ha.ASR_SR):
        return file_id, "", None
    fp = ha.fingerprint(wav16)

    split = assign_split(file_id)
    rows, idx = [], 0
    for block in doc.blocks:
        for seg in ls.interpolate_block(block):
            t = seg["turn"]
            spk = doc.speakers.get(t.speaker, {})
            rows.append({
                "file_id":     file_id,
                "idx":         idx,
                "block_idx":   block.block_idx,
                "turn_idx":    t.turn_idx,
                "block_start": block.start,
                "block_end":   block.end,
                "speaker_raw": t.speaker,
                "speaker_id":  ls.namespace_speaker(file_id, t.speaker),
                "gender":      spk.get("gender", "unknown"),
                "role":        spk.get("role", ""),
                "sentence":    t.text,
                "language":    doc.language,
                "start":       round(seg["start"], 3),
                "end":         round(seg["end"], 3),
                "duration":    round(seg["end"] - seg["start"], 3),
                "alignment":   seg["alignment"],
                "split":       split,
                "audio_src":   str(audio_path),
                "wav16":       str(wav16),
            })
            idx += 1
    return file_id, fp, rows


def stage_parse(pairs: list[tuple], args) -> None:
    cfg = hm.config_hash(PARSE_CFG)
    fp_registry_path = MANIFEST_DIR / "fingerprints.json"
    fp_registry = {}
    if fp_registry_path.exists():
        fp_registry = json.loads(fp_registry_path.read_text())

    pending = [
        p for p in pairs
        if not hm.stage_done(hm.status_path(MANIFEST_DIR, p[0]), "parse", cfg)
    ]
    log.info(f"parse: {len(pending)} pending / {len(pairs)} total")
    if not pending:
        return

    with ThreadPoolExecutor(max_workers=args.num_proc) as pool:
        for file_id, fp, rows in pool.map(_parse_one, pending):
            spath = hm.status_path(MANIFEST_DIR, file_id)
            if rows is None:
                log.warning(f"  [fail] {file_id}: decode failed")
                continue

            dup_of = fp_registry.get(fp)
            if fp and dup_of and dup_of != file_id:
                log.warning(f"  [dup] {file_id} is a duplicate of {dup_of} — skipping")
                hm.mark_stage(spath, "parse", cfg, duplicate_of=dup_of, rows=0)
                continue

            if fp:
                fp_registry[fp] = file_id
            hm.write_manifest(hm.manifest_path(MANIFEST_DIR, file_id), rows)
            hm.mark_stage(spath, "parse", cfg, rows=len(rows))
            n_exact = sum(1 for r in rows if r["alignment"] == "exact")
            log.info(f"  {file_id}: {len(rows)} turns ({n_exact} exact) → {rows[0]['split']}")

    fp_registry_path.write_text(json.dumps(fp_registry, indent=2))


# ---------------------------------------------------------------- align stage

def stage_align(pairs: list[tuple], args) -> None:
    from halolib.align import ALIGN_CFG, run_align_stage
    cfg = hm.config_hash(ALIGN_CFG)
    pending = _pending_after(pairs, prereq="parse", stage="align", cfg=cfg)
    log.info(f"align: {len(pending)} pending / {len(pairs)} total")
    if pending:
        run_align_stage(pending, MANIFEST_DIR, cfg, num_proc=args.num_proc)


# ------------------------------------------------------------------- qc stage

def stage_qc(pairs: list[tuple], args) -> None:
    from halolib.qc import qc_config, run_qc_stage
    cfg = hm.config_hash(qc_config(args.whisper_model))
    pending = _pending_after(pairs, prereq="align", stage="qc", cfg=cfg)
    log.info(f"qc: {len(pending)} pending / {len(pairs)} total")
    if pending:
        run_qc_stage(pending, MANIFEST_DIR, cfg,
                     whisper_model=args.whisper_model, num_proc=args.num_proc)


def _pending_after(pairs: list[tuple], prereq: str, stage: str, cfg: str) -> list[tuple]:
    """Files whose prereq stage is done (any config) but `stage` isn't (this config)."""
    out = []
    for p in pairs:
        spath = hm.status_path(MANIFEST_DIR, p[0])
        prereq_entry = hm.get_stage(spath, prereq)
        if prereq_entry is None or prereq_entry.get("duplicate_of"):
            continue
        if not hm.stage_done(spath, stage, cfg):
            out.append(p)
    return out


# --------------------------------------------------------------- export stage

def _load_all_rows(pairs: list[tuple]) -> list[dict]:
    rows = []
    for file_id, _, _ in pairs:
        spath = hm.status_path(MANIFEST_DIR, file_id)
        parse_entry = hm.get_stage(spath, "parse")
        if parse_entry is None or parse_entry.get("duplicate_of"):
            continue
        rows.extend(hm.read_manifest(hm.manifest_path(MANIFEST_DIR, file_id)))
    return rows


def _slice_group(task: tuple) -> list[dict]:
    """Slice all of one file's gated segments from its cache WAV. Worker-safe."""
    wav_path, rows, sr, audio_dir = task
    import soundfile as sf
    out = []
    for row in rows:
        data, file_sr = ha.read_window(Path(wav_path), row["start"], row["end"])
        if data is None or len(data) == 0:
            continue
        meta = {k: row.get(k) for k in EXPORT_FIELDS}
        if audio_dir is not None:                      # audiofolder mode
            clip = f"{row['file_id']}_{row['idx']:04d}.wav"
            sf.write(Path(audio_dir) / clip, data, sr)
            out.append({"file_name": f"audio/{clip}", **meta})
        else:                                          # parquet mode
            out.append({"audio": {"bytes": ha.wav_bytes(data, sr), "path": None}, **meta})
    return out


EXPORT_FIELDS = [
    "sentence", "language", "duration", "speaker_id", "gender", "role",
    "speech_type", "source", "start", "end", "alignment", "align_score",
    "asr_cer", "overlap", "speech_ratio", "snr_db", "lufs", "clip_ratio",
]


def _finalize_row(row: dict) -> dict:
    """Fill derived/optional fields so exports have a stable schema."""
    return {
        **row,
        "speech_type": "spontaneous",
        "source":      row["file_id"],
        "align_score": row.get("align_score"),
        "asr_cer":     row.get("asr_cer"),
        "overlap":     row.get("overlap"),
        "speech_ratio": row.get("speech_ratio"),
        "snr_db":      row.get("snr_db"),
        "lufs":        row.get("lufs"),
        "clip_ratio":  row.get("clip_ratio"),
    }


def _export_audiofolder(rows: list[dict], subset: str, sr: int, wav_key: str, args):
    from halolib.shard import rows_to_dataset  # noqa: F401 (schema parity check)
    base = OUTPUT_DIR / subset
    written = 0
    for split in ("train", "test"):
        split_rows = [r for r in rows if r["split"] == split]
        if not split_rows:
            continue
        split_dir = base / split
        audio_dir = split_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        groups: dict[str, list[dict]] = {}
        for r in split_rows:
            groups.setdefault(r[wav_key], []).append(r)
        tasks = [(wav, grp, sr, str(audio_dir)) for wav, grp in groups.items()]

        with open(split_dir / "metadata.jsonl", "w", encoding="utf-8") as f, \
             ProcessPoolExecutor(max_workers=args.num_proc) as pool:
            for metas in pool.map(_slice_group, tasks):
                for m in metas:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
                    written += 1
    log.info(f"  {subset}: {written} clips → {base}")
    return written


def _export_parquet(rows: list[dict], subset: str, sr: int, wav_key: str, args):
    from halolib.shard import ShardStreamer
    token = os.environ["HF_TOKEN"]
    with ShardStreamer(args.repo, token, SHARD_WORKDIR,
                       prefix=f"data/{subset}", shard_size=SHARD_SIZE) as streamer:
        completed = streamer.completed()
        for split in ("train", "test"):
            split_rows = [r for r in rows if r["split"] == split]
            if not split_rows:
                continue
            n_shards = (len(split_rows) + SHARD_SIZE - 1) // SHARD_SIZE
            for shard_idx in range(n_shards):
                if f"data/{subset}/{split}/{shard_idx}" in completed:
                    log.info(f"  {subset}/{split} shard {shard_idx+1}/{n_shards} done, skipping")
                    continue
                batch = split_rows[shard_idx * SHARD_SIZE:(shard_idx + 1) * SHARD_SIZE]
                groups: dict[str, list[dict]] = {}
                for r in batch:
                    groups.setdefault(r[wav_key], []).append(r)
                tasks = [(wav, grp, sr, None) for wav, grp in groups.items()]
                with ProcessPoolExecutor(max_workers=args.num_proc) as pool:
                    shard_rows = [r for rs in pool.map(_slice_group, tasks) for r in rs]
                streamer.write_shard(split, shard_idx, n_shards, shard_rows, sr)


def stage_export(pairs: list[tuple], args) -> None:
    from halolib.qc import ASR_GATE, TTS_GATE, passes_gate

    all_rows = [_finalize_row(r) for r in _load_all_rows(pairs)]
    if not all_rows:
        log.warning("export: no manifest rows found — run parse first")
        return

    asr_rows = [r for r in all_rows if passes_gate(r, ASR_GATE)]
    tts_rows = [r for r in all_rows if passes_gate(r, TTS_GATE)]
    log.info(f"export: {len(all_rows)} rows → ASR {len(asr_rows)}, TTS {len(tts_rows)}")

    # TTS needs the 24k cache — decode lazily, once per source file
    tts_files = {(r["audio_src"], r["file_id"]) for r in tts_rows}
    for audio_src, file_id in tts_files:
        w24 = wav24_path(Path(audio_src), file_id)
        ha.decode_audio(Path(audio_src), w24, ha.TTS_SR)
    for r in tts_rows:
        r["wav24"] = str(wav24_path(Path(r["audio_src"]), r["file_id"]))

    _export_audiofolder(asr_rows, "asr", ha.ASR_SR, "wav16", args)
    if tts_rows:
        _export_audiofolder(tts_rows, "tts", ha.TTS_SR, "wav24", args)

    if args.push:
        log.info(f"export: pushing parquet shards to {args.repo}")
        _export_parquet(asr_rows, "asr", ha.ASR_SR, "wav16", args)
        if tts_rows:
            _export_parquet(tts_rows, "tts", ha.TTS_SR, "wav24", args)
        _push_card(args)


def _push_card(args) -> None:
    from huggingface_hub import DatasetCard
    card_path = Path(__file__).parent / "docs" / "halo_livestream_card.md"
    if card_path.exists():
        DatasetCard(card_path.read_text(encoding="utf-8")).push_to_hub(
            args.repo, token=os.environ["HF_TOKEN"])
        log.info("  dataset card pushed")


# ------------------------------------------------------------------------ main

STAGES = {"parse": stage_parse, "align": stage_align, "qc": stage_qc, "export": stage_export}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--stages", default="parse,align,qc,export",
                    help="comma-separated subset of: parse,align,qc,export")
    ap.add_argument("--file-id", default=None, help="process only this source id")
    ap.add_argument("--limit", type=int, default=None, help="max source files")
    ap.add_argument("--force", default=None,
                    help="re-run this stage even if marked done (clears its stamp)")
    ap.add_argument("--num-proc", type=int, default=NUM_WORKERS)
    ap.add_argument("--whisper-model", default="large-v3",
                    help="faster-whisper model for QC (large-v3 | medium | small)")
    ap.add_argument("--push", action="store_true", help="stream parquet shards to the Hub")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    args = ap.parse_args()

    setup_logging()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if s not in STAGES:
            ap.error(f"unknown stage: {s}")

    pairs = ls.find_pairs(LIVESTREAM_DIR)
    if args.file_id:
        pairs = [p for p in pairs if p[0] == args.file_id]
    if args.limit:
        pairs = pairs[:args.limit]
    log.info(f"Sources: {len(pairs)} | stages: {stages} | workers: {args.num_proc}")

    if args.force:
        for file_id, _, _ in pairs:
            hm.clear_stage(hm.status_path(MANIFEST_DIR, file_id), args.force)

    for s in stages:
        log.info(f"=== stage: {s} ===")
        STAGES[s](pairs, args)

    log.info("Done.")


if __name__ == "__main__":
    main()
