"""
Quality control for speech segments: ASR round-trip scoring, audio metrics,
overlap heuristics, and the export gates.

Heavy dependencies (faster-whisper, pyloudnorm) are imported lazily so the
gates and normalization helpers stay importable in a CPU-only environment.
"""

import re
import unicodedata
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------- gates
#
# Starting points, tuned in Phase 3 verification. A row passes a gate when
# every present metric satisfies its bound; metrics that are None (stage not
# run yet) are skipped for ASR but *required* by flags in the TTS gate.

ASR_GATE = {
    "min_dur": 0.3,
    "max_dur": 30.0,
    "max_cer": 0.50,
    "min_speech_ratio": 0.50,
    "max_clip_ratio": 0.01,
    "min_words": 1,
    "allow_overlap": True,
    "allow_interpolated": True,
    "min_align_score": 0.15,     # applied only to forced-aligned rows
    "require_metrics": False,    # rows without qc columns still pass
}

TTS_GATE = {
    "min_dur": 1.0,
    "max_dur": 15.0,
    "max_cer": 0.15,
    "min_speech_ratio": 0.85,
    "max_clip_ratio": 0.001,
    "min_words": 3,
    "allow_overlap": False,
    "allow_interpolated": False,  # forced alignment only
    "min_align_score": 0.50,
    "min_lufs": -30.0,
    "max_lufs": -10.0,
    "require_metrics": True,      # unscored rows never reach the TTS set
}


def passes_gate(row: dict, gate: dict) -> bool:
    dur = row.get("duration") or 0.0
    if not (gate["min_dur"] <= dur <= gate["max_dur"]):
        return False
    if len((row.get("sentence") or "").split()) < gate["min_words"]:
        return False

    alignment = row.get("alignment")
    if not gate["allow_interpolated"] and alignment != "forced":
        return False
    score = row.get("align_score")
    if alignment == "forced" and score is not None and score < gate["min_align_score"]:
        return False

    if not gate["allow_overlap"] and row.get("overlap"):
        return False

    checks = [
        ("asr_cer",      lambda v: v <= gate["max_cer"]),
        ("speech_ratio", lambda v: v >= gate["min_speech_ratio"]),
        ("clip_ratio",   lambda v: v <= gate["max_clip_ratio"]),
    ]
    if "min_lufs" in gate:
        checks.append(("lufs", lambda v: gate["min_lufs"] <= v <= gate["max_lufs"]))

    for key, ok in checks:
        v = row.get(key)
        if v is None:
            if gate["require_metrics"]:
                return False
            continue
        if not ok(v):
            return False
    return True


# --------------------------------------------------------- text normalization

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_for_cer(text: str) -> str:
    """Lowercase, unify apostrophes, strip punctuation, collapse whitespace.

    CER over this form is stable against Taglish orthography variants
    ("'pag" vs "pag", "ire-resume" vs "ire resume") that would wreck WER.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = _PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def cer(ref: str, hyp: str) -> float:
    import jiwer
    ref_n, hyp_n = normalize_for_cer(ref), normalize_for_cer(hyp)
    if not ref_n:
        return 1.0
    return min(1.0, jiwer.cer(ref_n, hyp_n))


# --------------------------------------------------------------- ASR scoring

def qc_config(whisper_model: str) -> dict:
    """The parameters the qc stage's config hash depends on."""
    return {"version": 1, "whisper_model": whisper_model, "gate_free": True}


class AsrScorer:
    """faster-whisper round-trip transcription for CER scoring."""

    def __init__(self, model_size: str = "large-v3",
                 compute_type: str = "int8", device: str = "cuda"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_16k: np.ndarray, language: str = "tl") -> str:
        segments, _ = self.model.transcribe(
            audio_16k.astype(np.float32),
            language=language,
            beam_size=1,
            without_timestamps=True,
            vad_filter=False,        # segments are already VAD-snapped
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    def close(self):
        del self.model
        try:
            import torch
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except ImportError:
            pass


# -------------------------------------------------------------- audio metrics

def loudness_metrics(audio: np.ndarray, sr: int) -> dict:
    out = {
        "peak":       float(np.max(np.abs(audio))) if len(audio) else 0.0,
        "clip_ratio": float(np.mean(np.abs(audio) >= 0.999)) if len(audio) else 0.0,
        "rms":        float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0,
        "lufs":       None,
    }
    try:
        import pyloudnorm
        if len(audio) >= sr * 0.4:  # pyloudnorm needs >=400ms
            meter = pyloudnorm.Meter(sr)
            lufs = meter.integrated_loudness(audio.astype(np.float64))
            if np.isfinite(lufs):
                out["lufs"] = round(float(lufs), 2)
    except ImportError:
        pass
    return out


def snr_proxy(window_audio: np.ndarray, sr: int,
              speech_regions: list[tuple[float, float]],
              window_start: float) -> float | None:
    """10·log10(speech RMS² / nonspeech RMS²) within one block window.

    Nonspeech in a livestream is background music/notification audio, so this
    doubles as a background-noise indicator. Returns None when the window has
    no nonspeech to measure against.
    """
    if len(window_audio) == 0:
        return None
    mask = np.zeros(len(window_audio), dtype=bool)
    for s, e in speech_regions:
        lo = max(0, int((s - window_start) * sr))
        hi = min(len(window_audio), int((e - window_start) * sr))
        if hi > lo:
            mask[lo:hi] = True
    speech, nonspeech = window_audio[mask], window_audio[~mask]
    if len(speech) < sr * 0.1 or len(nonspeech) < sr * 0.1:
        return None
    sp = float(np.mean(speech ** 2))
    ns = float(np.mean(nonspeech ** 2))
    if ns <= 1e-12:
        return 60.0
    return round(10.0 * np.log10(max(sp, 1e-12) / ns), 1)


# ---------------------------------------------------------- overlap heuristic

def flag_overlaps(rows: list[dict], gap_s: float = 0.15) -> None:
    """Mark probable overlapped speech in-place. Heuristic, no extra models:

    a) multi-turn block, adjacent turn within `gap_s`, and low boundary
       alignment confidence — speakers likely talking over each other;
    b) high CER despite confident alignment — classic overlap signature.
    """
    by_block: dict[tuple, list[dict]] = {}
    for r in rows:
        by_block.setdefault((r["file_id"], r["block_idx"]), []).append(r)

    for block_rows in by_block.values():
        block_rows.sort(key=lambda r: r["turn_idx"])
        for i, r in enumerate(block_rows):
            overlap = False
            if len(block_rows) > 1:
                tight_prev = i > 0 and (r["start"] - block_rows[i - 1]["end"]) < gap_s
                tight_next = (i < len(block_rows) - 1
                              and (block_rows[i + 1]["start"] - r["end"]) < gap_s)
                low_conf = (r.get("align_score") or 0.0) < 0.30
                if (tight_prev or tight_next) and low_conf:
                    overlap = True
            asr_cer_v = r.get("asr_cer")
            score = r.get("align_score")
            if (asr_cer_v is not None and score is not None
                    and asr_cer_v > 0.5 and score >= 0.5):
                overlap = True
            r["overlap"] = overlap


# ------------------------------------------------------------------ qc stage

def run_qc_stage(pairs: list[tuple], manifest_dir: Path, cfg: str,
                 whisper_model: str = "large-v3", num_proc: int = 4) -> None:
    """Score all pending files: ASR round-trip + audio metrics + overlap flag."""
    from concurrent.futures import ProcessPoolExecutor

    from . import audio as ha
    from . import manifest as hm
    from . import vad as hv

    scorer = AsrScorer(model_size=whisper_model)
    vad = hv.SileroVAD()
    try:
        for file_id, _, _ in pairs:
            mpath = hm.manifest_path(manifest_dir, file_id)
            rows = hm.read_manifest(mpath)
            if not rows:
                continue
            wav16 = Path(rows[0]["wav16"])

            # --- per-segment ASR + metrics (sequential GPU, cheap CPU inline) ---
            regions = hv.cached_speech_regions(
                hm.status_path(manifest_dir, file_id), wav16, vad)

            for r in rows:
                data, sr = ha.read_window(wav16, r["start"], r["end"])
                if data is None or len(data) == 0:
                    r["asr_cer"] = 1.0
                    continue
                hyp = scorer.transcribe(data)
                r["asr_text"] = hyp
                r["asr_cer"] = round(cer(r["sentence"], hyp), 4)
                m = loudness_metrics(data, sr)
                r["lufs"] = m["lufs"]
                r["clip_ratio"] = round(m["clip_ratio"], 5)

            # --- block-window SNR proxy ---
            blocks: dict[int, list[dict]] = {}
            for r in rows:
                blocks.setdefault(r["block_idx"], []).append(r)
            for block_rows in blocks.values():
                w_start = block_rows[0]["block_start"]
                w_end = block_rows[0]["block_end"]
                wdata, wsr = ha.read_window(wav16, w_start, w_end)
                snr = None
                if wdata is not None and len(wdata):
                    snr = snr_proxy(wdata, wsr, regions, w_start)
                for r in block_rows:
                    r["snr_db"] = snr

            flag_overlaps(rows)
            hm.write_manifest(mpath, rows)
            hm.mark_stage(hm.status_path(manifest_dir, file_id), "qc", cfg,
                          rows=len(rows))
            n_ov = sum(1 for r in rows if r.get("overlap"))
            cers = [r["asr_cer"] for r in rows if r.get("asr_cer") is not None]
            med = sorted(cers)[len(cers) // 2] if cers else None
            print(f"  {file_id}: qc done — median CER {med}, {n_ov} overlap-flagged")
    finally:
        scorer.close()
