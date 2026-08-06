"""
CTC forced alignment (torchaudio MMS_FA) — upgrades block-level transcript
timings to word-accurate per-turn boundaries.

MMS_FA is Meta's multilingual (1100+ languages, incl. Tagalog) wav2vec2
aligner over a lowercase-latin character set, so code-switched Taglish aligns
without language juggling.

Per block: align the concatenation of the block's turn texts against a padded
audio window around the block's human timestamp, with <star> tokens at turn
edges to absorb untranscribed audio (cross-talk, backchannels, music — common
in long livestream blocks), then split word spans back into turns. Falls back
to the parse stage's interpolated timings on any failure — segments are never
lost here; quality gating happens at export.
"""

import unicodedata
from pathlib import Path

import numpy as np

ALIGN_CFG = {
    "version":   4,
    "model":     "torchaudio.pipelines.MMS_FA",
    "star":      "turn-edges",   # <star> tokens between turns absorb unlabeled audio
    "pad":       2.0,    # window padding (s) around block bounds
    "min_score": 0.15,   # below this, retry then fall back
    "retry_margin": 0.75,  # per-turn retry window margin (s) around neighbor bounds
    "max_trim":  1.0,    # VAD snap limit per side (s)
    "vad_pad":   0.15,   # re-pad after VAD snap (s)
}

SR = 16000


def _normalize_word(word: str, char_dict: dict) -> str:
    """Reduce a word to the aligner's character set (lowercase latin + ').

    Excludes the blank token (id 0, '-' in MMS_FA) — hyphens in Taglish
    orthography ("ire-resume") would otherwise poison the CTC targets —
    and the star token, which is inserted structurally, never from text.
    """
    word = unicodedata.normalize("NFKD", word.lower())
    word = word.encode("ascii", "ignore").decode("ascii")  # strip diacritics
    return "".join(c for c in word
                   if c != "*" and char_dict.get(c, 0) != 0)


class ForcedAligner:
    def __init__(self, device: str = "cuda"):
        import torch
        import torchaudio
        import torchaudio.functional as F
        self.torch = torch
        self.F = F
        self.device = device if torch.cuda.is_available() else "cpu"
        bundle = torchaudio.pipelines.MMS_FA
        self.model = bundle.get_model(with_star=True).to(self.device).eval()
        self.char_dict = bundle.get_dict(star="*")
        self.star_id = self.char_dict["*"]

    def align_window(self, audio_16k: np.ndarray,
                     turn_words: list[list[str]]) -> list[list[dict]] | None:
        """Align each turn's whitespace-split words against the audio window.

        Returns per turn a list of {"start", "end", "score"} — one per input
        word, times in window seconds, score = mean frame probability (0-1) —
        or None on failure. Words with no alignable characters get score 0
        and timings interpolated from their aligned neighbors.
        """
        torch, F = self.torch, self.F

        # units: ("star", None) at every turn edge, ("word", (ti, wi, chars))
        units: list[tuple] = [("star", None)]
        norm_words: dict[tuple, str] = {}
        for ti, words in enumerate(turn_words):
            for wi, w in enumerate(words):
                chars = _normalize_word(w, self.char_dict)
                norm_words[(ti, wi)] = chars
                if chars:
                    units.append(("word", (ti, wi, chars)))
            units.append(("star", None))

        if not any(kind == "word" for kind, _ in units):
            return None

        try:
            with torch.inference_mode():
                waveform = torch.from_numpy(
                    audio_16k.astype(np.float32)).unsqueeze(0).to(self.device)
                emission, _ = self.model(waveform)

            tokens = []
            for kind, payload in units:
                if kind == "star":
                    tokens.append(self.star_id)
                else:
                    tokens.extend(self.char_dict[c] for c in payload[2])
            targets = torch.tensor([tokens], dtype=torch.int32, device=self.device)
            aligned, scores = F.forced_align(emission, targets, blank=0)
            aligned, scores = aligned[0], scores[0].exp()
            token_spans = F.merge_tokens(aligned, scores)

            # regroup char spans into units (stars consume 1 span each)
            ratio = waveform.size(1) / emission.size(1)
            word_spans: dict[tuple, dict] = {}
            cursor = 0
            for kind, payload in units:
                n = 1 if kind == "star" else len(payload[2])
                spans = token_spans[cursor:cursor + n]
                cursor += n
                if kind == "word" and spans:
                    ti, wi, _ = payload
                    word_spans[(ti, wi)] = {
                        "start": ratio * spans[0].start / SR,
                        "end":   ratio * spans[-1].end / SR,
                        "score": float(np.mean([s.score for s in spans])),
                    }
        except Exception as e:
            print(f"  [warn] align failed: {type(e).__name__}: {str(e)[:120]}")
            return None

        # assemble per-turn output; fill unalignable words from neighbors
        out: list[list[dict]] = []
        flat: list[dict | None] = []
        index: list[tuple] = []
        for ti, words in enumerate(turn_words):
            for wi in range(len(words)):
                flat.append(word_spans.get((ti, wi)))
                index.append((ti, wi))
        for i, w in enumerate(flat):
            if w is None:
                prev_end = next((flat[j]["end"] for j in range(i - 1, -1, -1) if flat[j]), 0.0)
                nxt = next((flat[j] for j in range(i + 1, len(flat)) if flat[j]), None)
                nxt_start = nxt["start"] if nxt else prev_end
                flat[i] = {"start": prev_end, "end": max(prev_end, nxt_start), "score": 0.0}
        pos = 0
        for ti, words in enumerate(turn_words):
            out.append(flat[pos:pos + len(words)])
            pos += len(words)
        return out

    def close(self):
        del self.model
        import gc
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


def pad_window(block_start: float, block_end: float,
               prev_end: float | None, next_start: float | None,
               pad: float, file_dur: float) -> tuple[float, float]:
    """Block bounds ± pad, clipped at midpoints to neighboring blocks so a
    mistimed human timestamp is covered without swallowing neighbor speech."""
    lo = block_start - pad
    hi = block_end + pad
    if prev_end is not None:
        lo = max(lo, (prev_end + block_start) / 2)
    if next_start is not None:
        hi = min(hi, (block_end + next_start) / 2)
    return max(0.0, lo), min(file_dur, hi)


def run_align_stage(pairs: list[tuple], manifest_dir: Path, cfg: str,
                    num_proc: int = 4) -> None:
    from . import audio as ha
    from . import manifest as hm
    from . import vad as hv

    aligner = ForcedAligner()
    vad = hv.SileroVAD()

    try:
        for file_id, _, _ in pairs:
            mpath = hm.manifest_path(manifest_dir, file_id)
            rows = hm.read_manifest(mpath)
            if not rows:
                continue

            # snapshot/restore the parse-time baseline so re-runs are idempotent
            for r in rows:
                if "start0" not in r:
                    r["start0"], r["end0"] = r["start"], r["end"]
                    r["alignment0"] = r["alignment"]
                else:
                    r["start"], r["end"] = r["start0"], r["end0"]
                    r["alignment"] = r["alignment0"]
                    r.pop("align_score", None)
                    r.pop("word_timings", None)
                r["duration"] = round(r["end"] - r["start"], 3)

            wav16 = Path(rows[0]["wav16"])
            file_dur = ha.audio_duration(wav16)
            regions = hv.cached_speech_regions(
                hm.status_path(manifest_dir, file_id), wav16, vad)

            blocks: dict[int, list[dict]] = {}
            for r in rows:
                blocks.setdefault(r["block_idx"], []).append(r)
            block_ids = sorted(blocks)

            n_forced = n_fallback = 0
            for bi, block_idx in enumerate(block_ids):
                block_rows = sorted(blocks[block_idx], key=lambda r: r["turn_idx"])
                b_start = block_rows[0]["block_start"]
                b_end = block_rows[0]["block_end"]
                prev_end = blocks[block_ids[bi - 1]][0]["block_end"] if bi > 0 else None
                next_start = (blocks[block_ids[bi + 1]][0]["block_start"]
                              if bi < len(block_ids) - 1 else None)
                w_lo, w_hi = pad_window(b_start, b_end, prev_end, next_start,
                                        ALIGN_CFG["pad"], file_dur)

                data, _ = ha.read_window(wav16, w_lo, w_hi)
                turn_words = [r["sentence"].split() for r in block_rows]
                aligned = None
                if data is not None and len(data) > 0:
                    aligned = aligner.align_window(data, turn_words)

                if aligned is None:
                    n_fallback += len(block_rows)
                    for r in block_rows:   # keep interpolated timings; add coverage
                        r["speech_ratio"] = hv.speech_ratio(r["start"], r["end"], regions)
                    continue

                # pass 1: score each turn from the block-level alignment
                results = []   # (row, t_start, t_end, score, words) in absolute time
                for r, words in zip(block_rows, aligned):
                    if not words:
                        results.append((r, None, None, 0.0, None))
                        continue
                    t_start = w_lo + words[0]["start"]
                    t_end = w_lo + words[-1]["end"]
                    scored = [w["score"] for w in words if w["score"] > 0]
                    score = round(float(np.mean(scored)), 4) if scored else 0.0
                    if t_end <= t_start:
                        results.append((r, None, None, 0.0, None))
                    else:
                        results.append((r, t_start, t_end, score, [
                            [round(w_lo + w["start"], 3), round(w_lo + w["end"], 3)]
                            for w in words
                        ]))

                # pass 2: low-score turns get a narrow-window retry between
                # their neighbors' bounds — quiet/remote speakers align far
                # better when not competing with the whole block's audio
                for i, (r, t_start, t_end, score, words) in enumerate(results):
                    if score >= ALIGN_CFG["min_score"]:
                        continue
                    margin = ALIGN_CFG["retry_margin"]
                    prev_bound = results[i - 1][2] if i > 0 and results[i - 1][2] else b_start
                    next_bound = (results[i + 1][1]
                                  if i < len(results) - 1 and results[i + 1][1] else b_end)
                    lo2 = max(w_lo, min(prev_bound, next_bound) - margin)
                    hi2 = min(w_hi, max(prev_bound, next_bound) + margin)
                    if hi2 - lo2 < 0.3:
                        continue
                    data2, _ = ha.read_window(wav16, lo2, hi2)
                    if data2 is None or len(data2) == 0:
                        continue
                    retry = aligner.align_window(data2, [r["sentence"].split()])
                    if not retry or not retry[0]:
                        continue
                    words2 = retry[0]
                    scored2 = [w["score"] for w in words2 if w["score"] > 0]
                    score2 = round(float(np.mean(scored2)), 4) if scored2 else 0.0
                    t2_start = lo2 + words2[0]["start"]
                    t2_end = lo2 + words2[-1]["end"]
                    if score2 > score and t2_end > t2_start:
                        results[i] = (r, t2_start, t2_end, score2, [
                            [round(lo2 + w["start"], 3), round(lo2 + w["end"], 3)]
                            for w in words2
                        ])

                # pass 3: commit. Above threshold → "forced"; below → keep the
                # aligner's (better-than-proportional) timings when available
                # but stay honestly labeled "interpolated", score recorded.
                for r, t_start, t_end, score, words in results:
                    if t_start is None:
                        n_fallback += 1
                        r["speech_ratio"] = hv.speech_ratio(r["start"], r["end"], regions)
                        continue
                    t_start, t_end = hv.trim_to_speech(
                        t_start, t_end, regions,
                        max_trim=ALIGN_CFG["max_trim"], pad=ALIGN_CFG["vad_pad"])
                    r["start"] = round(t_start, 3)
                    r["end"] = round(t_end, 3)
                    r["duration"] = round(t_end - t_start, 3)
                    r["align_score"] = score
                    r["speech_ratio"] = hv.speech_ratio(t_start, t_end, regions)
                    r["word_timings"] = words
                    if score >= ALIGN_CFG["min_score"]:
                        r["alignment"] = "forced"
                        n_forced += 1
                    else:
                        r["alignment"] = "interpolated"
                        n_fallback += 1

            hm.write_manifest(mpath, rows)
            hm.mark_stage(hm.status_path(manifest_dir, file_id), "align", cfg,
                          forced=n_forced, fallback=n_fallback)
            total = n_forced + n_fallback
            pct = 100 * n_forced / total if total else 0
            print(f"  {file_id}: {n_forced}/{total} forced ({pct:.0f}%)")
    finally:
        aligner.close()
