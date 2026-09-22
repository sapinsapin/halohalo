"""
Voice discovery, seeded by the same per-language keywords as the text scrape.

YouTube is the only web source with enough Philippine-language speech to be
worth a generic search, so this is a yt-dlp `ytsearch` wrapper with three
things bolted on that a plain search lacks:

  1. LID on the title + description (HaloLID/GlotLID), so a Waray query that
     returns Tagalog vlogs is caught before anything is downloaded.
  2. Licence gating. By default only videos YouTube tags as Creative Commons
     are downloaded. Everything else is *listed* in the candidates file with
     its licence, for a person to review — discovery is free, redistribution
     is not, and the corpus cards promise per-source provenance.
  3. A provenance sidecar per download (video id, channel, title, licence,
     duration, query, LID verdict), so a source can be excised later.

Downloaded audio lands as 16 kHz mono WAV, the format halolib.audio and the
livestream stages expect. It carries no transcript: it is raw speech for the
align/qc/pseudo-label path, not a finished dataset.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from halolib.lid import Ensemble

CC_MARKERS = ("creative commons",)


@dataclass
class Candidate:
    video_id: str
    url: str
    title: str
    channel: str
    duration: float | None
    query: str
    lang: str
    lid_lang: str
    lid_score: float
    license: str | None = None
    description: str = ""
    upload_date: str | None = None
    is_cc: bool = False
    probed: bool = False


def _ydl(quiet=True, **opts):
    import yt_dlp
    base = {"quiet": quiet, "no_warnings": True, "skip_download": True,
            "extract_flat": "in_playlist", "ignoreerrors": True}
    base.update(opts)
    return yt_dlp.YoutubeDL(base)


def discover(queries: list[str], lang: str, lid: Ensemble, max_per_query: int = 15,
             min_duration: float = 60, max_duration: float = 4 * 3600) -> list[Candidate]:
    """Flat search; LID on title only at this stage (cheap, no per-video call)."""
    out: dict[str, Candidate] = {}
    with _ydl() as ydl:
        for q in queries:
            try:
                info = ydl.extract_info(f"ytsearch{max_per_query}:{q}", download=False)
            except Exception as exc:
                print(f"  ytsearch failed {q!r}: {type(exc).__name__}")
                continue
            for e in (info or {}).get("entries") or []:
                if not e or e.get("id") in out:
                    continue
                dur = e.get("duration")
                if dur and not (min_duration <= dur <= max_duration):
                    continue
                title = e.get("title") or ""
                v = lid.identify(title) if title else None
                out[e["id"]] = Candidate(
                    video_id=e["id"], url=f"https://www.youtube.com/watch?v={e['id']}",
                    title=title, channel=e.get("channel") or e.get("uploader") or "",
                    duration=dur, query=q, lang=lang,
                    lid_lang=v.lang if v else "other", lid_score=round(v.score, 3) if v else 0.0,
                )
            time.sleep(1.0)
    return list(out.values())


def probe(cands: list[Candidate], lid: Ensemble, limit: int | None = None) -> None:
    """Per-video metadata: licence, description, upload date; re-run LID on
    title + description, which is far more reliable than the title alone."""
    with _ydl(extract_flat=False) as ydl:
        for c in cands[:limit]:
            try:
                info = ydl.extract_info(c.url, download=False)
            except Exception:
                continue
            if not info:
                continue
            c.license = info.get("license")
            c.description = (info.get("description") or "")[:2000]
            c.upload_date = info.get("upload_date")
            c.duration = info.get("duration") or c.duration
            c.is_cc = bool(c.license and any(m in c.license.lower() for m in CC_MARKERS))
            text = f"{c.title}\n{c.description}".strip()
            if text:
                v = lid.identify(text)
                c.lid_lang, c.lid_score = v.lang, round(v.score, 3)
            c.probed = True
            time.sleep(1.0)


def download_audio(cands: list[Candidate], out_dir: Path, cc_only: bool = True,
                   max_n: int | None = None, sr: int = 16000) -> int:
    """bestaudio -> 16 kHz mono WAV + JSON sidecar. Skips existing files."""
    import yt_dlp
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for c in cands:
        if max_n is not None and n >= max_n:
            break
        if cc_only and not c.is_cc:
            continue
        wav = out_dir / f"{c.video_id}.wav"
        if wav.exists():
            continue
        opts = {
            "quiet": True, "no_warnings": True, "ignoreerrors": True,
            "format": "bestaudio/best",
            "outtmpl": str(out_dir / f"{c.video_id}.%(ext)s"),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
            "postprocessor_args": ["-ar", str(sr), "-ac", "1"],
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([c.url])
        except Exception as exc:
            print(f"  download failed {c.video_id}: {type(exc).__name__}")
            continue
        if wav.exists():
            side = asdict(c)
            side["downloaded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            side["sample_rate"] = sr
            (out_dir / f"{c.video_id}.json").write_text(json.dumps(side, ensure_ascii=False, indent=1))
            n += 1
    return n


def run_voice(out_dir: Path, langs, seeds: dict, lid: Ensemble, max_per_query: int = 15,
              queries_per_lang: int = 8, probe_top: int = 40, do_download: bool = False,
              cc_only: bool = True, max_downloads: int = 20) -> dict[str, dict]:
    summary = {}
    for lang in langs:
        vdir = out_dir / "voice" / lang
        vdir.mkdir(parents=True, exist_ok=True)
        qs = seeds[lang]["queries"][:queries_per_lang]
        print(f"\n[{lang}] voice: {len(qs)} queries")
        cands = discover(qs, lang, lid, max_per_query=max_per_query)
        # keep the ones whose title already looks right, probe those first
        cands.sort(key=lambda c: (c.lid_lang != lang, -c.lid_score))
        probe(cands, lid, limit=probe_top)
        keep = [c for c in cands if c.lid_lang == lang]
        (vdir / "candidates.jsonl").write_text(
            "".join(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in cands))
        n_dl = 0
        if do_download:
            n_dl = download_audio(keep, vdir / "audio", cc_only=cc_only, max_n=max_downloads)
        stats = {"found": len(cands), "lid_match": len(keep),
                 "cc": sum(c.is_cc for c in keep), "downloaded": n_dl,
                 "hours_cc": round(sum((c.duration or 0) for c in keep if c.is_cc) / 3600, 2),
                 "hours_all_match": round(sum((c.duration or 0) for c in keep) / 3600, 2)}
        summary[lang] = stats
        print(f"[{lang}] voice " + "  ".join(f"{k}={v}" for k, v in stats.items()))
    (out_dir / "voice" / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
