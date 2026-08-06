"""
Parser for diarized livestream recordings: zipped {id}.json + {id}.mp4 pairs.

JSON schema:
  metadata.file_properties            — container/audio specs
  metadata.linguistic_profile         — primary_language ("Taglish"), content_theme
  metadata.speaker_profile.speakers[] — {speaker_id: "Speaker 1", gender, role}
  transcription[]                     — [{time_range: "MM:SS - MM:SS",
                                          dialogue: [{s, txt}, ...]}]

Only block-level time ranges exist; a block may bundle several speaker turns
with no per-turn timestamps. `interpolate_block` provides the character-count
baseline timing; the align stage upgrades it with forced alignment.
"""

import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

LANG_MAP = {
    "taglish":    "tgl-eng",
    "tagalog":    "fil",
    "filipino":   "fil",
    "english":    "eng",
    "cebuano":    "ceb",
    "ilocano":    "ilo",
    "bikol":      "bcl",
    "hiligaynon": "hil",
}

# Priority order for locating the audio track inside an extracted archive.
# Cached decodes ("_16k"/"_24k") are excluded so re-runs don't pick them up.
AUDIO_EXTS = [".mp4", ".mov", ".m4a", ".mp3", ".wav"]
CACHE_SUFFIXES = ("_16k", "_24k")

BRACKET_RE    = re.compile(r"\[[^\]]*\]")            # [laughter], [music], [inaudible]
ONLY_PUNCT_RE = re.compile(r"^[.\-–—!?,;:'\"\s]*$")

MIN_CHARS = 2


@dataclass
class Turn:
    block_idx: int
    turn_idx: int
    speaker: str          # raw label from JSON, e.g. "Speaker 1"
    text: str


@dataclass
class Block:
    block_idx: int
    start: float
    end: float
    turns: list[Turn] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class LivestreamDoc:
    file_id: str
    language: str                     # mapped code, e.g. "tgl-eng"
    speakers: dict[str, dict]         # raw label -> {gender, role}
    blocks: list[Block] = field(default_factory=list)


def clean_sentence(text: str) -> str:
    text = BRACKET_RE.sub("", text)
    return " ".join(text.split()).strip()


def parse_timestamp(ts: str) -> float:
    """'MM:SS' or 'HH:MM:SS' -> seconds."""
    parts = [float(p) for p in ts.strip().split(":")]
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    raise ValueError(f"bad timestamp: {ts!r}")


def parse_time_range(time_range: str) -> tuple[float, float]:
    start_s, end_s = time_range.split("-")
    return parse_timestamp(start_s), parse_timestamp(end_s)


def map_language(primary_language: str) -> str:
    return LANG_MAP.get(primary_language.strip().lower(), primary_language.strip().lower())


def namespace_speaker(file_id: str, speaker: str) -> str:
    """'Speaker 1' -> '{file_id}#S1' — raw labels collide across files."""
    m = re.match(r"speaker\s*(\d+)", speaker.strip(), re.IGNORECASE)
    tag = f"S{m.group(1)}" if m else re.sub(r"\s+", "_", speaker.strip())
    return f"{file_id}#{tag}"


def parse_doc(json_path: Path) -> LivestreamDoc:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    lang = map_language(meta.get("linguistic_profile", {}).get("primary_language", "unknown"))
    speakers = {
        spk["speaker_id"]: {
            "gender": spk.get("gender", "unknown").lower(),
            "role":   spk.get("role", ""),
        }
        for spk in meta.get("speaker_profile", {}).get("speakers", [])
        if "speaker_id" in spk
    }

    blocks = []
    for block_idx, raw in enumerate(data.get("transcription", [])):
        try:
            start, end = parse_time_range(raw.get("time_range", ""))
        except ValueError as e:
            print(f"  [warn] {json_path.stem}: {e}")
            continue
        if end <= start:
            continue

        turns = []
        for d in raw.get("dialogue", []):
            text = clean_sentence(d.get("txt", ""))
            if len(text) < MIN_CHARS or ONLY_PUNCT_RE.match(text):
                continue
            turns.append(Turn(block_idx, len(turns), d.get("s", "unknown"), text))

        if turns:
            blocks.append(Block(block_idx, start, end, turns))

    return LivestreamDoc(json_path.stem, lang, speakers, blocks)


def interpolate_block(block: Block) -> list[dict]:
    """Baseline per-turn timings: block range split by character-count weight.

    Single-turn blocks map exactly onto their range ("exact"); multi-turn
    blocks are approximated ("interpolated"). The align stage replaces both
    with forced-alignment boundaries where it succeeds.
    """
    if len(block.turns) == 1:
        t = block.turns[0]
        return [{"turn": t, "start": block.start, "end": block.end, "alignment": "exact"}]

    weights = [len(t.text) for t in block.turns]
    total_w = sum(weights)
    out = []
    cursor = block.start
    for t, w in zip(block.turns, weights):
        seg_end = cursor + block.duration * (w / total_w)
        out.append({"turn": t, "start": cursor, "end": seg_end, "alignment": "interpolated"})
        cursor = seg_end
    return out


def _is_cache(path: Path) -> bool:
    return path.stem.endswith(CACHE_SUFFIXES)


def find_pairs(root: Path) -> list[tuple[str, Path, Path]]:
    """Discover (file_id, json_path, audio_path) triples.

    Sources are .zip archives (extracted once into root/_extracted/{stem}) or
    loose {id}.json + {id}.<ext> pairs dropped directly under root.
    """
    pairs = []
    tmp_root = root / "_extracted"
    tmp_root.mkdir(parents=True, exist_ok=True)

    for zpath in sorted(root.glob("*.zip")):
        dest = tmp_root / zpath.stem
        if not dest.exists():
            dest.mkdir(parents=True)
            with zipfile.ZipFile(zpath) as zf:
                for name in zf.namelist():
                    base = os.path.basename(name)
                    if name.startswith("__MACOSX") or base.startswith("._") or not base:
                        continue
                    zf.extract(name, dest)

        json_files = [j for j in dest.rglob("*.json") if not j.name.endswith(".status.json")]
        if not json_files:
            print(f"  [skip] {zpath.name}: no JSON found")
            continue

        audio_path = None
        for ext in AUDIO_EXTS:
            cands = [c for c in dest.rglob(f"*{ext}") if not _is_cache(c)]
            if cands:
                audio_path = cands[0]
                break
        if audio_path is None:
            print(f"  [skip] {zpath.name}: no audio track found")
            continue

        pairs.append((json_files[0].stem, json_files[0], audio_path))

    for json_path in sorted(root.glob("*.json")):
        for ext in AUDIO_EXTS:
            audio_path = json_path.with_suffix(ext)
            if audio_path.exists():
                pairs.append((json_path.stem, json_path, audio_path))
                break

    return pairs
