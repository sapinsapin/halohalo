"""
Parser for the UP-DSP Philippine Language Dataset (PLD).

Layout on disk:
  PLD/{LANG}/{SPEAKER}/{speaker}.{session}.log
  PLD/{LANG}/{SPEAKER}/{speaker}.{session}.{NNNN}.wav

Unlike the Filipino Speech Corpus, PLD ships pre-segmented per-utterance WAVs
with the prompt text recorded inline, so there is no TRS/forced-alignment step:
one .log row == one training example.

Each .log is a session file: a `Key = Value` header describing the speaker and
recording conditions, followed by utterance rows of the form

    0800.120717.105610.0001.wav "BCL_Iso_BodyParts.txt" "ngipin "

Parsing quirks handled here:
  * UTF-8 BOM on the header (read with utf-8-sig)
  * "SpekaerDialect" — misspelled in the corpus itself
  * NOT_RECORDED in the filename column for prompts the speaker skipped
  * transcripts containing embedded double quotes, so the text field is matched
    greedily to the final quote rather than to the first
  * prompt sources that are not filenames ("Random Digit")
"""

import re
import unicodedata
from pathlib import Path

# `<file> "<prompt source>" "<transcript>"` — transcript is greedy so that
# embedded quotes stay part of the text instead of truncating it.
_ROW_RE = re.compile(r'^(\S+)\s+"([^"]*)"\s+"(.*)"\s*$')
_HEADER_RE = re.compile(r'^(\w+)\s*=\s*(.*?)\s*$')

# Directory name → BCP-47/ISO 639-3. PLD directories use the collector's own
# 3-letter codes, which do not always match ISO (e.g. BIK vs bcl).
LANG_CODES = {
    "BIK": "bcl",   # Bikol Central
    "CEB": "ceb",   # Cebuano
    "HIL": "hil",   # Hiligaynon
    "ILO": "ilo",   # Ilocano
    "TAG": "tgl",   # Tagalog
    "PAM": "pam",   # Kapampangan
    "PAN": "pag",   # Pangasinan
    "WAR": "war",   # Waray
    "MRW": "mrw",   # Maranao
    "TAU": "tsg",   # Tausug
    "MDH": "mdh",   # Maguindanao
    "IVA": "ivv",   # Ivatan
    "AKL": "akl",   # Aklanon
    "BTK": "btk",   # Batak
    "IVT": "ivv",
    "KIN": "kin",
    "SBL": "sbl",   # Sambal
    "SUR": "sgd",   # Surigaonon
    "YAK": "yka",   # Yakan
    "CHA": "cbk",   # Chavacano
    "MSK": "msk",
    "MBB": "mbb",
}

LANG_NAMES = {
    "BIK": "Bikol", "CEB": "Cebuano", "HIL": "Hiligaynon", "ILO": "Ilocano",
    "TAG": "Tagalog", "PAM": "Kapampangan", "PAN": "Pangasinan", "WAR": "Waray",
    "MRW": "Maranao", "TAU": "Tausug", "MDH": "Maguindanao", "IVA": "Ivatan",
    "AKL": "Aklanon", "SBL": "Sambal", "SUR": "Surigaonon", "YAK": "Yakan",
    "CHA": "Chavacano",
}


def speech_type_of(prompt_source: str) -> str:
    """Bucket a prompt source into a speech type.

    The corpus encodes elicitation style in the prompt filename:
      *_Iso_*        isolated words / short phrases (word-list reads)
      *_Utt_*        full read sentences — the TTS-useful material
      *Spontaneous*  free speech, typically 30-60s monologues
      "Random Digit" spoken digit strings
    """
    s = prompt_source.lower()
    if "spontaneous" in s:
        return "spontaneous"
    if "digit" in s:
        return "digits"
    if "_iso_" in s or s.startswith("engw"):
        return "isolated"
    if "_utt_" in s or s.startswith("engsen"):
        return "read"
    return "other"


def is_english_prompt(prompt_source: str) -> bool:
    """EngW.txt / EngSen.txt are English word and sentence lists read by the
    same speakers — accented English, not the directory's Philippine language.
    Mislabeling these would poison per-language training filters."""
    return Path(prompt_source).stem.lower().startswith("eng")


def prompt_category_of(prompt_source: str) -> str:
    """`BCL_Utt_News.txt` → `News`; `BCL_Spontaneous01.txt` → `Spontaneous01`."""
    stem = Path(prompt_source).stem if prompt_source.endswith(".txt") else prompt_source
    parts = stem.split("_")
    if parts and len(parts[0]) == 3 and parts[0].isupper():
        parts = parts[1:]                       # drop the language prefix
    if parts and parts[0] in ("Iso", "Utt"):
        parts = parts[1:]                       # drop the elicitation marker
    return "_".join(parts) if parts else stem.replace(" ", "")


def clean_transcript(text: str) -> str:
    """Normalize spacing and unicode punctuation; keep orthography intact.

    Transcripts are left otherwise untouched — these are the prompts the
    speaker actually read, and normalizing spelling would destroy the
    dialectal variation the corpus exists to capture.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return " ".join(text.split()).strip()


def parse_log(log_path: Path) -> tuple[dict, list[dict]]:
    """Return (session_meta, utterance_rows) for one .log file.

    Rows are returned for prompts that were actually recorded; NOT_RECORDED
    placeholders and unparseable lines are dropped.
    """
    meta: dict[str, str] = {}
    rows: list[dict] = []

    with log_path.open(encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue

            row = _ROW_RE.match(line.strip())
            if row:
                filename, prompt_source, transcript = row.groups()
                if filename.upper().startswith("NOT_RECORDED"):
                    continue
                text = clean_transcript(transcript)
                if not text:
                    continue
                stype = speech_type_of(prompt_source)
                rows.append({
                    "filename": filename,
                    "prompt_source": prompt_source,
                    "sentence": text,
                    "num_words": len(text.split()),
                    "speech_type": stype,
                    "prompt_category": prompt_category_of(prompt_source),
                    # Spontaneous sessions log the *elicitation question*, not a
                    # transcript of the answer — the same text repeats verbatim
                    # across speakers while the audio is 20-90s of free speech.
                    # Such rows are not usable (audio, text) supervision.
                    "text_is_prompt": stype == "spontaneous",
                })
                continue

            header = _HEADER_RE.match(line.strip())
            if header and "=" in line:
                key, value = header.groups()
                meta[key] = value.strip().strip('"').strip()

    return meta, rows


def speaker_meta(meta: dict, lang_dir: str, log_path: Path) -> dict:
    """Normalize session-header fields into dataset columns."""
    speaker = meta.get("SpeakerID") or log_path.stem.split(".")[0]

    age = meta.get("SpeakerAge", "")
    try:
        age_val = int(str(age).strip().strip('"'))
    except (TypeError, ValueError):
        age_val = -1

    gender = (meta.get("SpeakerGender") or "unknown").strip().lower()
    if gender not in ("male", "female"):
        gender = "unknown"

    return {
        # speaker ids repeat across languages, so namespace them
        "speaker_id":     f"{lang_dir}_{speaker}",
        "gender":         gender,
        "age":            age_val,
        # the corpus itself misspells this key
        "speaker_dialect": meta.get("SpeakerDialect") or meta.get("SpekaerDialect") or "",
        "mother_dialect": meta.get("MotherDialect", ""),
        "father_dialect": meta.get("FatherDialect", ""),
        "profession":     meta.get("SpeakerProfession", ""),
        "session_id":     meta.get("SessionID") or ".".join(log_path.stem.split(".")[1:]),
        "session_environment": meta.get("SessionEnvironment", ""),
    }


def iter_sessions(pld_dir: Path, languages: list[str] | None = None):
    """Yield (lang_dir, log_path) for every session log under PLD/."""
    for lang_path in sorted(p for p in pld_dir.iterdir() if p.is_dir()):
        lang_dir = lang_path.name
        if languages and lang_dir not in languages:
            continue
        for log_path in sorted(lang_path.rglob("*.log")):
            yield lang_dir, log_path


def index_corpus(pld_dir: Path, languages: list[str] | None = None) -> tuple[list[dict], dict]:
    """Walk the corpus and build one entry per recorded utterance.

    Audio is not opened here — entries carry the wav path so callers can decide
    whether to read durations (stats) or bytes (sharding).
    """
    entries: list[dict] = []
    counts = {"sessions": 0, "rows": 0, "missing_wav": 0, "empty_sessions": 0}

    for lang_dir, log_path in iter_sessions(pld_dir, languages):
        meta, rows = parse_log(log_path)
        counts["sessions"] += 1
        if not rows:
            counts["empty_sessions"] += 1
            continue

        spk = speaker_meta(meta, lang_dir, log_path)
        session_dir = log_path.parent

        for row in rows:
            wav_path = session_dir / row["filename"]
            if not wav_path.exists():
                counts["missing_wav"] += 1
                continue
            counts["rows"] += 1
            english = is_english_prompt(row["prompt_source"])
            entries.append({
                "wav_path":  wav_path,
                # language of the spoken content; the English word/sentence
                # lists are read by the same speakers but are not the
                # directory's language
                "language":  "eng" if english else LANG_CODES.get(lang_dir, lang_dir.lower()),
                "language_name": "English" if english else LANG_NAMES.get(lang_dir, lang_dir),
                # which PLD language collection the session belongs to
                "corpus_language": LANG_CODES.get(lang_dir, lang_dir.lower()),
                "lang_dir":  lang_dir,
                "source_file": wav_path.stem,
                **{k: v for k, v in row.items() if k != "filename"},
                **spk,
            })

    return entries, counts
