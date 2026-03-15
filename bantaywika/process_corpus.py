"""
Process Philippine text corpora into FineWeb-compatible JSONL.

FineWeb schema per record:
  text        : str  — document text
  id          : str  — unique identifier
  source      : str  — corpus source label
  language    : str  — ISO 639-3 code
  token_count : int  — whitespace word count
"""

import hashlib
import html
import json
import os
import random
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

CORPUS_ROOT = Path(os.environ["CORPUS_TEXT_DIR"])
OUTPUT_DIR  = Path(os.environ["CORPUS_OUTPUT_DIR"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TEST_RATIO  = 0.1
MIN_WORDS   = 50

# ---------------------------------------------------------------------------
# Sources: (directory, glob, source_label, language)
# Always use the most-processed .tok.cleaned variant where available.
# FilNet = "100 nobelang tagalog" folder (not FilNet/files which is empty)
# ---------------------------------------------------------------------------
SOURCES = [
    # Filipiniana classic literary texts
    (CORPUS_ROOT / "Filipiniana",
     "*.cleaned", "filipiniana", "fil"),

    # Project Gutenberg — fully processed chain
    # Exclude decade-aggregated files (1900_1909, 1910_1919) — they duplicate individual titles
    (CORPUS_ROOT / "Proj_Gutenberg",
     "*.cleaned.noBlank.noHex.convHTML.tok.cleaned", "proj_gutenberg", "fil"),

    # Newspaper corpus — aggregated tok.cleaned only (individual .txt files are raw duplicates)
    (CORPUS_ROOT / "Newspaper Corpus" / "Filipino Texts",
     "Fil_News_ALL.txt.noBlank.noHex.convHTML.tok.cleaned", "newspaper_fil", "fil"),
    (CORPUS_ROOT / "Newspaper Corpus" / "Filipino Texts",
     "DLSU_News_Tagalog.txt.clean", "newspaper_fil", "fil"),

    # Palito literary + religious
    (CORPUS_ROOT / "Palito",
     "*.noBlank.noHex.convHTML.tok.cleaned", "palito", "fil"),

    # 100 Nobelang Tagalog (FilNet)
    (CORPUS_ROOT / "FilNet" / "100 nobelang tagalog",
     "*.odt.txt.noBlank.noHex.convHTML.tok.cleaned", "filnet_novels", "fil"),

    # Transcribed novels — batches 1-4 (batch_1 and batch_2 are identical, dedup handles it)
    (CORPUS_ROOT / "Transcribed" / "batch_1",
     "*.txt.noBlank.noHex.convHTML.tok.cleaned", "transcribed", "fil"),
    (CORPUS_ROOT / "Transcribed" / "batch_2",
     "*.txt.noBlank.noHex.convHTML.tok.cleaned", "transcribed", "fil"),
    (CORPUS_ROOT / "Transcribed" / "batch_3",
     "*.txt.noBlank.noHex.convHTML.tok.cleaned", "transcribed", "fil"),
    (CORPUS_ROOT / "Transcribed" / "batch_4",
     "*.txt.noBlank.noHex.convHTML.tok.cleaned", "transcribed", "fil"),

    # ISIP — Cebuano and Ilocano news (raw, no cleaned versions exist)
    (CORPUS_ROOT / "ISIP", "CEB_News.txt",     "isip_ceb", "ceb"),
    (CORPUS_ROOT / "ISIP", "CEB_Sun_News.txt", "isip_ceb", "ceb"),
    (CORPUS_ROOT / "ISIP", "ILK_News.txt",     "isip_ilk", "ilo"),
    # CEB_News_with_TGL_and_ENG_translations — excluded (trilingual, mixed)
]


# ---------------------------------------------------------------------------
# Cleaning preprocessor
# Applied uniformly to ALL sources regardless of prior processing state.
# ---------------------------------------------------------------------------

def clean(text: str, lang: str = "fil") -> str:
    # 1. Decode HTML entities (&amp; &lt; etc.)
    text = html.unescape(text)

    # 2. Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Strip control characters (keep \n \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 4. Fix tokenizer spacing artifacts from the old pipeline:
    #    n~g → ng,  n~ga → nga,  i~y → iy  etc.
    text = re.sub(r"(\w)~(\w)", r"\1\2", text)

    # 5. Rejoin split contractions/clitics produced by the tokenizer:
    #    "' t" → "'t",  "' y" → "'y",  "' ng" → "'ng",  "' ko" etc.
    text = re.sub(r"'\s+([a-zA-Z])", r"'\1", text)

    # 6. Remove footnote/reference markers: [ 2 ], [3], ( 4 )
    text = re.sub(r"\[\s*\d+\s*\]", "", text)
    text = re.sub(r"\(\s*\d+\s*\)", "", text)

    # 7. Remove metadata/header lines:
    #    "Title : ...", "= heading =", "* * *", "....", "---"
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip metadata headers
        if re.match(r"^Title\s*:", stripped, re.IGNORECASE):
            continue
        # Skip section markers: = ... =, * * *, ----, ....
        if re.match(r"^[=\-\*\.]{2,}\s*.*\s*[=\-\*\.]{0,}$", stripped):
            continue
        # Skip lines that are purely punctuation/symbols
        if stripped and not re.search(r"[a-zA-Z\u00C0-\u024F]", stripped):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # 8. For Cebuano ISIP files: strip parenthetical Tagalog translations.
    #    Pattern: a line that is entirely "(Tagalog translation here)"
    if lang == "ceb":
        text = re.sub(r"^\s*\(.*?\)\s*$", "", text, flags=re.MULTILINE)

    # 9. Normalize whitespace within lines (collapse multiple spaces/tabs)
    text = re.sub(r"[ \t]+", " ", text)

    # 10. Strip leading/trailing whitespace per line
    text = "\n".join(line.strip() for line in text.splitlines())

    # 11. Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def word_count(text: str) -> int:
    return len(text.split())


def is_mostly_latin(text: str) -> bool:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    latin = sum(1 for c in chars if ord(c) < 0x250)
    return (latin / len(chars)) >= 0.40


def make_id(source: str, fname: str, idx: int) -> str:
    h = hashlib.md5(f"{source}:{fname}:{idx}".encode()).hexdigest()[:8]
    return f"{source}_{idx:06d}_{h}"


def split_into_documents(text: str) -> list[str]:
    """
    Split a file into logical documents.
    - Double-newline separated blocks → paragraph mode
    - Single-newline sentence-per-line (no blank lines) → line mode
    - Literary files with large paragraphs → one document per file
    News/sentence-per-line files are chunked into ~200-word documents.
    """
    # Try paragraph split first
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        return []

    # If only one "paragraph" (no blank lines), treat each line as a unit
    if len(paragraphs) == 1:
        paragraphs = [l.strip() for l in text.splitlines() if l.strip()]

    avg_words = sum(word_count(p) for p in paragraphs) / len(paragraphs)

    # Chunk if units are short (news/sentence-per-line style)
    if avg_words < 80 and len(paragraphs) > 5:
        docs, buf, buf_words = [], [], 0
        for para in paragraphs:
            pw = word_count(para)
            buf.append(para)
            buf_words += pw
            if buf_words >= 200:
                docs.append("\n".join(buf))
                buf, buf_words = [], 0
        if buf:
            docs.append("\n".join(buf))
        return docs
    else:
        return [text]


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_records() -> list[dict]:
    seen: set[str] = set()
    records: list[dict] = []

    # Decade-aggregated Proj_Gutenberg files that duplicate individual titles
    EXCLUDE_STEMS = {"1900_1909.txt", "1910_1919.txt"}

    for base_dir, pattern, source, lang in SOURCES:
        base_dir = Path(base_dir)
        if not base_dir.exists():
            print(f"  [skip] {base_dir} not found")
            continue

        files = [
            f for f in sorted(base_dir.glob(pattern))
            if f.is_file()
            # Strip all extensions to get the stem for exclusion check
            and not any(f.name.startswith(s) for s in EXCLUDE_STEMS)
        ]

        doc_idx = 0
        for fpath in files:
            try:
                raw = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"  [warn] {fpath.name}: {e}")
                continue

            text = clean(raw, lang)
            docs = split_into_documents(text)

            for doc in docs:
                wc = word_count(doc)
                if wc < MIN_WORDS:
                    continue
                if not is_mostly_latin(doc):
                    continue

                h = hashlib.md5(doc.encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)

                records.append({
                    "text": doc,
                    "id": make_id(source, fpath.name, doc_idx),
                    "source": source,
                    "language": lang,
                    "token_count": wc,
                })
                doc_idx += 1

        print(f"  {source:20s} ({lang})  {doc_idx:>5} docs  [{len(files)} files]")

    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    print(f"Corpus root : {CORPUS_ROOT}")
    print(f"Output dir  : {OUTPUT_DIR}\n")

    records = load_records()
    print(f"\nTotal after filtering + dedup: {len(records)}")

    random.seed(RANDOM_SEED)
    random.shuffle(records)

    split = int(len(records) * (1 - TEST_RATIO))
    train, test = records[:split], records[split:]

    write_jsonl(train, OUTPUT_DIR / "train.jsonl")
    write_jsonl(test,  OUTPUT_DIR / "test.jsonl")

    print(f"Train: {len(train)}  →  {OUTPUT_DIR / 'train.jsonl'}")
    print(f"Test : {len(test)}   →  {OUTPUT_DIR / 'test.jsonl'}")


if __name__ == "__main__":
    main()
