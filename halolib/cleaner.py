"""
Web-scraped text cleaner for Philippine language corpora.
Strips boilerplate, HTML, markdown noise and filters low-quality documents.
"""

import re
import html

# ---------------------------------------------------------------------------
# Boilerplate patterns — drop entire lines matching any of these
# ---------------------------------------------------------------------------

BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*\|[\s\|\-:]+\|\s*$"),
    re.compile(r"^\s*\|.*\|\s*$"),
    re.compile(r"^\s*-\s*\[.+?\]\(https?://[^\)]+\)"),
    re.compile(r"^\s*\[.+?\]\(https?://[^\)]+\)\s*$"),
    re.compile(r"^\s*https?://\S+\s*$"),
    re.compile(r"^\s*!\[.*?\]\(.*?\)\s*$"),
    re.compile(r"^\s*\[Skip to", re.IGNORECASE),
    re.compile(r"^\s*(Share|Tweet|Pin|Email|Print|Like|Follow|Subscribe|Sign [Ii]n|Log [Ii]n|Sign [Uu]p|Log [Oo]ut|Create Blog|Register)\b"),
    re.compile(r"^\s*(Home|About|Contact|Privacy|Terms|FAQ|Search|Menu|Navigation|Sidebar|Footer|Header)\s*[|\-\\]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(WordPress|Blogger|Tumblr|Blogspot)\b", re.IGNORECASE),
    re.compile(r"^\s*-\s*About WordPress\s*$", re.IGNORECASE),
    re.compile(r"^\s*-\s*Search\s*$", re.IGNORECASE),
    re.compile(r"^\s*-\s*(Documentation|Support|Feedback|Learn WordPress|Get WordPress)\s*$", re.IGNORECASE),
    re.compile(r"^\s*BREAKING NEWS\s*:?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Close this dialog\s*$", re.IGNORECASE),
    re.compile(r"^\s*Close suggestions.*$", re.IGNORECASE),
    re.compile(r"^\s*SearchSearch\s*$", re.IGNORECASE),
    re.compile(r"^\s*Download free for \d+ days\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*ratings?\s*", re.IGNORECASE),
    re.compile(r"^\s*\d+%\s*found this document useful", re.IGNORECASE),
    re.compile(r"^\s*\d+[KM]?\s*views?\s*\d*\s*pages?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(Change Language|en\s*Change Language)\s*$", re.IGNORECASE),
    re.compile(r"^\s*en\s*$"),
    re.compile(r"^\s*(Facebook|Twitter|Instagram|YouTube|TikTok|LinkedIn|Pinterest|Vimeo|VKontakte)\s*$", re.IGNORECASE),
    re.compile(r"cookie", re.IGNORECASE),
    re.compile(r"Opens in a new window", re.IGNORECASE),
    re.compile(r"^\s*ArticlePDF Available\s*$", re.IGNORECASE),
    re.compile(r"^\s*DOI\s*:", re.IGNORECASE),
    re.compile(r"^\s*Uploadedby\b", re.IGNORECASE),
    re.compile(r"^\s*AI-en", re.IGNORECASE),
    re.compile(r"^\s*PPTX?,\s*PDF", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*views?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Latest\s*$", re.IGNORECASE),
    re.compile(r"^\s*MoreShare\b", re.IGNORECASE),
    re.compile(r"^\s*Report Abuse\s*$", re.IGNORECASE),
]


def is_boilerplate_line(line: str) -> bool:
    return any(p.search(line) for p in BOILERPLATE_PATTERNS)


def clean_inline(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\(https?://[^\)]+\)", r"\1", text)
    text = re.sub(r"\[\*{0,2}([^\]\n]+)\*{0,2}\]\(", r"\1 ", text)
    text = re.sub(r"\[([^\]\n]*)\]\s*$", r"\1", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]{1,100})\](?!\()", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[\s*\d+\s*\]", "", text)
    text = re.sub(r"\(\s*\d+\s*\)", "", text)
    text = re.sub(r"\\+", " ", text)
    text = re.sub(r"\s*\|\s*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def clean_text(text: str) -> str:
    """Clean a single document string."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    cleaned = []
    for line in text.splitlines():
        if is_boilerplate_line(line):
            continue
        line = clean_inline(line).strip()
        if not line:
            cleaned.append("")
            continue
        if not re.search(r"[a-zA-Z\u00C0-\u024F]", line):
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_usable(text: str, min_words: int = 30, min_latin_ratio: float = 0.40) -> bool:
    """Return True if the document meets minimum quality thresholds."""
    if len(text.split()) < min_words:
        return False
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    return sum(1 for c in chars if ord(c) < 0x250) / len(chars) >= min_latin_ratio
