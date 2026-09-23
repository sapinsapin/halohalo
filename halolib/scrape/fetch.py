"""
Polite fetching and main-text extraction.

Politeness is not optional for a scraper that runs unattended against small
regional sites: robots.txt is honoured, one request per host per second, a
descriptive User-Agent with a contact URL, bounded response size, and no
retries on 4xx. trafilatura does the boilerplate removal — it is the
extractor FineWeb and most modern web corpora use, and it returns the title
and date alongside the text.
"""

from __future__ import annotations

import time
import urllib.robotparser as robotparser
from dataclasses import dataclass
from urllib.parse import urlsplit

import requests

USER_AGENT = ("halohalo-scraper/0.1 (+https://github.com/sapinsapin/halohalo; "
              "Philippine-language corpus research)")
MAX_BYTES = 5_000_000
TIMEOUT = 20


@dataclass
class Page:
    url: str
    text: str
    title: str = ""
    date: str | None = None
    status: int = 0


class Fetcher:
    def __init__(self, min_interval: float = 1.0, respect_robots: bool = True):
        self.min_interval = min_interval
        self.respect_robots = respect_robots
        self._last: dict[str, float] = {}
        self._robots: dict[str, robotparser.RobotFileParser | None] = {}
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    # -- politeness -----------------------------------------------------------

    def _allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parts = urlsplit(url)
        host = f"{parts.scheme}://{parts.netloc}"
        if host not in self._robots:
            rp = robotparser.RobotFileParser()
            try:
                r = self.session.get(f"{host}/robots.txt", timeout=10)
                if r.status_code == 200:
                    rp.parse(r.text.splitlines())
                    self._robots[host] = rp
                else:
                    self._robots[host] = None      # no robots -> allowed
            except requests.RequestException:
                self._robots[host] = None
        rp = self._robots[host]
        return rp is None or rp.can_fetch(USER_AGENT, url)

    def _throttle(self, url: str) -> None:
        host = urlsplit(url).netloc
        wait = self.min_interval - (time.monotonic() - self._last.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        self._last[host] = time.monotonic()

    # -- fetch + extract ------------------------------------------------------

    def fetch_html(self, url: str) -> tuple[str | None, int]:
        if not self._allowed(url):
            return None, 999                        # our code for robots-denied
        self._throttle(url)
        try:
            r = self.session.get(url, timeout=TIMEOUT, stream=True)
            if r.status_code != 200:
                return None, r.status_code
            ctype = r.headers.get("content-type", "")
            if "html" not in ctype and "xml" not in ctype and "text" not in ctype:
                return None, 415
            buf = b""
            for chunk in r.iter_content(65536):
                buf += chunk
                if len(buf) > MAX_BYTES:
                    break
            return decode_body(buf, r.encoding), 200
        except requests.RequestException:
            return None, 0

    def fetch(self, url: str) -> Page | None:
        html, status = self.fetch_html(url)
        if html is None:
            return Page(url, "", status=status)
        return extract(html, url, status)


def decode_body(buf: bytes, declared: str | None) -> str:
    """Decode a response body without trusting the declared charset.

    Servers declare nonsense ("charset=empty" took down a ten-language run
    with LookupError: unknown encoding). Try the declared encoding, then
    UTF-8, and never raise."""
    for enc in (declared, "utf-8"):
        if not enc:
            continue
        try:
            return buf.decode(enc, errors="replace")
        except LookupError:
            continue
    return buf.decode("utf-8", errors="replace")


def extract(html: str, url: str, status: int = 200) -> Page | None:
    """Main text + metadata via trafilatura; None when nothing usable."""
    import trafilatura
    doc = trafilatura.bare_extraction(html, url=url, include_comments=False,
                                      include_tables=False, favor_precision=True,
                                      with_metadata=True)
    if not doc:
        return None
    text = (doc.text if hasattr(doc, "text") else doc.get("text")) or ""
    title = (doc.title if hasattr(doc, "title") else doc.get("title")) or ""
    date = doc.date if hasattr(doc, "date") else doc.get("date")
    if not text.strip():
        return None
    return Page(url=url, text=text, title=title, date=date, status=status)
