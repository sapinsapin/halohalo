"""
Seed keywords per language, mined from the corpora we already hold.

The seed is what makes the scrape *targeted* rather than "search the web for
Cebuano": for each language we score every token by how much more often it
appears in that language's text than in the other nine (log-odds with a
prior), keep the ones that are frequent enough to be real vocabulary, and turn
the top terms into search queries. Function words come out on top naturally —
`kag`/`sang` for Hiligaynon, `ken`/`ti` for Ilocano, `ha`/`hin` for Waray —
and those are exactly the words a search engine matches on to find pages
written *in* the language rather than *about* it.

Sources, all local or already on the Hub:
  PLD        — read + isolated prompts, 10 languages (index_corpus, local disk)
  halohalo   — web text for tgl/hil/bcl (language column, tgl -> fil)
  BantayWika — literary/reference text (language column)
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from halolib.lid import LANGS, split_sentences

WIKI_DOC = re.compile(r"wikipedia|ensiklopedya|wikitext|malayang ensiklopedya|pumunta sa nabigasyon", re.I)


def _seed_lid():
    """HaloLID if trained, else None (seeds still work, just less clean)."""
    try:
        from halolib.lid import HaloLID
        return HaloLID()
    except Exception:
        return None

WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)

# halohalo / BantayWika use ISO codes that differ from PLD's in one place.
CODE_MAP = {"tgl": "fil", "tl": "fil", "fil": "fil"}


def _code(c: str) -> str:
    c = (c or "").lower()
    return CODE_MAP.get(c, c)


def load_seed_texts(max_per_lang: int = 20000, token: str | None = None,
                    include_hub: bool = True, scrape_dir: Path | None = None,
                    ) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    """({lang: [text, ...]}, {lang: PLD vocabulary}) from PLD plus the Hub
    text corpora. The PLD vocabulary is human-written and is used to anchor
    keywords (see anchor_to_vocab)."""
    by_lang: dict[str, list[str]] = defaultdict(list)
    pld_vocab: dict[str, set[str]] = defaultdict(set)

    root = Path(os.environ.get("PLD_RAW", "/mnt/d/backup/dsp_bkp/Speech_Corpora/PLD_raw/PLD"))
    if root.exists():
        from halolib.pld import index_corpus
        entries, _ = index_corpus(root)
        seen: set[str] = set()
        for e in entries:
            if e.get("text_is_prompt"):
                continue
            t = (e.get("sentence") or "").strip()
            key = (e["language"], t.lower())
            if t:
                pld_vocab[e["language"]].update(w.lower() for w in WORD.findall(t))
            if t and key not in seen and len(by_lang[e["language"]]) < max_per_lang:
                seen.add(key)
                by_lang[e["language"]].append(t)

    if include_hub:
        from datasets import load_dataset
        # halohalo already contains BantayWika (build_halohalo.py appends it)
        for repo in ("sapinsapin/halohalo",):
            try:
                ds = load_dataset(repo, split="train", token=token)
            except Exception as exc:  # offline / gated — PLD alone still works
                print(f"  seeds: skipping {repo}: {type(exc).__name__}")
                continue
            lid = _seed_lid()
            kept = dropped = 0
            for r in ds:
                lang = _code(r.get("language"))
                text = r.get("text") or ""
                if lang not in LANGS or len(by_lang[lang]) >= max_per_lang or not text:
                    continue
                if WIKI_DOC.search(text[:3000]):      # wiki UI would become the keywords
                    dropped += 1
                    continue
                # Web pages mix languages (halo-hil is heavily English). Keep
                # only the sentences our LID agrees are in the page's language,
                # so the keywords come from the language and not its neighbours.
                if lid is not None:
                    sents = [x for x in split_sentences(text[:4000])
                             if (p := lid.predict(x)).lang == lang and p.score >= 0.5]
                    text = " ".join(sents)
                if len(text) >= 200:
                    by_lang[lang].append(text)
                    kept += 1
                else:
                    dropped += 1
            print(f"  seeds: {repo}: kept {kept} docs, dropped {dropped} (wiki chrome / off-language)")

    if scrape_dir is not None:
        # The flywheel: pages the gate accepted with both models agreeing and
        # high confidence become seed text for the next round's queries. Web
        # prose is a better source of search terms than read prompts, and
        # each round's finds shape the next round's searches.
        import glob

        import pyarrow.parquet as pq
        n0 = sum(len(v) for v in by_lang.values())
        for f in sorted(glob.glob(str(scrape_dir / "*" / "shard-*.parquet"))):
            t = pq.read_table(f, columns=["language", "text", "lid_models_agree", "lid_score"])
            for lang, text, agree, score in zip(*(t.column(c).to_pylist() for c in
                                                  ("language", "text", "lid_models_agree", "lid_score"))):
                if agree and score >= 0.9 and lang in LANGS and len(by_lang[lang]) < max_per_lang:
                    by_lang[lang].append(text[:4000])
        print(f"  seeds: scrape shards: +{sum(len(v) for v in by_lang.values()) - n0} texts")

    return dict(by_lang), dict(pld_vocab)


# Web and wiki chrome that survives cleaning, plus other languages' names as
# they appear in interlanguage link lists. Round 1 seeds for fil were
# "ensiklopedya wikipedia wikitext deutsch français" because Tagalog
# Wikipedia dumps in the seed text carried their interface with them.
CHROME = {
    "wikipedia", "wikipediang", "wikitext", "wiki", "ensiklopedya", "namespace",
    "stub", "edit", "baguhin", "tagagamit", "lathalaing", "utc", "http", "https",
    "www", "com", "html", "php", "facebook", "twitter", "youtube", "copyright",
    "deutsch", "français", "español", "italiano", "português", "english",
    "nederlands", "polski", "русский", "svenska", "sugbuanon", "cebuano",
    "ilokano", "tagalog", "winaray", "kapampangan", "pangasinan", "bikol",
}


def anchor_to_vocab(keywords: list[tuple[str, float]], vocab: set[str],
                    min_vocab: int = 1000, exclude: set[str] | None = None) -> list[tuple[str, float]]:
    """Drop chrome, and — when the language has a real PLD vocabulary — keep
    only keywords that human speakers actually wrote in PLD. That removes
    corpus-specific junk (wiki UI, boilerplate) while keeping the function
    words that make queries find in-language pages."""
    kw = [(w, s) for w, s in keywords if w not in CHROME and w not in (exclude or ())]
    if len(vocab) >= min_vocab:
        kw = [(w, s) for w, s in kw if w in vocab]
    return kw


def contrastive_keywords(texts_by_lang: dict[str, list[str]], top_n: int = 60,
                         min_docs: int = 5, prior: float = 0.5) -> dict[str, list[tuple[str, float]]]:
    """Per language: tokens ranked by log-odds vs the pooled other languages.

    Uses document frequency, not raw counts, so a single long text cannot
    dominate; `min_docs` keeps one-off spellings out."""
    df: dict[str, Counter] = {}
    ndocs: dict[str, int] = {}
    for lang, texts in texts_by_lang.items():
        c = Counter()
        for t in texts:
            c.update(set(w.lower() for w in WORD.findall(t)))
        df[lang], ndocs[lang] = c, len(texts)

    out = {}
    for lang in df:
        n_in = ndocs[lang]
        n_out = sum(n for l, n in ndocs.items() if l != lang) or 1
        scored = []
        for w, f_in in df[lang].items():
            if f_in < min_docs:
                continue
            f_out = sum(df[l].get(w, 0) for l in df if l != lang)
            # log-odds of appearing in this language's docs vs everyone else's
            lo = math.log((f_in + prior) / (n_in + 2 * prior)) - math.log((f_out + prior) / (n_out + 2 * prior))
            scored.append((w, lo * math.sqrt(f_in)))
        scored.sort(key=lambda x: -x[1])
        out[lang] = scored[:top_n]
    return out


def build_queries(keywords: list[tuple[str, float]], n_queries: int = 20,
                  terms_per_query: int = 3, seed: int = 42,
                  exclude: set[str] | None = None) -> list[str]:
    """Combine distinctive terms into queries. Deterministic given the seed so
    a re-run finds the same pages and the manifest can skip them; a different
    seed (the flywheel passes the round number) samples different
    combinations, and `exclude` keeps spent queries out."""
    import random
    rng = random.Random(seed)
    words = [w for w, _ in keywords]
    if not words:
        return []
    top = words[: max(terms_per_query * 4, 12)]
    queries: list[str] = []
    seen = set(exclude or ())
    for _ in range(n_queries * 5):
        if len(queries) >= n_queries:
            break
        q = " ".join(rng.sample(top, min(terms_per_query, len(top))))
        if q not in seen:
            seen.add(q)
            queries.append(q)
    return queries


def prepare_seeds(out_dir: Path, langs=LANGS, n_queries: int = 20, refresh: bool = False,
                  token: str | None = None, query_seed: int = 42,
                  include_scrape: bool = False, used_queries_path: Path | None = None,
                  ) -> dict[str, dict]:
    """Mine keywords, cache per language as JSON, and return
    {lang: {"keywords": [...], "queries": [...]}}.

    `include_scrape` folds accepted scrape pages into the seed text and
    `used_queries_path` records every query ever issued so a later round never
    repeats one — together these are what make repeated rounds find new
    pages instead of the same ones."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = {l: out_dir / f"{l}.json" for l in langs}
    if not refresh and all(p.exists() for p in cached.values()):
        return {l: json.loads(p.read_text()) for l, p in cached.items()}

    used: dict[str, list[str]] = {}
    if used_queries_path and used_queries_path.exists():
        used = json.loads(used_queries_path.read_text())

    scrape_dir = (out_dir.parent / "text") if include_scrape else None
    print("  seeds: mining keywords from PLD / halohalo"
          + (" / accepted scrape pages" if include_scrape else "") + " ...")
    texts, pld_vocab = load_seed_texts(token=token, scrape_dir=scrape_dir)
    kws = contrastive_keywords(texts, top_n=400)
    seeds = {}
    for lang in langs:
        # English words leak into Philippine web text (local news especially);
        # a keyword that English speakers also write finds English pages.
        eng = pld_vocab.get("eng", set()) if lang != "eng" else set()
        kw = anchor_to_vocab(kws.get(lang, []), pld_vocab.get(lang, set()), exclude=eng)[:60]
        queries = build_queries(kw, n_queries=n_queries, seed=query_seed,
                                exclude=set(used.get(lang, [])))
        seeds[lang] = {
            "keywords": [[w, round(s, 3)] for w, s in kw],
            "queries": queries,
            "n_seed_texts": len(texts.get(lang, [])),
            "query_seed": query_seed,
        }
        cached[lang].write_text(json.dumps(seeds[lang], ensure_ascii=False, indent=1))
        used.setdefault(lang, []).extend(q for q in queries if q not in used.get(lang, []))
        print(f"  seeds: {lang}: {seeds[lang]['n_seed_texts']} texts -> "
              f"{[w for w, _ in kw[:8]]}")
    if used_queries_path:
        used_queries_path.write_text(json.dumps(used, ensure_ascii=False, indent=1))
    return seeds
