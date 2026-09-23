# Seeded web scrape — text and voice for the ten PLD languages

`scrape_web.py` turns the corpora we already hold into more corpus: distinctive
words mined from PLD, halohalo and BantayWika become search queries, the pages
they find are gated by language identification, cleaned, deduplicated and
written in the FineWeb-compatible schema `sapinsapin/halohalo` uses — with
enough provenance per row that any source can be audited or excised later.
The same seeds drive YouTube discovery for speech.

```bash
python scrape_web.py                                  # all ten languages, Tavily
python scrape_web.py --langs ceb,war --max-docs 200
python scrape_web.py --backend fineweb2 --max-docs 2000    # retained corpus source
python scrape_web.py --backend direct --urls my_urls.txt --langs ilo
python scrape_web.py --voice --langs ilo --download-audio  # CC-licensed audio only
python scrape_web.py --push                           # append to sapinsapin/halo-{lang}
python scripts/train_lid.py --push                    # retrain the LID, publish
```

```
# .env
TAVILY_API_KEY=...        # default search backend (https://app.tavily.com)
SCRAPE_DIR=/mnt/d/halohalo/finetune_runs/scrape   # optional; default shown
```

---

## Why seeded, why gated

A search engine asked for "Cebuano news" returns pages *about* Cebuano, mostly
in English. Asked for `karon adlawa gobyerno siyudad` it returns pages *in*
Cebuano, because those are words only Cebuano pages contain. The seed stage
(`halolib/scrape/seeds.py`) computes exactly that: for each language, tokens
ranked by log-odds of appearing in its documents versus the other nine, from
the text we already have. Function words dominate the top of every list —
`kag`/`sang` (hil), `ken`/`ti` (ilo), `ha`/`hin` (war) — which is what you
want, since they are frequent, language-specific, and topic-neutral.

Search results still contain the wrong language, machine translation,
bot-written encyclopedias and boilerplate. The gate is language
identification at document level (`halolib/lid.py`): sentences vote weighted
by length, so a Waray page with an English navigation bar is Waray and an
English page with one Waray quotation is not. Two models vote — our own
HaloLID and GlotLID — and the row records both verdicts, whether they agreed,
and the sentence-level agreement, so the gate's decisions can be checked
rather than trusted.

## Stages

```
seeds ──► search ──► fetch/extract ──► clean ──► LID gate ──► dedup ──► shards
 PLD      tavily      trafilatura     halolib   HaloLID+     exact +    parquet
 halohalo direct      (or backend      cleaner  GlotLID      MinHash    + manifest
 BantayW. fineweb2     raw text)                doc vote     LSH 0.75
```

| Stage | Module | What it does |
|---|---|---|
| seeds | `scrape/seeds.py` | Mines contrastive keywords per language; caches `seeds/{lang}.json`; builds deterministic queries |
| search | `scrape/search.py` | Pluggable backends yielding `Hit`s; Tavily returns page text with the result so most hits never need a fetch |
| fetch | `scrape/fetch.py` | robots.txt, 1 req/s/host, descriptive UA, 5 MB cap; trafilatura main-text + title + date |
| clean | `halolib/cleaner.py` | The same boilerplate stripper and usability filter as `clean_halo.py` (≥30 words, Latin ratio) |
| LID gate | `halolib/lid.py` | Accept iff decider says the target language with score ≥ 0.6 and sentence agreement ≥ 0.6; disputed calls are penalised, not dropped |
| dedup | `scrape/dedup.py` | md5 `content_hash` (same field as the Hub sets) + MinHash LSH near-dups; seeded from existing shards on resume |
| shards | `scrape/pipeline.py` | `text/{lang}/shard-NNNNN.parquet` + `manifest.jsonl` (every URL, outcome, reason) |

Every stage is resumable: URLs in the manifest are skipped, shards on disk
seed the dedup index, seeds are cached. Kill it and re-run.

## Backends

**Tavily (default).** `TavilyBackend` calls `search(query, search_depth=
"advanced", include_raw_content=True, exclude_domains=…)`. Needs
`TAVILY_API_KEY`; without it the driver exits with the instruction, it does
not silently fall back. `tests/test_scrape.py` verifies the adapter against a
stub client so the default path is covered without a key or network.

Two things the first live runs (2026-09-23) taught, both now built in:

- **Social and video domains are excluded.** The very first Hiligaynon query
  returned three Facebook posts with zero extractable text — each one a
  wasted credit and a wasted fetch. Facebook, Instagram, TikTok, X, YouTube,
  Pinterest and Threads are excluded by default.
- **`raw_content` is a whole-page dump, not an article.** Every accepted
  document began with the site's menu (`* HOME * BOMBO TUGUEGARAO * BOMBO
  LAOAG …`). Hits are therefore flagged `page_dump`, and the pipeline fetches
  the live page and extracts the main text with trafilatura, using the dump
  only when the fetch yields nothing. In the Hiligaynon pilot 11 of 17 pages
  came through the fetch path; the fallback is cleaned by the two menu rules
  described under *Stages*.

Cost: an advanced search is 2 credits, so 20 queries per language is ~40
credits, ~400 for all ten — inside Tavily's free tier. Wall-clock is set by
the polite fetcher (one request per second per host), not by the API.

### First full Tavily run (2026-09-23), 20 queries × 10 hits per language

After the gate, dedup, the 20k-word cap, and the host exclusions:

| lang | accepted | words | LID rejects | fetch fails | note |
|---|---|---|---|---|---|
| bcl | 108 | 108,716 | 32 | 16 | 41 % jw.org |
| ceb | 69 | 128,757 | 59 | 39 | a 477k-word Bible capped out |
| eng | 110 | 214,876 | 0 | 82 | |
| fil | 40 | 60,270 | 68 | 47 | 55 of the rejects were English pages |
| hil | 94 | 45,723 | 43 | 40 | Bombo Radyo, Digicast Negros |
| ilo | 70 | 149,940 | 21 | 34 | |
| pag | 98 | 87,551 | 25 | 16 | 46 % jw.org |
| pam | 39 | 56,868 | 69 | 36 | |
| tsg | **9** | 9,916 | 78 | 46 | 70 of the rejects were English pages *about* Tausug |
| war | 49 | 80,269 | 51 | 42 | 13 bot-Wikipedia pages purged |
| **total** | **686** | **942,886** | | | |

Together with the FineWeb-2 pass the scrape now holds **21,936 documents /
11.4 M words** across the ten languages.

What the numbers say:

- **The gate is not over-filtering code-switched text.** Where rejections
  are high (fil, tsg), the rejected pages were identified as *English*
  outright — search results for a small language are dominated by pages
  about it in English. Only a handful of rejections were target-language
  pages under the agreement threshold (fil 6, tsg 3, all at 0.56–0.58).
- **Tausug is search-limited, not gate-limited.** Nine pages from 200 hits.
  More queries won't fix that; the fix is a curated `direct` list of the few
  sites that publish in Tausug, and the community drive.
- **Scripture dominates the small languages** (jw.org, bible.com, ebible.org,
  churchofjesuschrist.org are top hosts for bcl, pag, war, tsg). It is real
  text in the language, and it is a narrow register; the CPT mix has to
  account for it, which is why `top_hosts` is in every summary.
- **Machine-translation farms hide in the URL path, not just the subdomain.**
  `alltechbuzz.net/ceb/…`, `qc-solar.com/pag/…`, `pilotech.ai/pam/…`: about
  450 accepted hosts had a language-code first path segment, and nearly all
  were manufacturers, casino sites and Spanish tech blogs "in Cebuano".
  `mt_path_excluded` now applies the subdomain rule one slash later, with a
  named allowlist for human-translation publishers (jw.org, ebible.org, the
  LDS sites, Global Digital Library). It purged 601 documents, 467 of them
  Cebuano. A tempting shortcut — *"a host accepted in ≥3 languages is a
  farm"* — was tested first and is **wrong here**: the multi-language hosts
  are the Philippine Information Agency, Bombo Radyo, RMN and SunStar's
  regional editions, i.e. the best sources we have. Genuine multilingual
  publishing is common in the Philippines; don't penalise it.
- **Whole-document giants were the biggest distortion**: a 477k-word Cebuano
  Bible, a 221k-word Waray one, PDFs of the *Pasyon*. 23 + 13 documents over
  20k words (1.4 M + 0.7 M words) sit in `text_overlength/`, not in the
  shards.

**direct.** A file of URLs. No search. For curated domain lists (regional
newspapers, radio station sites, government portals in the language) and for
smoke tests — `scripts/scrape_smoke_urls.txt` has one page per language.

**fineweb2.** Streams `HuggingFaceFW/fineweb-2` for the language and pushes it
through the same gate, cleaner and dedup. This is the CommonCrawl-derived
corpus the `halo-*` sets came from, retained here so "fresh scrape" and
"bulk corpus" produce rows that are interchangeable. Carries over
`ingest_fineweb2.py`'s exclusions: Lsjbot Wikipedia for ceb/war (about 95 %
bot-written place stubs) and machine-translation content farms on
language-code subdomains (43 % of Cebuano FineWeb-2 by words).

Adding a backend is one class with a `search(query, lang, max_results) ->
list[Hit]` method and a line in `BACKENDS`.

## Row schema

halohalo's FineWeb-compatible columns unchanged — `id text url date dump
file_path detected_lang word_count title source language token_count
content_hash crawled_at` — so `prep_halohalo.py`, `append_to` and the
tokenizer reports consume the shards as-is. Plus:

| column | meaning |
|---|---|
| `backend`, `query` | which backend found it and with what query |
| `lid_model` | `halolid+glotlid` or the single model used |
| `lid_score` | decider's confidence (×0.8 if the two models disagreed) |
| `lid_agreement` | fraction of characters whose sentence voted for the winner |
| `lid_models_agree` | HaloLID and GlotLID picked the same language |
| `lid_detail` | both verdicts as JSON |

`source` is `web:{backend}` and `dump` is `web-{backend}-{YYYYMM}` (or the
FineWeb-2 dump id), which keeps the provenance discipline the dataset cards
promise: a shard can be traced to a backend, a month and a query.

## Language identification — and improving it

`scripts/train_lid.py` trains **HaloLID**, a fastText supervised model over
the ten PLD codes plus `other`, and scores it against **GlotLID v3** on a
held-out split. Training text is PLD prompts (sentence-deduplicated first —
prompts repeat across speakers, and a naïve split leaks them), halohalo and
BantayWika sentences; the `other` class is Indonesian and Malay web text so
the nearest non-Philippine languages are rejected rather than absorbed into
Cebuano. Character n-grams (2–5) carry most of the signal between languages
this closely related. The quantised model is tens of MB against GlotLID's
1.7 GB and runs an order of magnitude faster, which matters at web scale.

Results go to `$FINETUNE_DIR/lid/results.json` and the model card; `--push`
publishes to `sapinsapin/halo-lid`.

### LID rounds so far (2026-09-22)

Scored on the **PLD held-out set** (3,966 human-labelled sentences), the
only test set whose labels don't come from a model:

| round | change | acc | macro F1 | ≤5 words | fil | war | tsg |
|---|---|---|---|---|---|---|---|
| r1 | page labels as-is | *(mixed test set; eng 0.63)* | | | | | |
| r2 | + relabel confident English, cap classes at 20k | 0.854 | 0.833 | 0.722 | 0.759 | 0.763 | 0.822 |
| r3 | + **consensus labels** for web text, + scraped shards | 0.882 | 0.864 | 0.748 | 0.887 | 0.714 | 0.794 |
| **r4** | + flywheel round 1 (Tavily + expansion pages folded in) | **0.894** | **0.879** | **0.764** | **0.930** | **0.804** | **0.822** |
| GlotLID v3 | reference | 0.705 | 0.752 | 0.428 | 0.878 | 0.598 | 0.439 |

- **r3 → r4.** The flywheel's first round added 622 web pages, and the
  retrain recovered exactly the two languages r3 had lost: Waray 0.714 →
  0.804, Tausug 0.794 → 0.822. The mechanism is the one predicted above —
  consensus labelling had starved them of web text, and the new pages
  (Tausug's training rows grew from 4,611 to 5,465) fed them again. Small
  dips on bcl (0.855 → 0.843) and hil (0.853 → 0.824); macro F1 up, so
  promoted.

What each round taught us:

- **r1 → r2.** Web sentences inherit their page's label, so English
  sentences on Hiligaynon pages trained the model to call English `hil`.
- **r2 → r3.** Some page labels are simply wrong. The r2 model rejected
  5,747 FineWeb-2 *Filipino* pages as Hiligaynon; tracing that back
  showed `sapinsapin/halo-hil` is mostly English and Tagalog (GlotLID, 2,000
  sampled sentences: 44 % eng, 21 % fil, 12 % hil). r3 trains on a web
  sentence only when GlotLID agrees with its page label, which dropped
  38k `hil`-labelled sentences.
- **Still open.** Waray and Tausug slipped in r3. Their test sets are small
  (224 and 107), so it's borderline, but the likely cause is consensus
  labelling: it starves exactly the languages GlotLID is weak on. The next
  round should exempt tsg/war web text from consensus, or weight their PLD
  data up.
- **Don't read the web-test numbers as a comparison** from r3 on. Web labels
  are GlotLID consensus, so GlotLID scores ~1.0 on them by construction.

The loop that makes it improve "as we go": accepted pages where **both**
models agreed with score ≥ 0.9 are folded back in with
`--extra-parquet finetune_runs/scrape/text`, and pages where they
*disagreed* (`lid_models_agree = false` in the shards) are the hard examples
worth a human look before the next round. Retrain, re-score against the same
held-out set, publish only if it did not regress.

## Voice

`--voice` runs `halolib/scrape/voice.py`: yt-dlp `ytsearch` with the same
queries, LID on title (cheap, at discovery) then title + description (after a
per-video probe), duration bounds of 1 min – 4 h. Candidates are written to
`voice/{lang}/candidates.jsonl` with licence, channel, duration and the LID
verdict.

Download is **Creative Commons only by default** (`--all-licenses` to
override). Discovery costs nothing and is always safe; redistribution is not,
and the corpus cards promise per-source provenance, so non-CC finds are
listed for a person to decide on, not fetched. Downloads land as 16 kHz mono
WAV with a JSON sidecar (video id, channel, title, licence, query, LID). They
carry no transcript: this is raw speech for the livestream pipeline's
align / qc / pseudo-label path, not a finished dataset. `summary.json`
reports CC hours found per language, which is the number that decides
whether YouTube is worth pursuing for a given language at all.

**First run (2026-09-22), 8 queries per language:** Creative Commons speech
in Philippine languages on YouTube is effectively **zero**: no CC video in
any of the nine, one in English. LID-matched non-CC speech does exist:

| lang | found | LID-matched | CC | matched hours |
|---|---|---|---|---|
| ilo | 115 | 85 | 0 | 71.1 |
| pam | 88 | 45 | 0 | 16.3 |
| hil | 109 | 40 | 0 | 5.4 |
| bcl | 98 | 26 | 0 | 4.3 |
| ceb | 102 | 18 | 0 | 4.2 |
| war | 101 | 30 | 0 | 3.7 |
| fil | 99 | 22 | 0 | 2.6 |
| tsg | 80 | 4 | 0 | 2.5 |
| pag | 79 | 8 | 0 | 0.6 |

So for speech, the web route is a **permissions** problem, not a discovery
problem. The candidates files list channels: the next step is to ask
channel owners (regional radio, church, and LGU channels especially) for
permission, not to widen the search. The `--all-licenses` switch exists for
research use under a clear legal basis, not as a default.

## The flywheel — scrape, accumulate, improve LID, iterate

`scripts/flywheel.py` turns the one-shot scrape into a loop in which each
round is seeded by the rounds before it:

```
        ┌──────────────────────────────────────────────────────────┐
        │                                                          │
  PLD + halohalo + accepted pages ──► seeds ──► scrape ──► expand ─┤
        (keywords re-mined each round,   (Tavily,   (same sites,   │
         new query combinations,          advanced)  basic depth)  │
         spent queries never reused)                               │
                                                                   ▼
                          rounds.jsonl ◄── LID retrain, promote only if
                                            PLD macro-F1 does not regress
```

- **Seeds from what we found.** `load_seed_texts(scrape_dir=…)` folds in every
  accepted page whose two LID verdicts agreed at ≥0.9 confidence. Web prose
  yields better search terms than read prompts, and each round's finds shape
  the next round's queries. The query sampler takes the round number as its
  seed, so combinations differ, and `seeds/used_queries.json` guarantees a
  query is never issued twice.
- **Expansion inside known-good hosts.** The hosts that produced accepted
  pages become `include_domains` for a second pass at basic depth (1 credit
  a query instead of 2). Scripture mirrors, bot encyclopedias and generic
  platforms are excluded from expansion, so the loop does not amplify the
  sources that already dominate the small languages.
- **LID never regresses.** After each round HaloLID is retrained with the
  accumulated shards and promoted only if macro F1 on the human-labelled PLD
  set holds (within 0.002); otherwise the previous model is restored. The
  gate for round *n+1* is at least as good as the gate for round *n*.
- **Credit-aware.** Each round reads Tavily's usage endpoint and scales its
  query counts to what the plan has left rather than overrunning it.

```bash
python scripts/flywheel.py --rounds 1                          # ~240 credits at defaults
python scripts/flywheel.py --rounds 2 --queries-per-lang 6 --expand-queries 4
python scripts/flywheel.py --rounds 1 --langs tsg,war --no-lid
```

Everything accumulates in place: shards, manifests, `used_queries.json`, and
one JSON line per round in `finetune_runs/flywheel/rounds.jsonl` with credits
before/after, documents and words gained per language, and the LID decision.

### Round 1 (2026-09-23): 10 seed + 4 expansion queries per language

| lang | accepted | words | expansion hosts (examples) |
|---|---|---|---|
| eng | 117 | 181,643 | — |
| fil | 97 | 124,886 | remate.ph, pep.ph |
| bcl | 84 | 86,444 | bicolmail.net, bicolstandard.com, magbikolkita.com |
| ilo | 66 | 53,444 | tawidnewsmag.com, nordis.net, ilocossentinel.com |
| pam | 59 | 47,717 | punto.com.ph, kapampanganlibrary.whereishome.info |
| hil | 54 | 55,921 | aksyonradyoiloilo.com.ph, rmniloilo.net, digicastnegros.com |
| war | 45 | 39,778 | tacloban.bomboradyo.com, isumat.com, tacloban.gov.ph |
| ceb | 43 | 28,952 | sunstar.com.ph, rmn.ph, archives.pia.gov.ph |
| pag | 33 | 54,230 | punch.dagupan.com, vinceimbat.com |
| tsg | 24 | 22,082 | — (no eligible hosts yet) |
| **total** | **622** | **695,097** | |

Tausug went from 9 accepted in the one-shot run to 24 — the web-derived seed
terms (`hambuuk dayn ampa sabab`) find Tausug pages where read-prompt
vocabulary found English pages *about* Tausug. Cebuano's count is net of the
MT-farm purge that ran during the round. ~240 credits.

The round also surfaced the two exclusions above (MT farms by path,
adult hosts) and cost one lesson in robustness: the process died silently
during the LID retrain, so the record is now written before that step.

### Round 2: 6 seed + 3 expansion queries, nine Philippine languages, ~135 credits

312 documents / 236k words: ilo 68, hil 53, bcl 42, fil 34, ceb 32, pag 26,
pam 26, war 22, tsg 9. Expansion inside known-good hosts now out-yields the
seed pass for several languages (Bikol 25 of 42, Cebuano 23 of 32), which is
the flywheel working: the hosts found in round 1 are the best places to look
in round 2.

**The LID retrain was rejected** — macro F1 0.8792 → 0.8754 on the PLD set —
and r4 was restored. That is the promotion gate doing what it is for. More
web text does not automatically make a better identifier; the PLD test set
is read prompts, and each round tilts the training mix further toward web
prose. Two consequences worth acting on later: retrain less often than every
round (every ~1,000 new documents, say), and build the human-labelled *web*
test set the CPT plan already calls for, so the gate measures the domain the
gate is used in.

## What "AI-ready" means here, concretely

- FineWeb schema, so every downstream script we have already reads it.
- LID-gated with the evidence kept per row, not just a boolean.
- Exact and near deduplicated, and deduplicated again against the Hub on
  push (`append_to` compares `content_hash`).
- Cleaned by the same code as the published sets, so vocabulary statistics
  are comparable across `halo-*`, FineWeb-2 ingests and fresh scrapes.
- Provenance per row (backend, query, month, URL) and per audio file
  (sidecar), so a source can be excised if permission changes.
- Resumable and idempotent: re-running admits nothing twice.

## Known limits

- **Host skew.** Search results for a small language cluster on whichever
  sites publish in it at all: Bikol's first Tavily run was 41 % jw.org, with
  two more religious sites in its top six. Real Bikol, but a corpus that is
  half scripture is a narrow one. The summary prints the top hosts per
  language and `--max-per-host` can cap them; by default nothing is capped,
  because for the smallest languages one site may be most of what exists and
  the pretraining mix is the right place to rebalance.
- Wikipedia in the smoke list is a smoke list. ceb/war Wikipedia are excluded
  from real runs for the reason above.
- YouTube auto-captions are not used as transcripts. They are wrong often
  enough on these languages that the QC stage's whisper round-trip is the
  right judge, not YouTube's.
- HaloLID's `other` class covers Indonesian and Malay only; Spanish or
  Chavacano pages will be forced into the nearest Philippine label with low
  confidence — which the agreement threshold usually catches, but not always.
