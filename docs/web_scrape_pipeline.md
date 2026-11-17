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

From 2026-09-23 the PLD test set is a fixed hash split (4,035 sentences) and
the rows above are **not comparable** with the rows below; within the table
below, every row is scored on the same sentences.

| model | data | acc | macro F1 | ≤5 words | fil | war | tsg |
|---|---|---|---|---|---|---|---|
| r4 recipe (control) | scrape rows before flywheel round 2 | 0.889 | 0.869 | 0.764 | 0.920 | 0.707 | 0.807 |
| **r5** | + flywheel round 2 | **0.895** | **0.875** | **0.772** | **0.927** | 0.693 | **0.842** |
| r6 | + rounds 3–4 (1,267 pages) | 0.892 | 0.873 | 0.768 | 0.920 | 0.693 | 0.825 |
| **r7** | + round 5 (1,407 pages) | **0.896** | **0.878** | **0.777** | **0.929** | **0.744** | **0.842** |
| GlotLID v3 | reference | 0.710 | 0.748 | 0.433 | 0.901 | 0.470 | 0.465 |

On a fair test Waray is the weak language (0.69–0.71, next lowest 0.83);
r4's 0.804 was an artefact of the shifting split.

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

### Language codes

The labels are ISO 639-3 individual-language codes, the convention of
FineWeb-2, GlotLID and the LID leaderboard (which append the script:
`ceb_Latn`): `bcl` Central Bikol, `ceb` Cebuano, `eng` English, `fil`
Filipino, `hil` Hiligaynon, `ilo` Iloko, `pag` Pangasinan, `pam`
Kapampangan, `tsg` Tausug, `war` Waray. Two merges to keep in mind when
comparing with other benchmarks:

- **`fil` covers Tagalog (`tgl`)** too. FLORES+, FineWeb-2 and Gherbal use
  `fil_Latn`; WiLI-2018 and Bouquet use `tgl_Latn`; CommonLID has both as
  separate gold labels.
- **`bcl` is Central Bikol.** The PLD's raw code is `BIK`, the Bikol
  *macrolanguage* (`halolib/pld.py` maps it to `bcl`); CommonLID carries
  both `bcl` and `bik`. Our Bikol web sources (Naga and Legazpi radio,
  Bicol Mail) are Central Bikol, so the mapping holds for the corpus.

`other` is internal, not an ISO code. The HaloLID model card declared
`[tl, …, en]` (2-letter codes, and no `fil` although `fil` is its class); it
now uses the same 639-3 list as the dataset cards, plus `tl`.

### Benchmarks against the public LID leaderboard (2026-10-02)

`scripts/bench_lid.py` compares HaloLID (r8) with the 18 models of Omneity
Labs' [LID leaderboard](https://huggingface.co/spaces/omneity-labs/lid-benchmark),
which publishes per-language results but does not run submissions. Of its
eight benchmarks, our languages appear in FLORES+ (ceb, eng, fil, ilo, pag,
war), WiLI-2018 (bcl, ceb, eng, ilo, pag, pam, tgl, war), CommonLID (bcl,
bik, eng, fil, tgl) and Bouquet (ceb, eng, ilo, tgl, tsg, war). **None has
Hiligaynon.** FLORES+, CommonLID and Bouquet are gated on the Hub (accept
the terms once per account); WiLI-2018 is open.

**WiLI-2018, full test set (117,500 paragraphs, 235 languages).** HaloLID
ranks 13th of 19 by mean per-language F1 over our eight (0.214; GlotLID
0.721, Gherbal-v4 0.497). Recall is 0.96–1.00 for seven of the eight, but
precision is 0.01–0.22: HaloLID's `other` class was trained only on
Indonesian and Malay, so text in the other ~225 languages has nowhere to
go. English absorbs Europe (44,701 false positives: Scots, French, Dutch,
German…), and scripts it has never seen (Thai, Hebrew, Malayalam, Burmese)
land on Philippine classes. Only 41 % of non-Philippine text reaches
`other`. In the scraper this is masked (pages come from target-language
queries and must also pass GlotLID), but **HaloLID is not an open-set
identifier**. The fix is a broad `other` class (FineWeb-2 text from ~100+
languages, Latin and non-Latin) plus a script check.

**Our PLD held-out set (4,035 human-labelled sentences, all ten
languages)**, with Gherbal-v4 queried through the Sawalni API
(`/v1/language/identify`):

| model | macro F1, all 10 | macro F1, Gherbal's 6 | accuracy |
|---|---|---|---|
| **HaloLID r8** | **0.878** | **0.864** | **0.895** |
| GlotLID | 0.748 | 0.747 | 0.710 |
| Gherbal-v4 | 0.380 | 0.634 | 0.426 |

Gherbal-v4 supports six of our ten (not bcl, hil, pam, tsg). HaloLID's lead
here is real but partly home advantage: it trains on other PLD sentences in
the same style, and as a closed set it cannot confuse Cebuano with Javanese.
The two results together said what r8 was: the best identifier *among*
our ten languages, and not a general one.

### r9: an open-set `other` class (2026-10-03)

Three changes, each tested on its own:

1. **Broad negatives.** `scripts/build_lid_negatives.py` samples one row
   group per language from FineWeb-2 (a different source from every
   benchmark): the 150 largest Latin-script languages, the neighbours and
   the *other* Philippine languages weighted ×4 (Kinaray-a, Chavacano,
   Maranao, Javanese, Sundanese, Malay…), and 15 non-Latin languages:
   38,610 sentences from 186 languages in `lid/negatives.jsonl`, which
   `train_lid.py` now uses for `other` (capped at 60,000 rather than 20,000).
2. **A Latin-script rule.** All ten languages are written in Latin script, so
   `HaloLID.predict` returns `other` when under half of a text's letters are
   Latin. Alone it lifts WiLI mean F1 from 0.214 to 0.369.
3. **Cleaned negatives.** The first broad model cost 0.004 PLD macro F1,
   English worst (0.972 → 0.954): FineWeb-2's small configs carry English
   boilerplate, so `other` was being taught English. `consensus_web` now
   drops any negative GlotLID calls one of our ten (1,359: 1,134 English,
   89 Hiligaynon…), and the loss disappeared.

Trained on r8's exact data (`--extra-before`, so the comparison isolates the
`other` class), scored against r8 on the same sentences:

| | WiLI-2018 mean F1 (our 8) | PLD macro F1 | web macro F1 |
|---|---|---|---|
| r8 | 0.214 | 0.8781 | 0.9344 |
| r8 + script rule | 0.369 | — | — |
| **r9** | **0.813** | **0.8788** | **0.9881** |

WiLI precision went from 0.01–0.22 to 0.72–0.998 for seven of the eight
languages, with recall unchanged, and r9 now has the best mean of the 19
models on our languages (GlotLID 0.721, which lacks a Tagalog score
because it answers `fil`; on the seven shared languages GlotLID 0.824 and
r9 0.803 are level). Still weak: **English** (F1 0.33; WiLI's English
class is hard for every model, 0.21–0.38) and **Kapampangan** (recall 0.34:
WiLI's `pam` paragraphs go to English for every model, GlotLID 0.51).
On the PLD set r9 scores 0.879 macro F1 against GlotLID's 0.748 and
Gherbal-v4's 0.380. Promoted locally as r9 through the flywheel gate; r8 is
kept in `lid_bench/r8_backup/`.

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
  PLD + halohalo + accepted pages ──► seeds ──► scrape ──► expand ──► crawl ─┤
        (keywords re-mined each round,   (Tavily,   (same sites,  (their      │
         new query combinations,          advanced)  basic depth)  sitemaps,  │
         spent queries never reused)                               free)      │
                                                                              ▼
            rounds.jsonl, lid_events.jsonl ◄── every ~1,000 new docs: retrain LID,
            host_stats.json                    score it against the incumbent on
                                               the same test set, promote only if
                                               neither PLD nor web F1 regresses
```

- **Seeds from what we found.** `load_seed_texts(scrape_dir=…)` folds in every
  accepted page whose two LID verdicts agreed at ≥0.9 confidence. Web prose
  yields better search terms than read prompts, and each round's finds shape
  the next round's queries. The query sampler takes the round number as its
  seed, so combinations differ, and `seeds/used_queries.json` guarantees a
  query is never issued twice.
- **Expansion inside known-good hosts.** The hosts that produced accepted
  pages become `include_domains` for a second pass at basic depth (1 credit
  a query instead of 2). Scripture mirrors, bot encyclopedias and the bare
  blog platforms are excluded from expansion, so the loop does not amplify
  the sources that already dominate the small languages. A single blog on a
  platform (`tausugblog.blogspot.com`) is a site and is eligible.
- **A free crawl of the same hosts.** `SitemapBackend` reads each known-good
  host's sitemap (robots.txt `Sitemap:` lines, then the WordPress/Blogger
  locations, then the home page's links) and serves a sample of pages no
  manifest has seen yet, `--crawl-per-host` per host per round. It costs no
  credits, so the loop keeps growing after the search allowance is spent.
- **Hosts that stop paying are retired**, per pass type: two passes in a row
  with no new accepted page, or ≥ 20 pages tried at < 10 % acceptance
  (multilingual sites whose pages are mostly another language).
  `host_stats.json` holds the per-host counts.
- **LID never regresses, measured fairly.** Once `--retrain-every` (1,000)
  new documents have accumulated since the last gate, HaloLID is retrained
  and the candidate and the incumbent are scored on the *same* held-out
  sentences in the same run (`train_lid.py --baseline`). The candidate is
  promoted only if neither PLD nor web macro F1 drops by more than 0.002
  against the incumbent, and PLD stays within 0.002 of the best score any
  promoted model has had, so small tolerated drops cannot accumulate.
- **Credit-aware.** Each round reads Tavily's usage endpoint and scales its
  query counts to what is left of the plan and of `--credit-budget` (a total
  for the invocation), and records the credits actually spent.

```bash
python scripts/flywheel.py --rounds 1                          # ~220 credits at defaults, 9 langs
python scripts/flywheel.py --rounds 2 --queries-per-lang 6 --expand-queries 4
python scripts/flywheel.py --rounds 3 --queries-per-lang 0 --expand-queries 0   # crawl only, free
python scripts/flywheel.py --rounds 1 --langs tsg,war --no-lid
python scripts/flywheel.py --lid-only                          # run the LID gate now
```

Everything accumulates in place: shards, manifests, `used_queries.json`, and
one JSON line per round in `finetune_runs/flywheel/rounds.jsonl` with credits
spent, documents and words gained per language, the hosts each pass used,
and the LID decision (or why it was deferred).

### The held-out split is a function of the sentence

Until round 2, `train_lid.py` chose its test set with one seeded shuffle
that walked every language's lists in turn. More web text for one language
changed how much randomness the shuffle consumed and therefore *which PLD
sentences* the next language held out. Two consequences:

- the gate compared the new model's F1 on one test set with the old model's
  F1 (read from its `results.json`) on another. GlotLID, which never
  changes, scored 0.7524 in the r4 run and 0.7483 in the round-2 run: a
  0.004 swing from the test set alone, the size of the "regression" that got
  the round-2 model rejected;
- sentences the old model had trained on could land in the new test set.

The split is now `sha1(sentence) < 10 %` (`SPLIT = "hash-v1"`): a sentence
is test or train forever, the PLD test set is identical in every run, and no
model has trained on any test sentence. An incumbent trained before the
change cannot be scored fairly on the new split, so `--lid-rebase CUTOFF`
retrains the incumbent's recipe (scrape rows crawled before `CUTOFF`) on
the new split as a control, trains a candidate with everything, and
promotes whichever wins.

A second bug inflated spending: `run_text` iterated every query in the seed
cache, not `queries_per_lang` of them, so round 2's "3 expansion queries"
ran 6 per language (60 hits = 6 × 10). It now slices.

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

### Round 2: 6 seed + 6 expansion queries (3 intended), nine Philippine languages, ~160 credits

312 documents / 236k words: ilo 68, hil 53, bcl 42, fil 34, ceb 32, pag 26,
pam 26, war 22, tsg 9. Expansion inside known-good hosts now out-yields the
seed pass for several languages (Bikol 25 of 42, Cebuano 23 of 32), which is
the flywheel working: the hosts found in round 1 are the best places to look
in round 2.

**The LID retrain was rejected** — macro F1 0.8792 → 0.8754 on the PLD set —
and r4 was restored. In hindsight that verdict was noise: the two numbers
came from different test sets (see *The held-out split is a function of the
sentence* above). The fixes that followed: a hash split, the candidate and
the incumbent scored in the same run, retraining every ~1,000 new documents
rather than every round.

Tausug's expansion found 30 hits and 0 new pages, because its only
"eligible" host was `bibles.org` (277 of its 387 pages), whose pages were
all already in the manifest, while its genuinely Tausug blogs were
excluded by the `blogspot.com` platform rule. Both are fixed above.

The rebase (`--lid-rebase 2026-09-23T18:05:13+00:00`) reproduced r4's data
exactly (Tausug 5,465 training rows, as in r4) on the new split, and the
candidate with round 2's pages beat it on the same sentences: PLD macro F1
0.8687 → 0.8746, web 0.9914 → 0.9910 (within tolerance). Promoted as r5.
The round-2 pages the old gate rejected improve the model.

### Round 3: 2 seed + 4 expansion queries, plus the sitemap crawl, 72 credits

595 documents / 244k words: ceb 108, ilo 106, pam 90, hil 79, war 60,
fil 58, bcl 57, pag 28, tsg 9. By pass:

| pass | credits | accepted |
|---|---|---|
| seed | 36 | 50 |
| expansion | 36 | 139 |
| **crawl** | **0** | **404** |

The free crawl brought in twice what both paid passes did. Most of it came
from single blogs the platform rule used to exclude (Cebuano's three
Blogspot news blogs: 73 pages; Kapampangan's WordPress blogs). Tausug
remains the exception: its hosts' sitemaps are small or gone
(`ahmadiyyaphilippines.org` 404s, `tausugonline.com` does not answer), and
expansion returned only pages already held: 9 new documents in total.

Two further findings:

- **Scripture blogs.** With single blogs eligible, `anlibronimormon.blogspot.com`
  (the Book of Mormon in Bikol) supplied 23 of Bikol's 33 crawled pages.
  Hosts whose names contain `bible`, `biblia`, `mormon` or `scriptur` are
  now excluded from expansion and crawling by name; parish blogs (local
  news) stay eligible.
- **isumat.com**, the source of 555 Waray documents, is now a gambling site.
  Those documents came from FineWeb-2's CommonCrawl snapshots (2021–2024)
  and are genuine Waray news; the crawler finds no same-host URLs and the
  host is retired.

Hosts retired after round 3 for low acceptance on the crawl (multilingual
sites whose pages are mostly another language) include `bomboradyo.com`
for bcl/hil/pag, `punto.com.ph` for pam and `sufihub.com`/`islamhouse.com`
for tsg.

### Round 4: crawl only, 0 credits

With 12 credits left on the plan, round 4 ran the crawl alone (10 hosts ×
40 pages per language) with scripture blogs and round 3's retired hosts
excluded: 672 documents / 201k words — ceb 137, ilo 116, hil 107, pam 98,
bcl 59, war 57, fil 51, pag 39, tsg 8. Retiring low-acceptance hosts showed
at once: Hiligaynon's LID rejections fell from 80 of 150 crawled pages to 32
of 160.

About 1,070 documents had accumulated since the rebase, so the round
retrained. The candidate scored PLD macro F1 0.8734 against r5's 0.8746
(−0.0012, inside the 0.002 tolerance) and web 0.9913 against 0.9905, and
was promoted as r6: a tie, not an improvement. It exposed a flaw in a
per-step tolerance: a run of within-tolerance promotions can drift downhill
indefinitely. Because the PLD test set is now fixed, scores are comparable
across runs, and the gate also requires PLD macro F1 within the tolerance
of the **best** score any promoted model has had (`best_pld_f1`, from
`lid_events.jsonl`).

**Corpus after four rounds: 23,570 documents / 12.41M words** (from
22,303 / 11.96M after round 2).

### Round 5: crawl only, 0 credits, 12 hosts × 60 pages

1,407 documents / 400k words: ilo 300, ceb 284, fil 277, hil 213, pam 112,
bcl 97, tsg 47, pag 36, war 41. Tausug's 47 is more than rounds 1–4 of
the flywheel found for it together, nearly all from two Mindanao blogs
(`mowa-mindanaoans.blogspot.com` 39, `minsupala.blogspot.com` 7) that
earlier crawls had surfaced as accepted hosts and this wider crawl reached.

The retrain (r7) is the first real gain on the fixed split: PLD macro F1
0.8734 → **0.8778** (new best; accuracy 0.896), web 0.9910 → 0.9912, and
Waray, the weak language, 0.693 → 0.744. Bikol (0.882 → 0.871) and English
(0.967 → 0.944) slipped.

**Corpus after five rounds: 24,977 documents / 12.81M words**; Tausug 451.

### Audit, 2026-09-26: exclusions were never applied retroactively

Host exclusions run at scrape time, so each new rule left what it would have
caught in the shards. The adult rule of 2026-09-23 had not removed
`filipinosexstories.com`, the site its own comment names (63 Filipino rows),
nor `pserotica.com`, which its `erotic` fragment matches. An audit also found
adult hosts the fragments miss (`…sexgames.com`), gambling (online sabong,
casinos) and machine-translated game portals on `ph.` subdomains. The
crawl was amplifying the last two: `ph.sgames.org` went from 12 pages to
71 and `todosabongga.blogspot.com` from 13 to 59 in two rounds.

The fix: new adult fragments, `GAMBLING_HOST_FRAGMENTS` and `SPAM_HOSTS` in
`excluded_host`, and `scripts/purge_excluded.py`, which applies every current
exclusion to the shards and marks the URLs `host_excluded` in the
manifests. The audit counted 283 such documents; by the time the purge ran
(after rounds 6–8, which still used the old rules) the crawl had taken
`ph.sgames.org` to 240 pages and `todosabongga.blogspot.com` to 202, and
the purge removed **595 documents / 285k words** (566 of them Filipino).
None of it had been published: the `sapinsapin/halo-{lang}` scrape
datasets do not exist on the Hub. Run the purge after adding any exclusion
rule (`--shards-only` while the flywheel runs).

### Rounds 6–8: new Tavily key (Growth plan), 10 seed + 6 expansion queries, crawl 12 × 60

6,040 documents / ~2.6M words for 702 credits (234 a round), before the
purge above; Tausug +277. Search is cheap at this scale: each round's
paid passes found ~500 documents, the crawl ~1,500.

**All three LID retrains were rejected** (PLD macro F1 −0.006 to −0.0085
against r7; web +0.0003 to +0.0016); r7 remains the incumbent. The drop is
mostly Waray (0.744 → 0.684 in round 6), whose 215-sentence test set makes
it volatile. Three rejections in a row point at the recipe, not the data:
`train_lid.py` caps every class at 20,000 sentences, PLD first, so for
capped languages more pages do not add web text, they change *which* web
sentences fill the cap, increasingly blog-crawl prose.

That hypothesis was tested and **rejected**. Doubling the cap to 40,000
(`finetune_runs/lid_exp_cap40k`, scored against r7 on the same sentences)
gives PLD macro F1 0.8705 (−0.0073, the same drop as the rejected
retrains) and web 0.9940 (+0.0034; ≤5-word web accuracy 0.952 → 0.966).
More web text makes the identifier better on web prose and worse on read
prompts (Bikol −0.018, Waray −0.033). The rejections are a **domain
trade-off**, not a broken recipe: the gate privileges PLD, while the
model's job is gating web pages. Deciding it needs the human-labelled web
test set: the current web labels are GlotLID consensus, so part of any
web gain is agreement with GlotLID.

### Rounds 9–10: 20 seed + 10 expansion queries, 450 credits a round

4,349 documents / 2.2M words (2,198 and 2,151), Tausug +240, with the
gambling/spam rules in force (the purge afterwards found nothing new).
Both LID retrains rejected the same way (PLD −0.0052 and −0.0087, web up),
so the flywheel now retrains every 4,000 documents until the web-vs-PLD
question is settled.

Round 11 died twice, three hours and then 30 minutes in, each time with
the WSL instance: CPU machine-check kernel panics on the host
(`%LOCALAPPDATA%\Temp\wsl-crashes\kernel-panic-*.txt`), a hardware fault
that kills every WSL job at once. Long runs are now relaunched from a
Windows-side supervisor after each panic. Rows
accepted in memory but not yet flushed were lost while their URLs stayed
`accepted` in the manifest, so no later run would fetch them again; round
1 had lost rows the same way. `Manifest.forget_unsaved` now un-sees any
accepted URL whose content hash is not in the shards, at the start of every
pass, so an interrupted run heals on the next one.

**Corpus after ten rounds (and the purge): 34,978 documents / 17.58M
words**; Tausug 968, from 387 before the flywheel's fixes.

Further hardening after round 11 was interrupted six times more that
afternoon (a second panic, then an out-of-memory condition in another
session's WSL job that killed every WSL process every ~90 s):

- **Rounds resume instead of restarting.** `round_NNN.progress.json` records
  each finished (pass, language) and that the seed queries were drawn; a
  relaunched round skips both. Before this, each relaunch drew and paid for
  a new seed pass: round 11 consumed 1,312 credits over four attempts
  without completing.
- **Shards are written durably** (temp file, fsync, rename). A VM killed
  after an ordinary write left `ceb/shard-00033` with a valid footer and
  garbage pages, which made every shard reader fail.
  `quarantine_corrupt_shards` runs at the start of every round, moves
  unreadable shards to `{lang}/quarantine/` and leaves an empty shard in the
  slot; the lost rows are re-fetched through `forget_unsaved`.
- The launcher never guesses a credit budget: if Tavily's usage endpoint
  does not answer, the round only crawls.

### Rounds 11–24: the rest of the 5,000 credits, then crawl only

A Windows-side supervisor carried the run through **11 WSL panics**
overnight (01:38–05:42, one every ~20 minutes); resumable rounds, durable
shards and the re-fetch of unsaved rows meant each cost minutes, not a
round. All 466 shards read cleanly afterwards.

- Paid search spent 4,832 of the 5,000-credit cap across rounds 6–15
  (rounds 11–15: 1,918 recorded, plus the interrupted attempts). Rounds
  16–24 were crawl-only and free, declining from ~1,500 to ~1,290
  documents a round as the known hosts' sitemaps are used up.
- Rounds 11–24 added 22,387 documents / 9.35M words. Tausug's yield dried up
  (0 in every crawl-only round: its hosts are exhausted); it ends at 1,257.
- Host concentration grew: Bombo Radyo's regional stations supplied about
  30 % of the documents accepted since round 11. Worth a per-host cap in the
  CPT mix, not in the scrape.
- LID: r8 promoted in round 15 at +0.0003 PLD macro F1 (a tie); four
  retrains rejected (−0.0043 to −0.0105), the same web-vs-PLD trade-off.
- An audit of the new top hosts found one more machine-translation farm,
  `lawandmore.services` (a Dutch law firm "in Cebuano": 676 pages, 1.51M
  words, ~6 % of Cebuano), now in `SPAM_HOSTS` and purged. The rest of the
  top 25 by words are blogs, regional news, *Bannawag* editorials, radio
  commentary and homilies.

**Corpus after 24 rounds and purges: 56,689 documents / 25.42M words**
(from 22,303 / 11.96M after round 2).

Still open: the human-labelled *web* test set the CPT plan calls for. The
web test set the gate uses has GlotLID-consensus labels, so it can catch a
model that forgets web prose but cannot say which of two good models is
better on it.

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
