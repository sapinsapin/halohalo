# Continued pretraining (CPT) of LLMs for the ten PLD languages

Plan written 2026-09-22. Every number in §1 was measured in this repo on that
date; everything in §§3–7 is a plan and says so.

**One-sentence summary.** Continued pretraining for these languages is limited
by data, not compute. A full run over every Philippine token we can currently
find costs a few hundred dollars of GPU time. So the plan spends its effort on
the data engine and on evaluation, runs a small three-way base-model bake-off
before committing, and sizes the main run to the data rather than the budget.

---

## 0. What already exists — and the lesson in it

The org has three CPT models: `llama31-8b-balitanlp-cpt` and
`gpt-oss-20b-balitanlp-cpt` (Filipino news, BalitaNLP), and `bikoLLM`
(Llama-3.1-8B on halo-bikol). All three are single-language, and **none of
their cards reports an evaluation** — every one says "More information
needed". Nobody can say whether they are better than their base models at
Filipino or Bikol, or what they forgot doing it.

So the first deliverable of this plan is an evaluation harness, and nothing
trains until it can score the bases.

---

## 1. Measured facts

### 1.1 Tokenizers — no candidate base fits these languages

Fertility (tokens per word) on sentence-deduplicated PLD prompts, 3,000 per
language, measured with each model's own tokenizer:

| tokenizer | vocab | bcl | ceb | eng | fil | hil | ilo | pag | pam | tsg | war | PH mean | vs eng |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **gpt-oss** | 200k | 1.68 | 1.66 | 1.06 | 1.59 | 1.60 | 1.72 | 1.72 | 1.67 | 1.65 | 1.54 | **1.65** | **1.56×** |
| SEA-LION v3 (Gemma-2) | 256k | 1.70 | 1.69 | 1.04 | 1.66 | 1.64 | 1.71 | 1.70 | 1.67 | 1.65 | 1.56 | 1.66 | 1.61× |
| Gemma-3 | 262k | 1.72 | 1.71 | 1.04 | 1.69 | 1.65 | 1.73 | 1.72 | 1.69 | 1.65 | 1.58 | 1.68 | 1.62× |
| Mistral-Nemo | 131k | 1.88 | 1.84 | 1.08 | 1.86 | 1.81 | 1.92 | 1.83 | 1.82 | 1.81 | 1.71 | 1.83 | 1.69× |
| Llama-3.1 | 128k | 1.93 | 1.88 | 1.07 | 1.93 | 1.85 | 1.96 | 1.87 | 1.87 | 1.88 | 1.75 | 1.88 | 1.76× |
| Qwen3 | 152k | 1.95 | 1.90 | 1.07 | 1.96 | 1.87 | 1.97 | 1.89 | 1.88 | 1.90 | 1.77 | 1.90 | 1.77× |

Every base pays a 56–77 % token tax on Philippine text. The spread between
the best and worst base (1.65 vs 1.90 tokens per word) is itself 15 %: the
same corpus is 15 % more expensive to train on, and the model sees 15 % less
text per context window. That's a real input to the base choice, not a
footnote. See [`reference/tokenizers.md`](reference/tokenizers.md) for why
fertility also understates the damage (stem identity).

### 1.2 Data — the binding constraint

FineWeb-2 document counts, and our scrape of the first ≤3,000 documents per
language through the full gate (LID + clean + dedup,
[`web_scrape_pipeline.md`](web_scrape_pipeline.md)). The estimate
extrapolates the measured acceptance rate and words per document to the
whole FineWeb-2 split, at Llama-3.1 fertility:

| lang | FW-2 docs | scraped | accept rate | words/doc | **est. clean tokens** |
|---|---|---|---|---|---|
| fil | *large* | *running* | — | — | **measure** — the only language likely in the billions |
| ceb | 206,653 | 3,000 | 82 % | 592 | **~189 M** (upper bound: residual MT/bot text survives the domain filters) |
| hil | 44,190 | 3,000 | 96 % | 492 | ~39 M |
| ilo | 21,528 | 3,000 | 98 % | 402 | ~17 M |
| bcl | 8,895 | 3,000 | 93 % | 434 | ~7 M |
| pag | 2,752 | 2,353 | 89 % | 560 | ~2.6 M |
| pam | 2,032 | *running* | — | — | ~2 M (by size) |
| war | 2,258 | 1,845 | 92 % | 305 | ~1.1 M |
| tsg | 353 | 345 | 99 % | 405 | **~0.3 M** |

Add our own corpora: halohalo (hil, tgl, bcl web + BantayWika literary fil,
about 45k documents), BalitaNLP (Filipino news, used by the earlier runs),
and PLD transcripts (read prompts, small).

**What this means.** Seven of the nine Philippine languages have under 40 M
tokens of web text *in total*. Tausug has 0.3 M, which is less than one
batch of a serious pretraining run. A Chinchilla-style budget
(~20 tokens per parameter) for an 8B model is 160 B tokens, and we have
perhaps 1–3 B across all ten languages. So:

- The data mix, not the step count, decides quality.
- The small languages cannot be trained alone. They have to borrow from
  Filipino and Cebuano through joint multilingual training, and from
  related Austronesian languages.
- The web scrape (Tavily, news sites, radio transcripts, community text)
  is the highest-return investment in this plan. It is the only lever that
  moves the ceiling.

---

## 2. Goals and non-goals

**Goal.** One multilingual base model, continued-pretrained on all ten
languages jointly, that is measurably better than its starting point at
every one of them (bits per byte, downstream tasks) and measurably not worse
at English. Released with its evaluation, its data card, and its licence
lineage.

**Also a goal.** A cheap, repeatable recipe: when the scrape doubles a
language's data, the model can be refreshed for tens of dollars.

**Not a goal (yet).** Instruction tuning, chat, or safety tuning. Those come
after the base is proven, and they need Philippine-language instruction data
we don't have yet.

---

## 3. Evaluation first (phase 0)

Built and frozen before any training, following the frozen-split discipline
the speech work already uses.

| metric | per language | why |
|---|---|---|
| **bits per byte** on held-out text | all 10 | Tokenizer-independent, so bases with different tokenizers and extended vocabularies compare fairly. Perplexity doesn't. Held-out = speaker/document-disjoint slices of scrape + halohalo, decontaminated. |
| **Belebele** reading comprehension | **tgl, ceb, ilo, war** (verified 2026-09-22: 4 of our 10 among its 122 configs) | Multiple-choice, parallel across languages, so gains are comparable across languages. |
| **SIB-200** topic classification | **tgl, ceb, ilo, pag, war** (verified: 5 of 10 among 205 configs) | Derived from FLORES; cheap, few-shot. |
| **FLORES-200** ↔ eng | at least **tgl, ceb, ilo, pag, war** (inferred: SIB-200 is built from FLORES-200 and covers these) | Few-shot translation, chrF++. Also a check on the parallel-data ablation. |
| **Filipino LLM suites** (FilBench, Global-MMLU fil, SEA-HELM fil) | fil | Filipino has dedicated suites; use them rather than inventing one. **[verify availability and licences]** |
| **English retention** (MMLU, HellaSwag, ARC subset) | eng | CPT's known failure mode is forgetting. Report the delta. |
| **Native-speaker generation review** | fil, ceb, hil, ilo first | 50 prompts per language, blind A/B against the base. The only check that catches fluent nonsense. |

**The coverage gap is the real problem.** Five of the ten languages —
**bcl, hil, pam, tsg, and eng-as-Philippine-English** — appear in *no*
standard NLU benchmark. For them, bits per byte and native-speaker review are
the only signals, and bits per byte cannot tell whether the model learned
Philippine grammar or just memorised surface statistics. So phase 0 must also
build:

- **A morphological probe set.** Austronesian voice/focus alternation (actor,
  patient, locative, instrument), aspect and reduplication are what make these
  languages distinct, and nothing in MMLU, Belebele or SIB-200 touches them.
  Extract 50–200 verb lemmas per language from PLD transcripts and build
  minimal pairs with a native speaker or a reference grammar. Cheap
  (~$200–500 of annotator time) and it is the only diagnostic that answers the
  question the whole project is about.
- **A code-switching probe**, from held-out livestream Taglish, since that is
  how people actually write.

Deliverable: `scripts/eval_llm.py` with one row per (model, language,
metric), and the base-model scores published *before* training starts, so
every later claim has a baseline row.

Decontamination: 13-gram overlap filter between the training mix and every
eval set, applied to the mix, logged per source.

---

## 4. Base model — a bake-off, not a guess

Three candidates, chosen for different reasons:

| base | size | licence | tokenizer (PH mean) | why it's in |
|---|---|---|---|---|
| **Qwen3-4B / 8B** | dense | Apache-2.0 | 1.90 | Strong multilingual base, permissive, dense and easy to train; the commercial-track default if it wins. |
| **Gemma-3-4B / SEA-LION v3** | dense | Gemma terms | 1.68 / 1.66 | Better tokenizer. SEA-LION was already continued-pretrained on Southeast Asian data, so it may start ahead. **[verify SEA-LION v3 Filipino coverage]** |
| **gpt-oss-20b** | MoE, ~3.6B active | Apache-2.0 | **1.65** | Best tokenizer, permissive, cheap per token (MoE), and the org already has a BalitaNLP CPT of it to compare against. Harder to train (MoE routing). |

**Bake-off protocol.** Each base gets the *same* ~1 B-token mix (§5) and the
same schedule, on one RTX PRO 6000 for the ≤4B variants. Score all three
with the phase-0 harness and pick on the eval, weighted towards the small
languages; English retention acts as a gate. The earlier single-language
CPTs join the table as extra rows once the harness can score them.

Licence note: the research track may use any of these. The commercial track
can only ship an Apache-2.0 base and commercially clean data, and **PLD
transcripts are CC-BY-NC**, so they stay out of any commercial-track mix
(see [`pld_sota_track.md`](pld_sota_track.md) §1).

---

## 5. Data mix

### 5.1 Sources

| source | languages | role | licence |
|---|---|---|---|
| FineWeb-2 via `scrape_web.py --backend fineweb2` | all 9 PH | bulk web text, LID-gated, bot/MT-farm filtered | ODC-By |
| Tavily scrape (`scrape_web.py`, default backend) | all 10 | fresh, targeted text beyond CommonCrawl; the lever for the small languages | per-site; provenance kept per row |
| halohalo + BantayWika | fil, hil, bcl (+ilo, ceb) | curated literary and web text — **re-gate per sentence with LID first**: `halo-hil` is mostly English and Tagalog under a `hil` label (see `reference/datasets-benchmarks.md`) | per card |
| BalitaNLP | fil | news | **[verify licence]** |
| Indonesian / Malay (FineWeb-2 `ind_Latn`, `zsm_Latn`) | — | related-language bridge (ablation) | ODC-By |
| English replay (FineWeb-Edu sample) | eng | anti-forgetting | ODC-By |
| fil↔eng parallel (small) | fil, ceb, ilo | cross-lingual alignment (ablation) | **[source TBD]** |
| PLD transcripts | 10 | read prompts; research track only | CC-BY-NC |

### 5.2 Mixing

- **Temperature sampling across languages**, p ∝ n^α with α ≈ 0.3 (the
  mT5 / XLM-R setting). Pure proportional sampling would let Cebuano
  (~190 M) drown Waray (~1 M). α = 0.3 upsamples the small languages
  without letting them dominate.
- **Epoch cap: ≤4 repetitions of any source.** Data-constrained scaling work
  (Muennighoff et al., 2023) finds repeated data nearly as good as fresh up to
  about four epochs, with sharply diminishing returns after that. For tsg and
  war that cap binds almost immediately, which is honest: their share is
  limited by what exists.
- **English replay ≈ 25 %** of tokens. Continual-pretraining work (Ibrahim
  et al., 2024) shows modest replay plus LR re-warming/re-decaying prevents
  most forgetting at a small cost to in-domain gains. Tune on the English
  retention metric.
- **Quality filtering.** The LID gate, cleaner and dedup already run. Later,
  train a FineWeb-Edu-style quality classifier on Filipino annotations; not
  before we have annotators.

**Indicative first mix (~1 B tokens, bake-off size):** 45 % fil, 15 % ceb,
25 % English replay, 15 % shared by hil/ilo/bcl/pag/pam/war/tsg under α = 0.3
with the epoch cap. Recompute it from the measured table once the fil and pam
scrapes finish; don't hard-code it.

---

## 6. Tokenizer — measure, then decide

Vocabulary extension is a much better deal for CPT than it was for ASR
(where we judged it risky): CPT trains new embeddings on hundreds of millions
of tokens, which is exactly what new rows need.

- **Ablation arm:** extend the winning base's BPE with ~8–16k Philippine
  merges, mined from the §5 mix pooled across all nine languages (shared
  units, since most affixes are cognate). Compare on bits per byte and
  downstream tasks at equal *compute*, not equal steps — the extended model
  covers more text per step.
- **Implementation:** extend the **merge list**, not `add_tokens`. Measured
  in this repo: `add_tokens` on byte-level BPE orphans the leading space, so
  `maganda` becomes `Ġ|maganda` and saves nothing. Appending merges makes
  `Ġmaganda` a real token (20 → 16 tokens on a test sentence, exact round
  trip). Place new ids after *all* existing special tokens. Initialise each
  new embedding (input and output) as the mean of the subtokens it replaces.
- **Keep it only if** it wins on bits per byte per FLOP and doesn't regress
  English.

---

## 7. Training recipe and compute

**Recipe (starting point, tuned in the bake-off):** full-parameter CPT
(LoRA learns less and forgets less — Biderman et al., 2024 — so it's the
wrong trade when learning a language is the point; keep it as an ablation).
LR re-warm to ~1–3e-5 for 8B-class (higher for 4B), cosine decay to 10 %,
bf16, sequence length 4,096 with document packing, ~1 M tokens per batch,
checkpoint every 30 min. The stack is torchtitan or HF + FSDP;
`bantaywika/prepare_torchtitan.py` already produces torchtitan-ready text.

**Compute (estimates, from 6·N·D at 40 % MFU; replace with the step rate
`scripts/step_rate.sh` measures in the first hour):**

| job | hardware | ≈ per 1 B tokens | 1 B-token bake-off | 3 B-token main run |
|---|---|---|---|---|
| 4B dense | 1 × RTX PRO 6000 (96 GB), preemptible ~$1/h | ~70 h, ~$70 | ~$70 per base | ~$210 |
| 8B dense | 8 × H100 node (~$17/h) | ~4 h, ~$70 | — | ~$210 |
| gpt-oss-20b (3.6B active) | 1–2 × RTX PRO 6000 | ~60–90 h | ~$60–90 | ~$250 |

Memory check: 8B full-parameter AdamW needs ~16 bytes per parameter
(~128 GB) before activations, so it doesn't fit one 96 GB card. 4B (~64 GB)
does, with gradient checkpointing. Hence ≤4B for the bake-off on one card,
and 8B on a node only if the bake-off says 8B is worth it.

**Whole plan, all phases: roughly $300–700 of GPU time.** For comparison,
the human evaluation panel and a year of Tavily credits both cost more than
that. That ratio is the plan's argument for spending effort on data and
evaluation.

---

## 8. Phases and gates

| phase | work | gate to pass |
|---|---|---|
| **0 · harness** (week 1) | `eval_llm.py`, frozen held-out splits, decontamination, base scores for all candidates plus the three existing CPT models | every candidate scored on every metric |
| **1 · data** (weeks 1–3, overlaps) | finish the FineWeb-2 scrape for all 9; get a `TAVILY_API_KEY` and run the targeted scrape; ingest BalitaNLP; build the mix with α and epoch cap; decontaminate | measured token table replaces §1.2 estimates |
| **2 · bake-off** (weeks 3–4) | 3 bases × 1 B tokens on RTX PRO 6000 | winner beats its own base on bpb for ≥ 8 of 10 languages with ≤ 2 points English loss |
| **3 · ablations** (week 5) | vocab extension · Indonesian/Malay bridge · parallel data · LoRA vs full | each kept only if it wins per FLOP |
| **4 · main run** (week 6) | winner + kept ablations, all data at the capped epochs, 4B or 8B | beats phase-2 checkpoint; human review prefers it over the base |
| **5 · release** | model + eval table + data card + licence lineage on `sapinsapin` | card has numbers, not "More information needed" |
| **refresh** (ongoing) | when the scrape grows a language by ≥ 2×, re-run phase 4 from the last checkpoint | same gates |

---

## 9. Risks

- **Too little data for the small languages, whatever we do.** Tausug and
  Waray may gain little from CPT alone. Mitigation: report per-language
  results honestly, and route those languages' budget into the community
  data drive rather than more epochs.
- **Machine-translated and bot text** inflates Cebuano and poisons style. The
  domain filters catch the known farms, but not all of them. Mitigation:
  report the domain mix per language (the scrape manifest has it), and audit
  the top domains by hand before the main run.
- **Licence contamination** of the commercial track: PLD is CC-BY-NC,
  BalitaNLP's terms are unverified, and web pages vary. Mitigation: two
  mixes, built from row-level provenance, never merged.
- **Forgetting.** Gated on the English retention metric.
- **Eval leakage.** FLORES-derived sets (Belebele, SIB-200) share sentences
  with web text, so the 13-gram filter runs before every mix is frozen.

---

## 10. First actions

1. Get a `TAVILY_API_KEY` into `.env`: the targeted scrape is the only lever
   on the small languages, and it's blocked on the key.
2. Build `scripts/eval_llm.py` and score the bases plus the three existing
   CPT models (phase 0). This alone answers whether the earlier CPT runs
   helped.
3. Let the running FineWeb-2 scrapes (fil, pam) finish, then regenerate §1.2
   from measured counts instead of estimates.

---

## 11. External review, 2026-09-22

Reviewed by `deepseek-v4-pro` via the NYO connector (GLM 5.3 and Kimi K3 both
failed to answer). Its factual claims were checked rather than taken on trust,
and two were wrong — a useful reminder to verify a reviewer as carefully as
the plan.

**Accepted, and changed above or to be changed:**

1. **The LID gate throws away code-switched text**, which is the normal
   register in the Philippines. Requiring ≥0.6 sentence agreement rejects a
   page that is 70 % Filipino and 30 % English — 182 of the 278 Filipino
   rejections in the re-scrape were exactly this. *Change:* move to a
   language-*proportion* rule (keep if the target language is ≥50 % of words,
   ≥30 % for the small languages), record the proportion per row, and keep
   the mixed pages tagged rather than dropped.
2. **No morphological evaluation.** Added to phase 0 above.
3. **The 4-epoch cap is prescriptive where the paper is descriptive.**
   Muennighoff et al. report diminishing returns around four epochs in a
   compute-optimal regime with plenty of unique data; for Tausug (~0.3 M
   tokens) the binding question is overfitting, not epoch count. *Change:*
   per-language early stopping on validation bits per byte, with the cap as a
   default rather than a rule.
4. **α = 0.3 starves the smallest languages.** Recompute the mix from the
   measured token table and floor each language's share, rather than taking
   α from high-resource multilingual work.
5. **"Data-bound, not compute-bound" is asserted, not tested.** Add a cheap
   ablation: the same compute on an unfiltered FineWeb-2 dump versus the
   gated mix. If the gate doesn't win, the data engine isn't earning its
   keep.
6. **40 % MFU is optimistic** for single-card full-parameter training with
   deep gradient accumulation. Re-estimate from a measured step rate before
   committing; budget 30 %.

**Rejected, with evidence:**

- *"Belebele covers only Filipino; Cebuano, Ilocano and Waray are absent"*
  (stated at very high confidence) — **wrong**: Belebele's 122 configs
  include `ceb_Latn`, `ilo_Latn`, `tgl_Latn`, `war_Latn`.
- *"SIB-200 includes no Philippine language"* (high confidence) — **wrong**:
  its 205 configs include `ceb_Latn`, `ilo_Latn`, `pag_Latn`, `tgl_Latn`,
  `war_Latn`.

**Accepted as a caveat on our own claim:** HaloLID's win over GlotLID is
measured on PLD-domain text, which is HaloLID's training *domain* even though
the test sentences are held out and deduplicated. We have no human-labelled
*web* benchmark, so the honest statement is "better on PLD-domain text, and
unmeasured on web text against human labels". Building a small human-labelled
web set (say 100 sentences per language) is the fix, and it is cheap.
