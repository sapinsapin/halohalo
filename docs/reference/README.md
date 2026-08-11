# Reference — under-resourced language AI knowledge base

Working notes for the research behind halohalo and the Sapin-sapin initiative:
who else is solving these problems, what data and benchmarks exist, which
methods actually survive contact with a low-resource language, and what to
read.

This is a **knowledge base, not a bookmark list**. Every entry answers one
question: *what does this let us do that we could not do otherwise?* If an
entry has no honest answer to that, it does not belong here.

## Contents

| File | What's in it |
|---|---|
| [research-groups.md](research-groups.md) | Labs, orgs, and communities working the same problem — including the ones whose playbooks we should copy |
| [tokenizers.md](tokenizers.md) | **Measured:** pretrained tokenizers fit Philippine languages badly (1.75–1.92× English fertility). Morfessor assessment, and what vocabulary extension would buy |
| [datasets-benchmarks.md](datasets-benchmarks.md) | Corpora we can train on, benchmarks we must not train on, and where the Philippine gaps are |
| [methods-and-toolkits.md](methods-and-toolkits.md) | Techniques that work when you have hundreds of hours instead of hundreds of thousands, and the software that implements them |
| [reading-list.md](reading-list.md) | Papers worth reading properly, grouped by what they're for |

## Entry format

Keep entries short and honest. Each one gets:

- **What it is** — one or two lines, no marketing.
- **Why it matters to us** — the specific transfer to Philippine-language
  work. This is the part that makes the entry worth keeping.
- **Caveats** — licence, language coverage, staleness, or "we haven't
  verified this yet". Say so plainly; a confident-sounding wrong entry costs
  more than a missing one.

Mark anything unverified with **[unverified]** rather than deleting it —
a lead worth chasing is useful, a fact we've asserted without checking is not.

## Where this connects to the rest of the repo

- [`../sota_speech_plan.md`](../sota_speech_plan.md) — the cluster-scale TTS +
  S2S plan this research feeds.
- [`../orpheus_pilot_plan.md`](../orpheus_pilot_plan.md) — the single-GPU
  QLoRA pilot that tests the plan's assumptions cheaply.
- [`../livestream_pipeline.md`](../livestream_pipeline.md) — our data engine;
  several entries here are about making it better.
- [`../cards/`](../cards/) — the published dataset cards.
