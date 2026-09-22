"""
Language identification for the ten PLD languages.

Two models behind one interface:

  GlotLID  — cis-lmu/glotlid (fastText, ~1.7 GB, 2,000+ labels). The strongest
             open LID for low-resource languages and the model FineWeb-2 was
             filtered with. Baseline, and the second opinion in the ensemble.
  HaloLID  — our own fastText supervised model trained on PLD, halohalo and
             BantayWika text by scripts/train_lid.py. Small (quantised, tens of
             MB), fast, and specialised: it only has to separate ten closely
             related Philippine languages from each other, English, and a
             handful of confusable neighbours (Indonesian/Malay).

Label space is the PLD code set. GlotLID's `tgl_Latn`/`fil_Latn` both map to
`fil` because PLD does not distinguish them. Anything outside the set is
`other` — a real label, so the models can abstain instead of guessing.

Document-level identification votes over sentences weighted by length, which
matters for web pages: a Cebuano article with an English nav bar should still
come out Cebuano, and a Cebuano page with an English body should not.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

LANGS = ("bcl", "ceb", "eng", "fil", "hil", "ilo", "pag", "pam", "tsg", "war")
OTHER = "other"

# GlotLID label -> PLD code. Everything else -> OTHER.
GLOTLID_MAP = {
    "bcl_Latn": "bcl", "ceb_Latn": "ceb", "eng_Latn": "eng",
    "tgl_Latn": "fil", "fil_Latn": "fil", "hil_Latn": "hil",
    "ilo_Latn": "ilo", "pag_Latn": "pag", "pam_Latn": "pam",
    "tsg_Latn": "tsg", "war_Latn": "war",
}

_WS = re.compile(r"\s+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_DIGITS_PUNCT = re.compile(r"[\d_]+|[^\w\s'\-]")


def normalize(text: str) -> str:
    """fastText input: one line, lowercase, no digits or stray punctuation."""
    t = _DIGITS_PUNCT.sub(" ", text.lower())
    return _WS.sub(" ", t).strip()


def split_sentences(text: str, min_chars: int = 15) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) >= min_chars]


@dataclass(frozen=True)
class Prediction:
    lang: str        # PLD code or "other"
    score: float     # model probability for that label
    raw: str         # the model's own label, for debugging

    @property
    def known(self) -> bool:
        return self.lang != OTHER


class _FastTextLID:
    """Shared wrapper: subclasses only differ in where the model comes from and
    how its labels map onto PLD codes."""

    name = "fasttext"

    def __init__(self, model_path: str | Path):
        import fasttext
        fasttext.FastText.eprint = lambda *_: None      # silence load warning
        self._model = fasttext.load_model(str(model_path))

    def _map(self, label: str) -> str:
        raise NotImplementedError

    def predict(self, text: str, k: int = 3) -> Prediction:
        t = normalize(text)
        if not t:
            return Prediction(OTHER, 0.0, "")
        labels, probs = self._model.predict(t, k=k)
        best = labels[0].replace("__label__", "")
        return Prediction(self._map(best), float(probs[0]), best)

    def predict_document(self, text: str, min_chars: int = 15) -> tuple[Prediction, float]:
        """Length-weighted sentence vote.

        Returns the winning prediction and the *agreement*: the fraction of
        characters that voted for the winner. Low agreement means a mixed
        page (boilerplate in another language, or code-switching), which is
        worth knowing separately from the winner's confidence."""
        sents = split_sentences(text, min_chars) or [text]
        weight: dict[str, float] = {}
        conf: dict[str, float] = {}
        total = 0.0
        for s in sents:
            p = self.predict(s)
            w = len(s)
            total += w
            weight[p.lang] = weight.get(p.lang, 0.0) + w
            conf[p.lang] = conf.get(p.lang, 0.0) + w * p.score
        lang = max(weight, key=weight.get)
        agreement = weight[lang] / total if total else 0.0
        score = conf[lang] / weight[lang] if weight[lang] else 0.0
        return Prediction(lang, score, lang), agreement


class GlotLID(_FastTextLID):
    name = "glotlid"
    REPO = "cis-lmu/glotlid"
    FILE = "model_v3.bin"

    def __init__(self, model_path: str | Path | None = None):
        if model_path is None:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(self.REPO, self.FILE,
                                         token=os.environ.get("HF_TOKEN"))
        super().__init__(model_path)

    def _map(self, label: str) -> str:
        return GLOTLID_MAP.get(label, OTHER)


class HaloLID(_FastTextLID):
    name = "halolid"
    REPO = "sapinsapin/halo-lid"
    FILE = "model.ftz"

    def __init__(self, model_path: str | Path | None = None):
        if model_path is None:
            local = Path(os.environ.get("FINETUNE_DIR", "finetune_runs")) / "lid" / self.FILE
            if local.exists():
                model_path = local
            else:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(self.REPO, self.FILE,
                                             token=os.environ.get("HF_TOKEN"))
        super().__init__(model_path)

    def _map(self, label: str) -> str:
        return label if label in LANGS else OTHER


@dataclass(frozen=True)
class Verdict:
    lang: str
    score: float          # confidence of the deciding model
    agreement: float      # sentence-level agreement of the deciding model
    models_agree: bool    # did GlotLID and HaloLID pick the same language?
    detail: dict

    def accept(self, want: str, min_score: float = 0.6, min_agreement: float = 0.6) -> bool:
        return (self.lang == want and self.score >= min_score
                and self.agreement >= min_agreement)


class Ensemble:
    """HaloLID decides; GlotLID is the second opinion. Disagreement is
    recorded, not resolved by fiat: a document the two models disagree on is
    accepted only if the decider is confident, and the disagreement is kept in
    the row so it can be audited or used to select hard examples for the next
    round of LID training."""

    def __init__(self, halo: HaloLID | None = None, glot: GlotLID | None = None):
        self.halo = halo
        self.glot = glot
        if halo is None and glot is None:
            raise ValueError("need at least one model")

    def identify(self, text: str) -> Verdict:
        detail = {}
        h = g = None
        if self.halo:
            h, ha = self.halo.predict_document(text)
            detail["halolid"] = {"lang": h.lang, "score": round(h.score, 4), "agreement": round(ha, 3)}
        if self.glot:
            g, ga = self.glot.predict_document(text)
            detail["glotlid"] = {"lang": g.lang, "score": round(g.score, 4), "agreement": round(ga, 3)}

        decider, agree = (h, detail["halolid"]["agreement"]) if h else (g, detail["glotlid"]["agreement"])
        models_agree = (h is None or g is None) or (h.lang == g.lang)
        score = decider.score if models_agree else decider.score * 0.8   # penalise disputed calls
        return Verdict(decider.lang, score, agree, models_agree, detail)


@lru_cache(maxsize=1)
def default_ensemble(use_glotlid: bool = True) -> Ensemble:
    """HaloLID if it exists locally or on the Hub, GlotLID as second opinion.
    Falls back to GlotLID alone before any HaloLID has been trained."""
    halo = None
    try:
        halo = HaloLID()
    except Exception:
        pass
    glot = GlotLID() if (use_glotlid or halo is None) else None
    return Ensemble(halo, glot)
