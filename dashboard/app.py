"""halohalo dashboard — live view of the sapinsapin org on the Hugging Face Hub.

Every page load queries the Hub API, so the dashboard always reflects what is
actually in the org. Private repos are included in counts and listed as rows,
but their names are withheld. Two sources for the private entries:

  live      — if the HF_TOKEN Space secret (a read-scoped token) is set, the
              API itself returns the org's private repos.
  manifest  — otherwise, private_manifest.json (a name-free snapshot holding
              only counts per repo type) fills in the private rows, labeled
              with its as-of date.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import pandas as pd
from huggingface_hub import HfApi

ORG = os.environ.get("DASHBOARD_ORG", "sapinsapin")
TOKEN = os.environ.get("HF_TOKEN")

MASKED = "🔒 (private — name withheld)"
MANIFEST = Path(__file__).parent / "private_manifest.json"


def _manifest_counts():
    """{'dataset': n, 'model': n} from the name-free snapshot, or {}."""
    try:
        m = json.loads(MANIFEST.read_text())
        return m.get("as_of", "?"), m.get("private_counts", {})
    except (OSError, json.JSONDecodeError):
        return "?", {}


def _public_row(info, kind):
    url_prefix = {"dataset": "datasets/", "model": "", "space": "spaces/"}[kind]
    name = info.id.split("/", 1)[1]
    return {
        "Name": f"[{name}](https://huggingface.co/{url_prefix}{info.id})",
        "Visibility": "public",
        "Task": getattr(info, "pipeline_tag", None) or "—",
        "Downloads (30d)": getattr(info, "downloads", 0) or 0,
        "Likes": getattr(info, "likes", 0) or 0,
        "Updated": (info.last_modified.strftime("%Y-%m-%d")
                    if getattr(info, "last_modified", None) else "—"),
    }


def _private_row():
    return {"Name": MASKED, "Visibility": "private", "Task": "—",
            "Downloads (30d)": None, "Likes": None, "Updated": "—"}


def _table(items, kind, n_private, with_task=False):
    pub = sorted((i for i in items if not i.private),
                 key=lambda i: -(getattr(i, "downloads", 0) or 0))
    rows = [_public_row(i, kind) for i in pub]
    rows += [_private_row() for _ in range(n_private)]
    df = pd.DataFrame(rows, columns=["Name", "Visibility", "Task",
                                     "Downloads (30d)", "Likes", "Updated"])
    if not with_task:
        df = df.drop(columns=["Task"])
    return df


def fetch():
    api = HfApi(token=TOKEN)
    datasets = list(api.list_datasets(
        author=ORG, expand=["downloads", "likes", "lastModified", "private"]))
    models = list(api.list_models(
        author=ORG, expand=["downloads", "likes", "lastModified", "private",
                            "pipeline_tag"]))

    # private rows: live from the API when a token can see them, otherwise
    # from the name-free snapshot manifest
    n_priv = {"dataset": sum(i.private for i in datasets),
              "model": sum(i.private for i in models)}
    if any(n_priv.values()):
        priv_note = "Private repos are listed live, names withheld."
    else:
        as_of, counts_snapshot = _manifest_counts()
        n_priv = {"dataset": counts_snapshot.get("dataset", 0),
                  "model": counts_snapshot.get("model", 0)}
        priv_note = (f"Private repos are listed from a name-free snapshot "
                     f"(as of {as_of}), names withheld."
                     if any(n_priv.values()) else "")

    def counts(items, kind):
        n_pub = sum(not i.private for i in items)
        return f"{n_pub + n_priv[kind]} ({n_pub} public · {n_priv[kind]} private)"

    dl = sum((getattr(i, "downloads", 0) or 0) for i in datasets + models)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = (
        f"## 🍧 halohalo — the [{ORG}](https://huggingface.co/{ORG}) org, live\n"
        f"**Datasets:** {counts(datasets, 'dataset')} &nbsp;·&nbsp; "
        f"**Models:** {counts(models, 'model')} &nbsp;·&nbsp; "
        f"**Downloads (30d):** {dl:,}\n\n"
        f"Philippine-language corpora (speech, web text, literary text) and the "
        f"models finetuned on them. Queried from the Hub API at {now} — reload "
        f"or hit Refresh for the current state. {priv_note}"
    )
    return (header,
            _table(datasets, "dataset", n_priv["dataset"]),
            _table(models, "model", n_priv["model"], with_task=True))


DF_KW = dict(interactive=False, wrap=True)

with gr.Blocks(title="halohalo — org dashboard and speech demo") as demo:
    header_md = gr.Markdown("Loading…")
    with gr.Tab("📚 Datasets"):
        datasets_df = gr.Dataframe(
            datatype=["markdown", "str", "number", "number", "str"], **DF_KW)
    with gr.Tab("🤖 Models"):
        models_df = gr.Dataframe(
            datatype=["markdown", "str", "str", "number", "number", "str"], **DF_KW)
        refresh = gr.Button("🔄 Refresh", size="sm")

    # The speech tabs are additive: everything heavy in them is imported
    # lazily, so if the ML stack is unavailable the dashboard above still
    # works and only these tabs report the problem.
    try:
        import speech_demo
        speech_demo.build_tabs()
        demo_ok = True
    except Exception as exc:                                  # noqa: BLE001
        demo_ok = False
        with gr.Tab("🎙️ Speech demo"):
            gr.Markdown(f"Speech tabs unavailable: `{type(exc).__name__}: {exc}`")

    if demo_ok:
        gr.Markdown(
            "---\n"
            "The speech tabs run the org's own models: **10 ASR + 10 TTS + 1 "
            "voice conversion**, finetuned on the Philippine Language Dataset "
            "for Bikol, Cebuano, Filipino, Hiligaynon, Ilocano, Kapampangan, "
            "Pangasinan, Tausug, Waray and Philippine English. They are "
            "baselines trained on prompted read speech — accuracy drops on "
            "spontaneous or noisy audio. Preloaded clips and voice presets "
            "come from the corpus collected by the **UP Diliman Digital "
            "Signal Processing Laboratory**. "
            "[Code](https://github.com/sapinsapin/halohalo)")

    outputs = [header_md, datasets_df, models_df]
    demo.load(fetch, inputs=None, outputs=outputs)
    refresh.click(fetch, inputs=None, outputs=outputs)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
