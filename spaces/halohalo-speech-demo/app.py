"""
halohalo speech demo — run the PLD-finetuned speech models for ten Philippine
languages from the browser.

Three tabs, one per model family:
  Transcribe        whisper-small-pld-<lang>   (record, upload, or preloaded)
  Synthesize        speecht5_tts-pld-<lang>
  Convert voice     speecht5_vc-pld            (any-to-any, all ten languages)

Design notes for a free CPU Space:
  - Models load lazily on first use and stay cached; loading all 21 up front
    would blow both the startup timeout and the RAM budget.
  - Speaker embeddings are precomputed (see prepare_space_assets.py), so
    speechbrain is not a runtime dependency here.
"""

import json
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import torch

ROOT = Path(__file__).parent
ASSETS = json.loads((ROOT / "assets.json").read_text(encoding="utf-8"))
ORG = "sapinsapin"
SR = 16000
VC_MAXLENRATIO = 1.3       # ≈1.0x input duration; see convert()

LANGS = ASSETS["languages"]
NAME_TO_CODE = {v["name"]: k for k, v in LANGS.items()}
CHOICES = [v["name"] for v in LANGS.values()]

torch.set_num_threads(4)


# ---------------------------------------------------------------- model cache

@lru_cache(maxsize=3)
def asr_model(lang: str):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    repo = f"{ORG}/whisper-small-pld-{lang}"
    proc = WhisperProcessor.from_pretrained(repo)
    model = WhisperForConditionalGeneration.from_pretrained(repo).eval()
    return proc, model


@lru_cache(maxsize=3)
def tts_model(lang: str):
    from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor
    repo = f"{ORG}/speecht5_tts-pld-{lang}"
    return (SpeechT5Processor.from_pretrained(repo),
            SpeechT5ForTextToSpeech.from_pretrained(repo).eval())


@lru_cache(maxsize=1)
def vc_model():
    from transformers import SpeechT5ForSpeechToSpeech, SpeechT5Processor
    repo = f"{ORG}/speecht5_vc-pld"
    return (SpeechT5Processor.from_pretrained(repo),
            SpeechT5ForSpeechToSpeech.from_pretrained(repo).eval())


@lru_cache(maxsize=1)
def vocoder():
    from transformers import SpeechT5HifiGan
    return SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").eval()


def load_xvector(rel: str) -> torch.Tensor:
    return torch.tensor(np.load(ROOT / rel)).unsqueeze(0)


def as_mono16k(audio):
    """Gradio hands back (sample_rate, np.int16|float array) from mic/upload."""
    if audio is None:
        return None
    sr, data = audio
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    peak = np.abs(data).max()
    if peak > 1.0:                       # int16 range → [-1, 1]
        data = data / 32768.0
    elif peak > 0:
        data = data / max(peak, 1e-9) * 0.95
    if sr != SR:                         # linear resample keeps scipy out of it
        n = int(round(len(data) * SR / sr))
        data = np.interp(np.linspace(0, len(data), n, endpoint=False),
                         np.arange(len(data)), data).astype(np.float32)
    return data


# -------------------------------------------------------------------- actions

def transcribe(lang_name, audio):
    if audio is None:
        return "Record, upload, or pick a sample clip first."
    lang = NAME_TO_CODE[lang_name]
    wav = as_mono16k(audio)
    if wav is None or len(wav) < SR * 0.2:
        return "That clip is too short to transcribe."
    proc, model = asr_model(lang)
    feats = proc.feature_extractor(wav, sampling_rate=SR,
                                   return_tensors="pt").input_features
    with torch.no_grad():
        ids = model.generate(feats, max_new_tokens=200)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def synthesize(lang_name, text, voice_label):
    if not (text or "").strip():
        return None, "Type something to synthesize."
    lang = NAME_TO_CODE[lang_name]
    voices = LANGS[lang]["voices"]
    if not voices:
        return None, f"No voice preset for {lang_name}."
    pick = next((v for v in voices if voice_label and v["id"] in voice_label),
                voices[0])
    proc, model = tts_model(lang)
    inputs = proc(text=text.strip(), return_tensors="pt")
    with torch.no_grad():
        speech = model.generate_speech(inputs["input_ids"],
                                       load_xvector(pick["file"]),
                                       vocoder=vocoder())
    return (SR, speech.numpy()), f"{lang_name} · voice {pick['id']}"


def convert(audio, voice_label):
    if audio is None:
        return None, "Record or upload the audio you want converted."
    wav = as_mono16k(audio)
    if wav is None or len(wav) < SR * 0.2:
        return None, "That clip is too short to convert."
    if len(wav) > SR * 12:                # keep CPU latency sane
        wav = wav[:SR * 12]
    target = VOICE_INDEX.get(voice_label)
    if target is None:
        return None, "Pick a target voice."
    proc, model = vc_model()
    inputs = proc(audio=wav, sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        # The stop token is unreliable on this checkpoint — left to its own
        # devices the decoder runs to the 20x default and babbles well past
        # the end of the utterance. Conversion preserves timing, so bounding
        # the output near the input duration is both safe and correct.
        speech = model.generate_speech(inputs["input_values"],
                                       load_xvector(target["file"]),
                                       vocoder=vocoder(),
                                       maxlenratio=VC_MAXLENRATIO)
    return (SR, speech.numpy()), f"Converted to voice {target['id']}"


# Flat index of every preset voice across languages, for the conversion tab.
VOICE_INDEX = {}
for code, meta in LANGS.items():
    for v in meta["voices"]:
        VOICE_INDEX[f"{meta['name']} · {v['id']} ({v['gender']})"] = v
VOICE_LABELS = list(VOICE_INDEX)


def sample_choices(lang_name):
    lang = NAME_TO_CODE[lang_name]
    return [f"{i + 1}. {c['transcript'][:60]}"
            for i, c in enumerate(LANGS[lang]["clips"])]


def load_sample(lang_name, label):
    """Return the chosen preloaded clip as (sr, data) plus its reference text."""
    import soundfile as sf
    lang = NAME_TO_CODE[lang_name]
    clips = LANGS[lang]["clips"]
    if not clips:
        return None, ""
    idx = 0
    if label and label[0].isdigit():
        idx = min(int(label.split(".")[0]) - 1, len(clips) - 1)
    clip = clips[idx]
    data, sr = sf.read(ROOT / clip["file"], dtype="float32")
    return (sr, data), clip["transcript"]


def voices_for(lang_name):
    lang = NAME_TO_CODE[lang_name]
    return [f"{v['id']} ({v['gender']})" for v in LANGS[lang]["voices"]]


# ----------------------------------------------------------------------- ui

CSS = """
.hh-hero{text-align:center;padding:8px 0 2px}
.hh-hero h1{margin:0;font-size:1.9rem}
.hh-hero p{margin:6px 0 0;opacity:.75}
footer{visibility:hidden}
"""

with gr.Blocks(title="halohalo — Philippine speech models") as demo:
    gr.HTML(
        "<div class='hh-hero'><h1>halohalo speech models</h1>"
        "<p>Speech recognition, synthesis and voice conversion for "
        "<b>ten Philippine languages</b> — Bikol, Cebuano, Filipino, "
        "Hiligaynon, Ilocano, Kapampangan, Pangasinan, Tausug, Waray "
        "and Philippine English.</p></div>")

    with gr.Tab("🎙️ Transcribe"):
        gr.Markdown(
            "Record yourself, upload a file, or load one of the preloaded "
            "clips. Each language uses its own finetuned Whisper model. "
            "*First run downloads the model — expect a slower first go.*")
        with gr.Row():
            with gr.Column():
                a_lang = gr.Dropdown(CHOICES, value="Cebuano", label="Language")
                a_sample = gr.Dropdown(sample_choices("Cebuano"),
                                       label="Preloaded clip (optional)")
                a_load = gr.Button("Load preloaded clip")
                a_audio = gr.Audio(sources=["microphone", "upload"],
                                   type="numpy", label="Audio")
            with gr.Column():
                a_ref = gr.Textbox(label="Reference transcript (preloaded clip)",
                                   interactive=False)
                a_out = gr.Textbox(label="Model transcription", lines=4)
                a_go = gr.Button("Transcribe", variant="primary")

        a_lang.change(lambda l: gr.update(choices=sample_choices(l), value=None),
                      a_lang, a_sample)
        a_load.click(load_sample, [a_lang, a_sample], [a_audio, a_ref])
        a_go.click(transcribe, [a_lang, a_audio], a_out)

    with gr.Tab("🔊 Synthesize"):
        gr.Markdown(
            "Type text in the chosen language and hear it in a voice from that "
            "language's speakers. Numbers are best written as words — the "
            "training text contains no verbalized numerals.")
        with gr.Row():
            with gr.Column():
                t_lang = gr.Dropdown(CHOICES, value="Cebuano", label="Language")
                t_voice = gr.Dropdown(voices_for("Cebuano"),
                                      value=(voices_for("Cebuano") or [None])[0],
                                      label="Voice")
                t_text = gr.Textbox(label="Text", lines=3,
                                    value="Maayong buntag sa tanan.")
                t_go = gr.Button("Synthesize", variant="primary")
            with gr.Column():
                t_audio = gr.Audio(label="Synthesized speech")
                t_note = gr.Textbox(label="Details", interactive=False)

        t_lang.change(
            lambda l: gr.update(choices=voices_for(l),
                                value=(voices_for(l) or [None])[0]),
            t_lang, t_voice)
        t_go.click(synthesize, [t_lang, t_text, t_voice], [t_audio, t_note])

    with gr.Tab("🎭 Convert voice"):
        gr.Markdown(
            "Say something and hear it in another speaker's voice. One model "
            "handles all ten languages: it was trained on pairs of different "
            "speakers reading the *same* sentence, which the corpus provides "
            "because its prompt lists are shared across speakers.")
        with gr.Row():
            with gr.Column():
                v_audio = gr.Audio(sources=["microphone", "upload"],
                                   type="numpy", label="Your audio (any language)")
                v_target = gr.Dropdown(VOICE_LABELS,
                                       value=VOICE_LABELS[0] if VOICE_LABELS else None,
                                       label="Target voice")
                v_go = gr.Button("Convert", variant="primary")
            with gr.Column():
                v_out = gr.Audio(label="Converted speech")
                v_note = gr.Textbox(label="Details", interactive=False)
        v_go.click(convert, [v_audio, v_target], [v_out, v_note])

    gr.Markdown(
        "---\n"
        "**Models** · [all 21 on the Hub](https://huggingface.co/sapinsapin) — "
        "10 ASR + 10 TTS + 1 voice conversion, each finetuned on the "
        "Philippine Language Dataset. "
        "**Code** · [github.com/sapinsapin/halohalo](https://github.com/sapinsapin/halohalo) · "
        "**Org dashboard** · [halohalo-dashboard](https://huggingface.co/spaces/sapinsapin/halohalo-dashboard)\n\n"
        "Preloaded clips and voice presets come from the Philippine Language "
        "Dataset, collected by the **UP Diliman Digital Signal Processing "
        "Laboratory**. They are included here purely to demonstrate the "
        "models. These are baselines trained on prompted read speech — "
        "accuracy drops on spontaneous or noisy audio.")

if __name__ == "__main__":
    # theme/css belong to launch() from Gradio 6 onwards
    demo.queue(max_size=12).launch(css=CSS, theme=gr.themes.Soft())
