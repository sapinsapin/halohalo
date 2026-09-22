"""
Interactive tabs for the halohalo speech models, mounted inside the org
dashboard Space.

Three tabs, one per model family:
  Transcribe        whisper-small-pld-<lang>   (record, upload, or preloaded)
  Synthesize        speecht5_tts-pld-<lang>
  Convert voice     speecht5_vc-pld            (any-to-any, all ten languages)

Two constraints shape this file:

  - It shares a free CPU Space with the dashboard, so torch and transformers
    are imported *inside* the functions that need them. A missing or broken
    ML dependency then degrades these tabs rather than taking the dashboard
    down with it, and the Space still boots in seconds.
  - Models load lazily on first use and stay cached; loading all 21 up front
    would exhaust both the startup timeout and the RAM budget.

Speaker embeddings are precomputed (see prepare_space_assets.py in the
halohalo repo), so speechbrain is not a runtime dependency here.
"""

import json
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np

ROOT = Path(__file__).parent
ORG = "sapinsapin"
SR = 16000
VC_MAXLENRATIO = 1.3       # ≈1.0x input duration; see convert()

_assets_path = ROOT / "assets.json"
ASSETS = (json.loads(_assets_path.read_text(encoding="utf-8"))
          if _assets_path.exists() else {"languages": {}})
LANGS = ASSETS["languages"]
NAME_TO_CODE = {v["name"]: k for k, v in LANGS.items()}
CHOICES = [v["name"] for v in LANGS.values()]
DEFAULT = "Cebuano" if "Cebuano" in CHOICES else (CHOICES[0] if CHOICES else None)

VOICE_INDEX = {}
for _code, _meta in LANGS.items():
    for _v in _meta["voices"]:
        # carry the language so a conversion can be filed under it later
        VOICE_INDEX[f"{_meta['name']} · {_v['id']} ({_v['gender']})"] = {
            **_v, "language": _code}
VOICE_LABELS = list(VOICE_INDEX)


# ---------------------------------------------------------------- model cache

@lru_cache(maxsize=1)
def _torch():
    import torch
    torch.set_num_threads(4)
    return torch


@lru_cache(maxsize=2)
def asr_model(lang: str):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    repo = f"{ORG}/whisper-small-pld-{lang}"
    return (WhisperProcessor.from_pretrained(repo),
            WhisperForConditionalGeneration.from_pretrained(repo).eval())


@lru_cache(maxsize=2)
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


def load_xvector(rel: str):
    return _torch().tensor(np.load(ROOT / rel)).unsqueeze(0)


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

def transcribe(lang_name, audio, reference=""):
    """Returns (text, rating context). The context is what feedback.record
    needs to reconstruct this example later."""
    if audio is None:
        return "Record, upload, or load a preloaded clip first.", None
    wav = as_mono16k(audio)
    if wav is None or len(wav) < SR * 0.2:
        return "That clip is too short to transcribe.", None
    lang = NAME_TO_CODE[lang_name]
    try:
        torch = _torch()
        proc, model = asr_model(lang)
        feats = proc.feature_extractor(wav, sampling_rate=SR,
                                       return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(feats, max_new_tokens=200)
        text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    except Exception as e:                # a failed tab must not 500 the Space
        return f"Could not transcribe: {type(e).__name__}: {e}", None

    ctx = {"task": "asr", "language": lang,
           "model": f"{ORG}/whisper-small-pld-{lang}",
           "output_text": text, "reference_text": (reference or "").strip() or None,
           "input_audio": (SR, wav)}
    return text, ctx


def synthesize(lang_name, text, voice_label):
    if not (text or "").strip():
        return None, "Type something to synthesize.", None
    lang = NAME_TO_CODE[lang_name]
    voices = LANGS[lang]["voices"]
    if not voices:
        return None, f"No voice preset for {lang_name}.", None
    pick = next((v for v in voices if voice_label and v["id"] in voice_label),
                voices[0])
    try:
        torch = _torch()
        proc, model = tts_model(lang)
        inputs = proc(text=text.strip(), return_tensors="pt")
        with torch.no_grad():
            speech = model.generate_speech(inputs["input_ids"],
                                           load_xvector(pick["file"]),
                                           vocoder=vocoder())
        out = (SR, speech.numpy())
    except Exception as e:
        return None, f"Could not synthesize: {type(e).__name__}: {e}", None

    ctx = {"task": "tts", "language": lang,
           "model": f"{ORG}/speecht5_tts-pld-{lang}", "voice": pick["id"],
           "input_text": text.strip(), "output_audio": out}
    return out, f"{lang_name} · voice {pick['id']}", ctx


def convert(audio, voice_label):
    if audio is None:
        return None, "Record or upload the audio you want converted.", None
    wav = as_mono16k(audio)
    if wav is None or len(wav) < SR * 0.2:
        return None, "That clip is too short to convert.", None
    if len(wav) > SR * 12:                # keep CPU latency sane
        wav = wav[:SR * 12]
    target = VOICE_INDEX.get(voice_label)
    if target is None:
        return None, "Pick a target voice.", None
    try:
        torch = _torch()
        proc, model = vc_model()
        inputs = proc(audio=wav, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            # The stop token is unreliable on this checkpoint — left alone the
            # decoder runs to the 20x default and babbles well past the end of
            # the utterance. Conversion preserves timing, so bounding output
            # near the input duration is both safe and correct.
            speech = model.generate_speech(inputs["input_values"],
                                           load_xvector(target["file"]),
                                           vocoder=vocoder(),
                                           maxlenratio=VC_MAXLENRATIO)
        out = (SR, speech.numpy())
    except Exception as e:
        return None, f"Could not convert: {type(e).__name__}: {e}", None

    ctx = {"task": "s2s", "model": f"{ORG}/speecht5_vc-pld",
           "language": target.get("language"),   # of the target voice
           "voice": target["id"], "input_audio": (SR, wav),
           "output_audio": out}
    return out, f"Converted to voice {target['id']}", ctx


# ------------------------------------------------------------------- helpers

def sample_choices(lang_name):
    if lang_name not in NAME_TO_CODE:
        return []
    clips = LANGS[NAME_TO_CODE[lang_name]]["clips"]
    return [f"{i + 1}. {c['transcript'][:60]}" for i, c in enumerate(clips)]


def load_sample(lang_name, label):
    """Return the chosen preloaded clip as (sr, data) plus its reference text."""
    import soundfile as sf
    if lang_name not in NAME_TO_CODE:
        return None, ""
    clips = LANGS[NAME_TO_CODE[lang_name]]["clips"]
    if not clips:
        return None, ""
    idx = 0
    if label and label[0].isdigit():
        idx = min(int(label.split(".")[0]) - 1, len(clips) - 1)
    clip = clips[idx]
    data, sr = sf.read(ROOT / clip["file"], dtype="float32")
    return (sr, data), clip["transcript"]


def voices_for(lang_name):
    if lang_name not in NAME_TO_CODE:
        return []
    return [f"{v['id']} ({v['gender']})"
            for v in LANGS[NAME_TO_CODE[lang_name]]["voices"]]


def _feedback_block(what: str):
    """Rate-this-output controls, returned so the caller can wire the state.

    Ratings only leave the browser when someone clicks, and the audio
    checkbox is shown next to the buttons rather than buried in a policy
    paragraph — people are sending us their own voice recordings.
    """
    import feedback

    with gr.Accordion(f"Was this {what} any good?", open=False):
        if feedback.enabled():
            gr.Markdown(
                f"Ratings train the next round of models. They are stored in a "
                f"**private** dataset (`{feedback.DETAIL}`) together with the "
                f"text, and with the audio if you leave the box ticked.")
        else:
            gr.Markdown(
                f"⚠️ Ratings are **not being saved** — {feedback.DETAIL}. The "
                f"buttons still work so you can see the flow.")
        comment = gr.Textbox(label="Anything to add? (optional)",
                             placeholder="e.g. wrong word, robotic prosody, "
                                         "clipped ending",
                             lines=1)
        keep = gr.Checkbox(value=True, label="Include the audio")
        with gr.Row():
            good = gr.Button("👍 Good", size="sm")
            bad = gr.Button("👎 Needs work", size="sm")
        status = gr.Markdown("")
    return comment, keep, good, bad, status


def _wire_feedback(state, comment, keep, good, bad, status):
    import feedback

    def rate(kind):
        def _fn(ctx, note, keep_audio):
            return feedback.record(kind, ctx, note, keep_audio)
        return _fn

    good.click(rate("good"), [state, comment, keep], status)
    bad.click(rate("bad"), [state, comment, keep], status)


# ----------------------------------------------------------------------- ui

def build_tabs():
    """Add the three demo tabs to the enclosing gr.Blocks context."""
    if not LANGS:
        with gr.Tab("🎙️ Speech demo"):
            gr.Markdown("Demo assets are missing from this Space "
                        "(`assets.json`), so the speech tabs are unavailable.")
        return

    # API-only: lets the Space's feedback wiring be checked from outside
    # without clicking through the UI, since a misnamed secret fails silently.
    import feedback as _fb
    _status_btn = gr.Button("feedback status", visible=False)
    _status_out = gr.Textbox(visible=False)
    _status_btn.click(_fb.status_report, None, _status_out,
                      api_name="feedback_status")

    with gr.Tab("🎙️ Transcribe"):
        gr.Markdown(
            "Record yourself, upload a file, or load a preloaded clip. Each "
            "language uses its own finetuned Whisper model. **The first run "
            "per language downloads a ~1GB model — give it a minute.**")
        with gr.Row():
            with gr.Column():
                a_lang = gr.Dropdown(CHOICES, value=DEFAULT, label="Language")
                a_sample = gr.Dropdown(sample_choices(DEFAULT),
                                       label="Preloaded clip (optional)")
                a_load = gr.Button("Load preloaded clip")
                a_audio = gr.Audio(sources=["microphone", "upload"],
                                   type="numpy", label="Audio")
            with gr.Column():
                a_ref = gr.Textbox(label="Reference transcript (preloaded clip)",
                                   interactive=False)
                a_out = gr.Textbox(label="Model transcription", lines=4)
                a_go = gr.Button("Transcribe", variant="primary")
                a_state = gr.State(None)
                a_fb = _feedback_block("transcription")

        a_lang.change(lambda l: gr.update(choices=sample_choices(l), value=None),
                      a_lang, a_sample)
        a_load.click(load_sample, [a_lang, a_sample], [a_audio, a_ref])
        a_go.click(transcribe, [a_lang, a_audio, a_ref], [a_out, a_state])
        _wire_feedback(a_state, *a_fb)

    with gr.Tab("🔊 Synthesize"):
        gr.Markdown(
            "Type text in the chosen language and hear it spoken by one of "
            "that language's speakers. Write numbers as words — the training "
            "text contains no verbalized numerals.")
        with gr.Row():
            with gr.Column():
                t_lang = gr.Dropdown(CHOICES, value=DEFAULT, label="Language")
                _v = voices_for(DEFAULT)
                t_voice = gr.Dropdown(_v, value=_v[0] if _v else None,
                                      label="Voice")
                t_text = gr.Textbox(label="Text", lines=3,
                                    value="Maayong buntag sa tanan.")
                t_go = gr.Button("Synthesize", variant="primary")
            with gr.Column():
                t_audio = gr.Audio(label="Synthesized speech")
                t_note = gr.Textbox(label="Details", interactive=False)
                t_state = gr.State(None)
                t_fb = _feedback_block("synthesis")

        t_lang.change(
            lambda l: gr.update(choices=voices_for(l),
                                value=(voices_for(l) or [None])[0]),
            t_lang, t_voice)
        t_go.click(synthesize, [t_lang, t_text, t_voice],
                   [t_audio, t_note, t_state])
        _wire_feedback(t_state, *t_fb)

    with gr.Tab("🎭 Convert voice"):
        gr.Markdown(
            "Say something and hear it in another speaker's voice. One model "
            "covers all ten languages: it was trained on pairs of different "
            "speakers reading the *same* sentence, which the corpus provides "
            "because its prompt lists are shared across speakers.")
        with gr.Row():
            with gr.Column():
                v_audio = gr.Audio(sources=["microphone", "upload"],
                                   type="numpy",
                                   label="Your audio (any language)")
                v_target = gr.Dropdown(
                    VOICE_LABELS,
                    value=VOICE_LABELS[0] if VOICE_LABELS else None,
                    label="Target voice")
                v_go = gr.Button("Convert", variant="primary")
            with gr.Column():
                v_out = gr.Audio(label="Converted speech")
                v_note = gr.Textbox(label="Details", interactive=False)
                v_state = gr.State(None)
                v_fb = _feedback_block("conversion")
        v_go.click(convert, [v_audio, v_target], [v_out, v_note, v_state])
        _wire_feedback(v_state, *v_fb)
