"""
The Orpheus token layout, text frontend and streaming window arithmetic,
kept free of torch so it can be tested anywhere and so the two generator
backends (transformers, llama.cpp) cannot disagree about it.

Everything here mirrors finetune_orpheus.py at the repo root, which is what
trained the sapinsapin/orpheus-3b-0.1-pretrained-char-pld-* adapters:

    [SOH] text_ids [EOT] [EOH] [SOAI] [SOS] audio_tokens [EOS] [EOAI]

with text = "{speaker_id}: {frontend_text(sentence, 'char')}" and audio
tokens = SNAC codes flattened seven to a frame.
"""

import re

SOH, EOT, EOH = 128259, 128009, 128260      # start/end of the "human" turn
SOAI, SOS = 128261, 128257                  # start of AI turn, start of speech
EOS_SPEECH, EOAI = 128258, 128262           # end of speech, end of AI turn
PAD = 128263

AUDIO_BASE = 128266
CODEBOOK_STRIDE = 4096
TOKENS_PER_FRAME = 7
# SNAC 24 kHz: hop 512, strides (4, 2, 1), so one coarse frame is 2048 samples.
SNAC_SR = 24000
SAMPLES_PER_FRAME = 2048
# The streaming decoder works on four frames at a time (Canopy's decoder.py).
WINDOW_FRAMES = 4

# llama.cpp streams the tokenizer's own strings for these ids.
CUSTOM_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")
CUSTOM_TOKEN_OFFSET = 128256          # <custom_token_N> has id 128256 + N

DELIM = "|"
PUNCT_RE = re.compile(r"[^\w\s']")
DIGIT_RE = re.compile(r"\d")


def frontend_char(text: str) -> str:
    """The `char` frontend the adapters were trained on: lower-cased, no
    punctuation, every character a token separated by spaces, words separated
    by " | ". Must match finetune_orpheus.frontend_text(text, "char")."""
    out = []
    for i, w in enumerate(PUNCT_RE.sub("", text.lower()).split()):
        if i:
            out.append(DELIM)
        out.extend(list(w))
    return " ".join(out)


def normalise_text(text: str) -> str:
    """Typographic apostrophes to ASCII, as the trainer's clean_text did."""
    return text.replace("’", "'").replace("‘", "'").strip()


def has_digits(text: str) -> bool:
    """The training text had no digits (clean_text rejected them), so digits
    are read as nothing or as noise. Surfaced, never corrected."""
    return bool(DIGIT_RE.search(text))


def prompt_text(voice_id: str, text: str) -> str:
    return f"{voice_id}: {frontend_char(normalise_text(text))}"


def wrap_prompt(text_ids: list[int]) -> list[int]:
    return [SOH] + list(text_ids) + [EOT, EOH, SOAI, SOS]


def is_audio_token(token_id: int) -> bool:
    return AUDIO_BASE <= token_id < AUDIO_BASE + TOKENS_PER_FRAME * CODEBOOK_STRIDE


def custom_token_id(piece: str) -> int | None:
    """<custom_token_N> -> token id, or None for any other text piece."""
    m = CUSTOM_TOKEN_RE.fullmatch(piece.strip())
    return CUSTOM_TOKEN_OFFSET + int(m.group(1)) if m else None


def codes_from_tokens(tokens: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Flat audio token ids -> the three SNAC codebooks (inverse of the
    trainer's encode layout). Whole frames only; a sampled token that landed
    outside its slot's band is wrapped into range so SNAC never indexes out
    of bounds."""
    tokens = [t for t in tokens if is_audio_token(t)]
    n_frames = len(tokens) // TOKENS_PER_FRAME
    c0, c1, c2 = [], [], []
    for j in range(n_frames):
        f = tokens[TOKENS_PER_FRAME * j:TOKENS_PER_FRAME * (j + 1)]
        v = [(t - AUDIO_BASE - i * CODEBOOK_STRIDE) % CODEBOOK_STRIDE
             for i, t in enumerate(f)]
        c0.append(v[0])
        c1 += [v[1], v[4]]
        c2 += [v[2], v[3], v[5], v[6]]
    return c0, c1, c2


class FrameWindow:
    """Decides, as audio tokens arrive, which frames to decode and which of
    the decoded samples to emit, so playback can start before generation ends.

    When frame i completes and i >= 3, the window [i-3, i] is decoded and its
    second frame (i-2) is emitted, as Canopy's decoder does: one frame of left
    context, two of right, which is what the non-causal SNAC decoder wants at
    a boundary. The very first window also yields frame 0, and `flush` yields
    the last two, so unlike Canopy's decoder nothing is dropped.
    Returns (window_tokens, slice_of_window_samples) pairs; the caller decodes
    the window and takes the slice.
    """

    def __init__(self):
        self.tokens: list[int] = []
        self.frames_emitted = 0

    def push(self, token_id: int) -> list[tuple[list[int], slice]]:
        if not is_audio_token(token_id):
            return []
        self.tokens.append(token_id)
        if len(self.tokens) % TOKENS_PER_FRAME:
            return []
        n = len(self.tokens) // TOKENS_PER_FRAME          # frames complete
        if n < WINDOW_FRAMES:
            return []
        window = self.tokens[-WINDOW_FRAMES * TOKENS_PER_FRAME:]
        first_emit = self.frames_emitted               # frame index to emit next
        target = n - 3                                 # the window's second frame
        if first_emit > target:
            return []
        # frames first_emit..target all lie inside the window (window starts at n-4)
        lo = (first_emit - (n - WINDOW_FRAMES)) * SAMPLES_PER_FRAME
        hi = (target - (n - WINDOW_FRAMES) + 1) * SAMPLES_PER_FRAME
        self.frames_emitted = target + 1
        return [(window, slice(lo, hi))]

    def flush(self) -> list[tuple[list[int], slice]]:
        n = len(self.tokens) // TOKENS_PER_FRAME
        if n == 0 or self.frames_emitted >= n:
            return []
        start = max(0, n - WINDOW_FRAMES)
        window = self.tokens[start * TOKENS_PER_FRAME:n * TOKENS_PER_FRAME]
        lo = (self.frames_emitted - start) * SAMPLES_PER_FRAME
        hi = (n - start) * SAMPLES_PER_FRAME
        self.frames_emitted = n
        return [(window, slice(lo, hi))]

    @property
    def frames(self) -> int:
        return len(self.tokens) // TOKENS_PER_FRAME
