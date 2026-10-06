import re

import orpheus_tokens as T


def _trainer_frontend_char(text):
    """finetune_orpheus.frontend_text(text, "char"), reproduced from the
    trainer (which imports torch at module scope, so cannot be imported here)."""
    out = []
    for i, w in enumerate(re.compile(r"[^\w\s']").sub("", text.lower()).split()):
        if i:
            out.append("|")
        out.extend(list(w))
    return " ".join(out)


def test_char_frontend_matches_the_trainer():
    for s in ["Magandang umaga po, kumusta kayo?", "Maayong buntag sa tanan.",
              "Dili ko kahibalo!", "Si Dr. Santos ang doktor namin.", "ano'ng oras na"]:
        assert T.frontend_char(s) == _trainer_frontend_char(s)
    assert T.frontend_char("Kumusta ka") == "k u m u s t a | k a"
    assert T.prompt_text("CEB_0200", "Maayo.") == "CEB_0200: m a a y o"
    assert T.normalise_text("ano’ng") == "ano'ng"
    assert T.has_digits("500 pesos") and not T.has_digits("limang daan")


def test_prompt_framing_matches_canopy():
    ids = T.wrap_prompt([1, 2, 3])
    assert ids == [128259, 1, 2, 3, 128009, 128260, 128261, 128257]


def test_codes_round_trip_the_trainer_layout():
    # the trainer's encode layout: frame j -> [c0[j], c1[2j], c2[4j], c2[4j+1], c1[2j+1], c2[4j+2], c2[4j+3]]
    c0, c1, c2 = [5, 9], [10, 11, 12, 13], [20, 21, 22, 23, 24, 25, 26, 27]
    tokens = []
    for j in range(2):
        tokens += [c0[j] + T.AUDIO_BASE,
                   c1[2 * j] + T.AUDIO_BASE + 1 * 4096,
                   c2[4 * j] + T.AUDIO_BASE + 2 * 4096,
                   c2[4 * j + 1] + T.AUDIO_BASE + 3 * 4096,
                   c1[2 * j + 1] + T.AUDIO_BASE + 4 * 4096,
                   c2[4 * j + 2] + T.AUDIO_BASE + 5 * 4096,
                   c2[4 * j + 3] + T.AUDIO_BASE + 6 * 4096]
    assert T.codes_from_tokens(tokens) == (c0, c1, c2)
    # a non-audio token is ignored, a partial frame is dropped
    assert T.codes_from_tokens([T.EOS_SPEECH] + tokens + tokens[:3]) == (c0, c1, c2)
    # a token sampled into the wrong band wraps rather than indexing out of range
    wrong = tokens[:]
    wrong[1] = 10 + T.AUDIO_BASE            # band 0 value in slot 1
    c = T.codes_from_tokens(wrong)
    assert 0 <= c[1][0] < 4096


def test_custom_token_pieces():
    assert T.custom_token_id("<custom_token_10>") == 128266
    assert T.custom_token_id("<custom_token_2>") == T.EOS_SPEECH
    assert T.custom_token_id("hello") is None


def test_frame_window_emits_every_frame_once_with_context():
    w = T.FrameWindow()
    emitted = []
    n_frames = 9
    for j in range(n_frames):
        for i in range(7):
            out = w.push(T.AUDIO_BASE + i * 4096 + j)
            for win, sl in out:
                assert len(win) == 28
                emitted.append((len(win) // 7, sl.start // 2048, sl.stop // 2048))
    for win, sl in w.flush():
        emitted.append((len(win) // 7, sl.start // 2048, sl.stop // 2048))
    # the first window yields frames 0..1, then one frame per new frame, then the flush
    assert emitted[0] == (4, 0, 2)
    assert emitted[1:-1] == [(4, 1, 2)] * (n_frames - 4)
    assert emitted[-1] == (4, 2, 4)
    total = sum(hi - lo for _, lo, hi in emitted)
    assert total == n_frames
    assert w.frames == n_frames


def test_frame_window_short_utterance_flushes_everything():
    w = T.FrameWindow()
    for j in range(2):
        for i in range(7):
            assert w.push(T.AUDIO_BASE + i * 4096 + j) == []
    out = w.flush()
    assert len(out) == 1
    win, sl = out[0]
    assert len(win) == 14 and (sl.start, sl.stop) == (0, 2 * 2048)
    assert w.flush() == []
