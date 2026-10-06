import numpy as np

from asr_stream import Endpointer, energy_vad

SR = 16000


def tone(seconds, amp=0.3):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def quiet(seconds):
    return np.zeros(int(SR * seconds), dtype=np.float32)


def test_energy_vad():
    assert energy_vad(tone(0.02))
    assert not energy_vad(quiet(0.02))
    assert not energy_vad(np.zeros(0, dtype=np.float32))


def test_two_phrases_become_two_finals_with_partials():
    ep = Endpointer(partial_every_s=1.0, silence_ms=600)
    stream = np.concatenate([quiet(0.5), tone(2.5), quiet(1.0), tone(1.2), quiet(0.2)])
    segs = []
    for i in range(0, stream.size, 1600):        # 100 ms frames, as the wire sends them
        segs += ep.feed(stream[i:i + 1600])
    segs += ep.finish()
    finals = [s for s in segs if s.final]
    partials = [s for s in segs if not s.final]
    assert len(finals) == 2
    assert len(partials) >= 2                      # the 2.5 s phrase yields partials
    # timestamps follow the stream, with a little pre-roll before the onset
    assert 0.2 <= finals[0].start_s <= 0.5
    assert 2.9 <= finals[0].end_s <= 3.7
    assert finals[1].start_s > finals[0].end_s
    # the final carries the whole phrase, partials a growing prefix of it
    assert finals[0].audio.size >= partials[-1].audio.size or partials[-1].start_s > finals[0].start_s


def test_short_blips_are_ignored_and_long_phrases_are_cut():
    ep = Endpointer(min_speech_ms=250, max_segment_s=2.0, partial_every_s=10)
    stream = np.concatenate([tone(0.1), quiet(1.0), tone(5.0)])
    segs = []
    for i in range(0, stream.size, 3200):
        segs += ep.feed(stream[i:i + 3200])
    segs += ep.finish()
    finals = [s for s in segs if s.final]
    assert all(s.audio.size / SR >= 0.25 for s in finals)
    assert len(finals) >= 2                        # the 5 s tone was cut at 2 s
    assert max(s.audio.size / SR for s in finals) <= 2.1


def test_silence_only_yields_nothing():
    ep = Endpointer()
    assert ep.feed(quiet(3.0)) == []
    assert ep.finish() == []
