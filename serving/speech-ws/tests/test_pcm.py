import numpy as np
import pytest
from scipy.signal import resample_poly

import pcm


def test_f32_to_pcm16_clips_and_is_little_endian():
    out = pcm.f32_to_pcm16(np.array([0.0, 1.0, -1.0, 2.0, -2.0, np.nan], dtype=np.float32))
    vals = np.frombuffer(out, dtype="<i2")
    assert vals.tolist() == [0, 32767, -32767, 32767, -32767, 0]
    assert pcm.f32_to_pcm16(np.array([0.5]))[0:2] == np.int16(16384).astype("<i2").tobytes()


def test_pcm16_round_trip_and_odd_tail():
    x = (np.arange(10) / 10).astype(np.float32)
    back = pcm.pcm16_to_f32(pcm.f32_to_pcm16(x) + b"\x01")   # a stray odd byte is dropped
    assert back.size == 10
    assert np.allclose(back, x, atol=1 / 32767)


def test_frames_are_even_and_cover_everything():
    data = bytes(range(256)) * 30          # 7680 bytes
    frames = pcm.frames_of(data, 3200)
    assert [len(f) for f in frames] == [3200, 3200, 1280]
    assert b"".join(frames) == data
    with pytest.raises(ValueError):
        pcm.frames_of(data, 3201)


def test_wav_header():
    w = pcm.wav_bytes(b"\x00\x00" * 16000)
    assert w[:4] == b"RIFF" and w[8:12] == b"WAVE" and len(w) == 44 + 32000
    import struct
    assert struct.unpack("<I", w[24:28])[0] == 16000


def test_stream_resampler_matches_one_shot_without_seams():
    sr_in, sr_out = 24000, 16000
    t = np.arange(sr_in * 2) / sr_in
    x = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 1250 * t)).astype(np.float32)
    whole = resample_poly(x.astype(np.float64), 2, 3)

    rs = pcm.StreamResampler(sr_in, sr_out)
    out = []
    # uneven chunks, like SNAC windows arriving (2048 samples) and a ragged start
    cuts = [0, 700, 2748, 4796, 6844, 8000, 10048, 12096, 20000, 30000, x.size]
    for a, b in zip(cuts, cuts[1:]):
        out.append(rs.process(x[a:b]))
    out.append(rs.flush())
    y = np.concatenate(out)
    # exact length, and no seam: the streamed result equals the one-shot one
    # everywhere except the filter's few-sample transient at the very edges
    assert y.size == whole.size
    assert np.max(np.abs(y[64:-64] - whole[64:-64])) < 2e-3
    # a seam would show as a spike in the first difference
    d = np.abs(np.diff(y))
    assert d.max() < 0.35


def test_stream_resampler_identity_and_flush_empty():
    rs = pcm.StreamResampler(16000, 16000)
    x = np.ones(5, dtype=np.float32)
    assert np.array_equal(rs.process(x), x)
    assert rs.flush().size == 0
    rs2 = pcm.StreamResampler(24000, 16000)
    assert rs2.process(np.zeros(0, dtype=np.float32)).size == 0
    assert rs2.flush().size == 0


def test_trim_edges_removes_floor_but_keeps_speech():
    sr = 16000
    speech = 0.5 * np.sin(2 * np.pi * 300 * np.arange(sr) / sr)
    floor = 0.005 * np.random.default_rng(0).standard_normal(sr)
    wav = np.concatenate([floor, speech, floor]).astype(np.float32)
    trimmed = pcm.trim_edges(wav)
    assert sr <= trimmed.size <= sr + int(sr * 0.2)
    assert pcm.trim_edges(np.zeros(4000, dtype=np.float32)).size == 0
