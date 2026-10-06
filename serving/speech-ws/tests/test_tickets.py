import pytest

import tickets

KEY = "serving-key"


def test_round_trip_and_scope():
    t = tickets.mint(KEY, 60, "speech", now=1000.0)
    assert tickets.verify(KEY, t, "tts", now=1030.0)
    assert tickets.verify(KEY, t, "asr", now=1030.0)
    assert tickets.expires_at(t) == 1060
    t_tts = tickets.mint(KEY, 60, "tts", now=1000.0)
    assert tickets.verify(KEY, t_tts, "tts", now=1001.0)
    assert not tickets.verify(KEY, t_tts, "asr", now=1001.0)


def test_expiry_and_tampering():
    t = tickets.mint(KEY, 60, now=1000.0)
    assert not tickets.verify(KEY, t, "tts", now=1061.0)
    assert not tickets.verify("other-key", t, "tts", now=1001.0)
    exp, nonce, scope, sig = t.split(".")
    assert not tickets.verify(KEY, f"{int(exp) + 1000}.{nonce}.{scope}.{sig}", "tts", now=1001.0)
    assert not tickets.verify(KEY, f"{exp}.{nonce}.asr.{sig}", "asr", now=1001.0)


def test_malformed():
    for bad in (None, "", "abc", "1.2.3", "x.y.speech.z", "1000.n.nope.sig"):
        assert not tickets.verify(KEY, bad, "tts", now=0.0)
    assert not tickets.verify("", tickets.mint(KEY, 60), "tts")
    with pytest.raises(ValueError):
        tickets.mint(KEY, 60, "video")
