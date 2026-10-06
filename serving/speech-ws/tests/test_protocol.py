import json

import pytest

import protocol as P


def test_end_frame_is_frozen():
    assert P.END == b'{"type":"end"}'
    assert len(P.END) == 14


def test_error_frame_shape():
    f = json.loads(P.error("x"))
    assert f == {"type": "error", "message": "x"}
    assert b" " not in P.error("x")           # compact, as firmware expects


@pytest.mark.parametrize("raw,msg", [
    ("this is not json", P.ERR_BAD_JSON),
    ("[1,2]", P.ERR_NOT_OBJECT),
    ("{}", P.ERR_NO_TEXT),
    ('{"text": 5}', P.ERR_NO_TEXT),
    ('{"text": "   "}', P.ERR_EMPTY_TEXT),
    ('{"text": "' + "A" * 501 + '"}', P.ERR_TOO_LONG),
    ('{"text": "' + "A" * 9000 + '"}', P.ERR_FRAME_TOO_BIG),
    (None, P.ERR_BAD_JSON),
])
def test_parse_rejects(raw, msg):
    with pytest.raises(P.BadRequest) as e:
        P.parse(raw)
    assert str(e.value) == msg


def test_parse_accepts_optional_fields_and_ignores_unknown():
    r = P.parse('{"text": " Kumusta ", "lang": "CEB", "voice": "female", "meta": 1, "x": 2}')
    assert (r.text, r.lang, r.voice, r.meta) == ("Kumusta", "CEB", "female", True)


def test_asr_control_frames():
    c = P.parse_asr_control('{"type":"start","lang":"ceb","sample_rate":48000,"partials":false}')
    assert (c.kind, c.lang, c.sample_rate, c.partials) == ("start", "ceb", 48000, False)
    assert P.parse_asr_control('{"type":"stop"}').kind == "stop"
    # a bare object is a start with defaults
    c = P.parse_asr_control("{}")
    assert (c.kind, c.lang, c.sample_rate, c.partials) == ("start", None, 16000, True)
    with pytest.raises(P.BadRequest):
        P.parse_asr_control('{"type":"start","sample_rate":44100}')
    with pytest.raises(P.BadRequest) as e:
        P.parse_asr_control('{"type":"flush"}')
    assert P.ERR_UNKNOWN_TYPE in str(e.value)
