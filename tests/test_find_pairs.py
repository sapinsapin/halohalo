"""
Discovery tests for halolib.sources.livestream.find_pairs.

Runs under pytest, or standalone with no test dependency:

    python tests/test_find_pairs.py
"""

import json
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from halolib.sources import livestream as ls  # noqa: E402

DOC = {
    "metadata": {"linguistic_profile": {"primary_language": "Taglish"}},
    "transcription": [{"time_range": "00:01 - 00:02", "dialogue": [{"s": "Speaker 1", "txt": "hi"}]}],
}


def _json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(DOC), encoding="utf-8")


def _audio(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00\x01\x02")


def _ids(root: Path) -> set[str]:
    return {file_id for file_id, _, _ in ls.find_pairs(root)}


def test_hub_layout_round_trips():
    """A downloaded halo-livestream-raw snapshot must feed straight back in."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _json(root / "transcripts" / "rec1.json")
        _audio(root / "audio" / "rec1.m4a")

        pairs = ls.find_pairs(root)
        assert len(pairs) == 1, pairs
        file_id, json_path, audio_path = pairs[0]
        assert file_id == "rec1"
        assert json_path.name == "rec1.json"
        assert audio_path.name == "rec1.m4a"


def test_loose_pair():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _json(root / "rec2.json")
        _audio(root / "rec2.mp4")
        assert _ids(root) == {"rec2"}


def test_directory_per_recording():
    """The layout the recorder hands over: {id}/ holding both files."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _json(root / "rec3" / "rec3.json")
        _audio(root / "rec3" / "rec3.mp4")
        assert _ids(root) == {"rec3"}


def test_zip_archive():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        staging = root / "_src"
        _json(staging / "rec4.json")
        _audio(staging / "rec4.mp4")
        with zipfile.ZipFile(root / "rec4.zip", "w") as zf:
            zf.write(staging / "rec4.json", "rec4.json")
            zf.write(staging / "rec4.mp4", "rec4.mp4")
        for leftover in staging.iterdir():
            leftover.unlink()
        staging.rmdir()

        assert _ids(root) == {"rec4"}


def test_mixed_layouts_dedupe_by_file_id():
    """The same recording in two layouts is yielded once, not twice."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _json(root / "transcripts" / "dup.json")
        _audio(root / "audio" / "dup.m4a")
        _json(root / "dup.json")
        _audio(root / "dup.mp4")

        pairs = ls.find_pairs(root)
        assert len(pairs) == 1, pairs
        # Hub layout is scanned first, so it wins.
        assert pairs[0][2].suffix == ".m4a"


def test_cache_decodes_are_not_mistaken_for_sources():
    """`_16k`/`_24k` decodes written next to a source must never be picked."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _json(root / "rec5" / "rec5.json")
        _audio(root / "rec5" / "rec5_16k.wav")
        _audio(root / "rec5" / "rec5_24k.wav")
        _audio(root / "rec5" / "rec5.mp4")

        pairs = ls.find_pairs(root)
        assert len(pairs) == 1
        assert pairs[0][2].name == "rec5.mp4"


def test_directory_without_audio_is_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _json(root / "rec6" / "rec6.json")
        assert _ids(root) == set()


def test_status_json_is_not_a_transcript():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "rec7.status.json").write_text("{}", encoding="utf-8")
        _audio(root / "rec7.status.mp4")
        assert _ids(root) == set()


def test_flac_is_discoverable():
    """Raw uploads encode PCM to FLAC; discovery has to find it again."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _json(root / "transcripts" / "rec8.json")
        _audio(root / "audio" / "rec8.flac")
        assert _ids(root) == {"rec8"}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as err:
            failed += 1
            print(f"  FAIL  {test.__name__}: {err}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
