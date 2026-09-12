"""
Index accumulation for the raw archive.

`index.jsonl` describes the whole dataset, not one run. A filtered run
(`--file-id`) that rewrote it from its own subset would drop every other
recording from the published index — and from the card built off it.

Runs under pytest, or standalone:  python tests/test_raw_index.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from halolib import raw  # noqa: E402


def _seed(stage: Path, file_ids) -> None:
    with open(stage / "index.jsonl", "w", encoding="utf-8") as f:
        for fid in file_ids:
            f.write(
                json.dumps(
                    {
                        "file_id": fid,
                        "audio": f"audio/{fid}.m4a",
                        "audio_bytes": 10,
                        "duration_seconds": 5,
                        "audio_sha256": "a" * 64,
                        "transcript_sha256": "b" * 64,
                    }
                )
                + "\n"
            )


def test_read_index_keys_by_file_id():
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        _seed(stage, ["rec1", "rec2"])
        assert sorted(raw.read_index(stage)) == ["rec1", "rec2"]


def test_missing_index_is_empty_not_an_error():
    with tempfile.TemporaryDirectory() as tmp:
        assert raw.read_index(Path(tmp)) == {}


def test_a_filtered_run_does_not_truncate_the_index():
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        _seed(stage, ["rec1", "rec2", "rec3"])

        # A run that staged nothing new must still leave every row in place.
        rows = raw.merge_index([], stage)
        assert [r["file_id"] for r in rows] == ["rec1", "rec2", "rec3"]
        assert sorted(raw.read_index(stage)) == ["rec1", "rec2", "rec3"]


def test_rows_are_sorted_for_a_stable_diff():
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        _seed(stage, ["zeta", "alpha", "mid"])
        rows = raw.merge_index([], stage)
        assert [r["file_id"] for r in rows] == ["alpha", "mid", "zeta"]


def test_a_truncated_line_does_not_poison_the_whole_index():
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        _seed(stage, ["rec1", "rec2"])
        with open(stage / "index.jsonl", "a", encoding="utf-8") as f:
            f.write('{"file_id": "rec3", "audio_by\n')  # killed mid-write

        assert sorted(raw.read_index(stage)) == ["rec1", "rec2"]


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
