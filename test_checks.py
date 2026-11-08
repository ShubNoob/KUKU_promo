"""
Tests for the parts that must never break: parsing and the timestamp checks that stop
the model from inventing or stretching clips. Run: python -m pytest tests/ -q
(or just: python tests/test_checks.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framer import apply_cut, frame                          # noqa: E402
from llm_nodes import LLM, Hook, HookList, fmt               # noqa: E402
from parser import parse_episode, to_seconds                 # noqa: E402

CFG = {"min_seconds": 20, "max_seconds": 60, "hook_must_land_within": 3.0,
       "snap_tolerance": 1.0}

SCRIPT = "\n".join(
    ["EPISODE NUMBER: Episode 7", "EPISODE LOGLINE: test", "CONTINUITY NOTES: test", "",
     "SCENE 01", "Scene heading: INT. HALL – DAY", "Scene start: 00:00.00",
     "Scene end: 01:30.00", ""]
    + sum([[r"\*\*CAMERA:\*\*", f"Start: ({fmt(t)}) | End: ({fmt(t + 5)})", "Shot.", "",
            f"**HERO (Angry):**", f"Start: ({fmt(t + 1)}) | End: ({fmt(t + 4)})",
            f'"Line at {t}\\!"', ""] for t in range(0, 90, 5)], []))
EP = parse_episode(SCRIPT, "episode-007-x.txt")


def test_parser():
    assert EP["episode"] == 7 and EP["duration"] == 90.0
    assert to_seconds("01:05.60") == 65.6
    d = [e for e in EP["events"] if e["type"] == "dialogue"][0]
    assert d["speaker"] == "HERO" and d["emotion"] == "Angry" and d["text"] == "Line at 0!"


def test_invented_hook_timestamp_rejected():
    clip, why = frame(EP, Hook(start_timestamp="00:12.50", hook_type="x", hook_description="x"), 1, CFG)
    assert clip is None and "matches no line" in why


def test_hook_snaps_and_lands_in_first_3s():
    clip, _ = frame(EP, Hook(start_timestamp="00:11.20", hook_type="x", hook_description="x"), 1, CFG)
    assert clip["hook_at"] == 11.0 and clip["start"] == 10.0     # starts on the camera cut
    assert clip["hook_at"] - clip["start"] <= 3


def test_hook_too_late_rejected():
    clip, why = frame(EP, Hook(start_timestamp="01:16.00", hook_type="x", hook_description="x"), 1, CFG)
    assert clip is None and "only" in why


def test_end_out_of_range_is_clamped():
    clip, _ = frame(EP, Hook(start_timestamp="00:06.00", hook_type="x", hook_description="x"), 1, CFG)
    end, note = apply_cut(clip, "01:40.00", EP, CFG)              # model asked for 95s
    assert clip["start"] + 20 <= end <= clip["start"] + 60 and "outside" in note
    end, note = apply_cut(clip, "00:09.00", EP, CFG)              # model asked for 4s
    assert end - clip["start"] >= 20 and "outside" in note


def test_end_snaps_to_real_line_end():
    clip, _ = frame(EP, Hook(start_timestamp="00:06.00", hook_type="x", hook_description="x"), 1, CFG)
    end, note = apply_cut(clip, "00:39.20", EP, CFG)
    assert end == 39.0 and note == ""


def test_garbage_end_rejected():
    clip, _ = frame(EP, Hook(start_timestamp="00:06.00", hook_type="x", hook_description="x"), 1, CFG)
    end, note = apply_cut(clip, "around the middle", EP, CFG)
    assert end is None and "not a valid time" in note


def test_llm_retries_bad_json_then_caches(tmp_path=None):
    import tempfile
    d = tmp_path or Path(tempfile.mkdtemp())
    llm = LLM({"provider": "gemini", "model": "fake", "requests_per_minute": 6000,
               "max_retries": 3}, str(d))
    replies = iter(["not json", '```json\n{"hooks": []}\n```'])
    llm._call = lambda p, s: next(replies)
    import time
    real_sleep, time.sleep = time.sleep, lambda s: None           # skip backoff waits
    try:
        assert llm.structured("p", HookList, "t").hooks == []
        assert llm.stats["retries"] == 1
        llm._call = lambda p, s: (_ for _ in ()).throw(AssertionError("should hit cache"))
        assert llm.structured("p", HookList, "t").hooks == []
        assert llm.stats["cache_hits"] == 1
    finally:
        time.sleep = real_sleep


def test_llm_gives_up_cleanly_on_rate_limit():
    import tempfile, time
    llm = LLM({"provider": "gemini", "model": "fake2", "requests_per_minute": 6000,
               "max_retries": 2}, tempfile.mkdtemp())
    llm._call = lambda p, s: (_ for _ in ()).throw(RuntimeError("429 RESOURCE_EXHAUSTED"))
    real_sleep, time.sleep = time.sleep, lambda s: None
    try:
        assert llm.structured("p", HookList, "t") is None and llm.stats["failed_calls"] == 1
    finally:
        time.sleep = real_sleep


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"{len(tests)} passed")
