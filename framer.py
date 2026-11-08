"""
Phase 3 - Dynamic Window Framer (pure Python), plus the timestamp checks that keep
the model honest.

- frame():     hook timestamp T from the Hook Hunter -> a clip window T .. T+60s.
- apply_cut(): end timestamp from the Judge -> a checked, final 20-60s clip.

The model is never trusted with time. Every timestamp it returns must match a real
event in the script (within `snap_tolerance`), and the final clip length is enforced
here, not in the prompt.
"""

from __future__ import annotations

from parser import to_seconds


def _parse(ts: str):
    try:
        return to_seconds(ts.strip().strip("[]()"))
    except (ValueError, IndexError, AttributeError):
        return None


def frame(ep: dict, hook, idx: int, cfg: dict):
    """Returns (clip, None) or (None, reason)."""
    tol = cfg["snap_tolerance"]
    t = _parse(hook.start_timestamp)
    if t is None:
        return None, f"hook timestamp '{hook.start_timestamp}' is not a valid time"

    # 1. The hook must be a real event: snap to the closest event start.
    starts = [e for e in ep["events"] if e["type"] != "camera"] or ep["events"]
    nearest = min(starts, key=lambda e: abs(e["start"] - t))
    if abs(nearest["start"] - t) > tol:
        return None, (f"hook timestamp {hook.start_timestamp} matches no line in the script "
                      f"(closest is {nearest['start']:.2f}s)")
    hook_t = nearest["start"]

    # 2. Start the clip on the camera shot that contains the hook, if that shot began
    #    less than `hook_must_land_within` seconds earlier (so we start on a clean cut).
    lead = cfg["hook_must_land_within"]
    cuts = [c for c in ep["cut_points"] if hook_t - lead < c <= hook_t]
    start = min(cuts) if cuts else hook_t

    # 3. Window = start .. start+60s, but never past the end of the episode.
    window_end = min(start + cfg["max_seconds"], ep["duration"])
    if window_end - start < cfg["min_seconds"]:
        return None, (f"hook at {hook_t:.1f}s leaves only {window_end - start:.1f}s before "
                      f"the episode ends (need {cfg['min_seconds']}s)")

    events = [e for e in ep["events"] if start - 0.01 <= e["start"] < window_end]
    return {
        "clip_id": f"E{ep['episode']:02d}-H{idx}",
        "episode": ep["episode"],
        "hook_event_id": nearest["id"],
        "hook_type": hook.hook_type,
        "hook_description": hook.hook_description,
        "hook_at": hook_t,
        "start": start,
        "window_end": window_end,
        "events": events,
    }, None


def apply_cut(clip: dict, end_ts: str, ep: dict, cfg: dict):
    """Snap the Judge's end timestamp to a real line end / camera cut inside the
    allowed 20-60s range. Returns (end_seconds, note) or (None, reason)."""
    s = clip["start"]
    lo, hi = s + cfg["min_seconds"], min(s + cfg["max_seconds"], clip["window_end"])
    valid = sorted({e["end"] for e in clip["events"] if lo <= e["end"] <= hi} |
                   {c for c in ep["cut_points"] if lo <= c <= hi})
    if not valid:
        return None, "no line ends between 20s and 60s after the start"

    t = _parse(end_ts)
    if t is None:
        return None, f"end timestamp '{end_ts}' is not a valid time"
    end = min(valid, key=lambda v: abs(v - t))
    if abs(end - t) <= cfg["snap_tolerance"]:
        return end, ""
    if t < lo or t > hi:
        return end, f"model end {end_ts} was outside 20-60s; moved to {end:.2f}s"
    return end, f"model end {end_ts} matched no line end; moved to {end:.2f}s"


def overlap(a: dict, b: dict) -> float:
    """Share of the shorter clip covered by the other (same episode only)."""
    if a["episode"] != b["episode"]:
        return 0.0
    inter = min(a["end"], b["end"]) - max(a["start"], b["start"])
    return max(0.0, inter) / max(0.01, min(a["end"] - a["start"], b["end"] - b["start"]))
