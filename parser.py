"""
Phase 1 - Script Parser (pure Python, no AI).

Reads Kuku screenplay .txt files and turns each one into a clean JSON timeline:
episode metadata (number, logline, continuity notes) + a flat, time-sorted list
of events (CAMERA / ACTION / DIALOGUE / SFX), every timestamp in float seconds.

Every event gets a stable ID like "E01-017" so later phases can refer to exact
moments without inventing timestamps.

Usage:
    python parser.py scripts/            # parse a folder of .txt files
    python parser.py scripts/ -o parsed/ # also write one JSON per episode
"""

import json
import re
import sys
from pathlib import Path

# --- regexes -----------------------------------------------------------------

# Accepts "02:18.00", "2:18", and "1:02:18.00" (hours, just in case)
TS = r"(\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?"

RE_EP_NUM = re.compile(r"EPISODE NUMBER:\s*Episode\s*(\d+)", re.I)
RE_LOGLINE = re.compile(r"EPISODE LOGLINE:\s*(.+)", re.I)
RE_CONTINUITY = re.compile(r"CONTINUITY NOTES:\s*(.+)", re.I)
RE_SCENE = re.compile(r"^SCENE\s+(\d+)", re.I)
RE_SCENE_HEADING = re.compile(r"^Scene heading:\s*(.+)", re.I)
RE_SCENE_START = re.compile(rf"^Scene start:\s*({TS})", re.I)
RE_SCENE_END = re.compile(rf"^Scene end:\s*({TS})", re.I)
# "**CAMERA:**", "**SOUND FX:**", "**LADY JULIANA (Confused, Breathless):**",
# "**MATRIARCH (O.S.) (Firm):**"  -> the whole label between ** and :**
RE_BLOCK = re.compile(r"^\*\*(.+?):\*\*\s*$")
RE_TIMING = re.compile(rf"Start:\s*\(({TS})\)\s*\|\s*End:\s*\(({TS})\)", re.I)

NON_DIALOGUE = {"CAMERA": "camera", "ACTION": "action",
                "SOUND FX": "sfx", "SFX": "sfx", "MUSIC": "sfx", "VO": "dialogue"}


def to_seconds(ts: str) -> float:
    """'01:05.60' -> 65.6 ; '1:02:18.00' -> 3738.0"""
    parts = ts.split(":")
    secs = float(parts[-1])
    mins = int(parts[-2])
    hours = int(parts[-3]) if len(parts) == 3 else 0
    return round(hours * 3600 + mins * 60 + secs, 2)


def clean(line: str) -> str:
    """Drop markdown escapes (\\*, \\!, \\-) that appear when files pass through
    Google Docs/Drive, plus stray whitespace."""
    return re.sub(r"\\([*!\-_#.()\[\]])", r"\1", line).strip()


def split_label(label: str):
    """'LADY JULIANA (Confused, Breathless)' -> ('LADY JULIANA', 'Confused, Breathless')
       'MATRIARCH (O.S.) (Firm)'             -> ('MATRIARCH', 'Firm')  [O.S. kept as flag]"""
    parens = re.findall(r"\(([^)]*)\)", label)
    speaker = re.sub(r"\s*\([^)]*\)", "", label).strip()
    offscreen = any(p.strip().upper() in {"O.S.", "V.O.", "O.C."} for p in parens)
    emotion = ", ".join(p for p in parens if p.strip().upper() not in {"O.S.", "V.O.", "O.C."})
    return speaker, emotion, offscreen


# --- parser ------------------------------------------------------------------

def parse_episode(text: str, source: str = "") -> dict:
    lines = [clean(l) for l in text.splitlines()]

    meta = {"source_file": source, "episode": None, "logline": "", "continuity": ""}
    for l in lines[:15]:
        if m := RE_EP_NUM.search(l):
            meta["episode"] = int(m.group(1))
        elif m := RE_LOGLINE.search(l):
            meta["logline"] = m.group(1).strip()
        elif m := RE_CONTINUITY.search(l):
            meta["continuity"] = m.group(1).strip()
    if meta["episode"] is None:  # fall back to the filename: episode-024-...
        m = re.search(r"episode-0*(\d+)", source, re.I)
        meta["episode"] = int(m.group(1)) if m else -1

    scenes, events, warnings = [], [], []
    scene = None
    i = 0
    while i < len(lines):
        l = lines[i]

        if m := RE_SCENE.match(l):
            scene = {"scene": int(m.group(1)), "heading": "", "start": None, "end": None}
            scenes.append(scene)
        elif scene and (m := RE_SCENE_HEADING.match(l)):
            scene["heading"] = m.group(1).strip()
        elif scene and (m := RE_SCENE_START.match(l)):
            scene["start"] = to_seconds(m.group(1))
        elif scene and (m := RE_SCENE_END.match(l)):
            scene["end"] = to_seconds(m.group(1))

        elif m := RE_BLOCK.match(l):
            label = m.group(1).strip()
            # next non-empty line should be the timing line
            j = i + 1
            while j < len(lines) and not lines[j]:
                j += 1
            t = RE_TIMING.search(lines[j]) if j < len(lines) else None
            if not t:
                if label.lower() != "actions and dialogue":
                    warnings.append(f"line {i+1}: block '{label}' has no timing, skipped")
                i += 1
                continue
            # body = following lines until a blank line or the next block
            k, body = j + 1, []
            while k < len(lines) and lines[k] and not RE_BLOCK.match(lines[k]) \
                    and not lines[k].startswith("--- SCRIPT"):
                body.append(lines[k])
                k += 1

            upper = label.upper()
            if upper in NON_DIALOGUE:
                etype, speaker, emotion, offscreen = NON_DIALOGUE[upper], None, "", False
            else:
                etype = "dialogue"
                speaker, emotion, offscreen = split_label(label)

            start, end = to_seconds(t.group(1)), to_seconds(t.group(3))
            if end < start:
                warnings.append(f"line {j+1}: end before start ({start}->{end})")
            events.append({
                "type": etype,
                "speaker": speaker,
                "emotion": emotion,
                "offscreen": offscreen,
                "text": " ".join(body).strip().strip('"'),
                "start": start,
                "end": end,
                "scene": scene["scene"] if scene else None,
            })
            i = k
            continue
        i += 1

    # sort by time (camera + dialogue overlap; keep file order as tie-break)
    events.sort(key=lambda e: (e["start"], e["end"]))
    ep = meta["episode"]
    for n, e in enumerate(events, 1):
        e["id"] = f"E{ep:02d}-{n:03d}"

    duration = max((e["end"] for e in events), default=0.0)
    # Shot boundaries = where the editor can actually cut (CAMERA block edges).
    cuts = sorted({e["start"] for e in events if e["type"] == "camera"} |
                  {e["end"] for e in events if e["type"] == "camera"})

    return {**meta, "duration": duration, "scenes": scenes,
            "cut_points": cuts, "events": events, "warnings": warnings}


def parse_folder(folder: str) -> list[dict]:
    files = sorted(Path(folder).rglob("*.txt"))
    eps = [parse_episode(f.read_text(encoding="utf-8", errors="replace"), f.name) for f in files]
    return sorted(eps, key=lambda e: e["episode"])


def summary(ep: dict) -> str:
    counts = {}
    for e in ep["events"]:
        counts[e["type"]] = counts.get(e["type"], 0) + 1
    return (f"Ep {ep['episode']:>2} | {ep['duration']:6.1f}s | {len(ep['scenes'])} scene(s) | "
            f"{len(ep['events'])} events {counts} | {len(ep['warnings'])} warning(s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    out = Path(sys.argv[sys.argv.index("-o") + 1]) if "-o" in sys.argv else None
    episodes = parse_folder(src) if src.is_dir() else \
        [parse_episode(src.read_text(encoding="utf-8"), src.name)]
    for ep in episodes:
        print(summary(ep))
        for w in ep["warnings"]:
            print("   !", w)
        if out:
            out.mkdir(parents=True, exist_ok=True)
            (out / f"episode_{ep['episode']:03d}.json").write_text(json.dumps(ep, indent=2))
