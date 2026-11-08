"""
Phase 0 - Rule Maker (optional, run once per genre).

Gemini watches real promo Shorts from ReelShort / DramaBox and turns them into a
grading rubric (rules.json) that the Hook Hunter and Judge use.

    python rule_maker.py reference_promos.txt              # all links in the file
    python rule_maker.py reference_promos.txt --limit 10   # quick test on 10
    python rule_maker.py reference_promos.txt --provider mock   # offline plumbing test

Steps:
  1. LLM (1 call per video): describe the hook, the cut, the length, the structure.
  2. Python: length statistics and shares (continuous vs stitched, captions, narration).
  3. LLM (1 call): group the hooks and cliffhangers into named patterns with counts.
Outputs: rules.json (used automatically by run.py) and reference_analysis.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import yaml
from pydantic import BaseModel

from llm_nodes import LLM


class PromoAnalysis(BaseModel):
    duration_seconds: float
    is_one_continuous_stretch: bool       # false if it joins moments from different scenes
    number_of_segments: int
    hook_first_3_seconds: str             # what the viewer sees/hears in the first 3s
    hook_type: str                        # 2-4 words, e.g. "public slap"
    cliffhanger_final_moment: str         # the last moment before the cut
    cliffhanger_type: str                 # 2-4 words, e.g. "unanswered question"
    has_text_overlay_or_captions: bool
    has_narration_or_voiceover: bool
    why_it_works: str                     # one line


class Archetype(BaseModel):
    name: str
    description: str
    example: str
    count: int


class Patterns(BaseModel):
    hook_archetypes: list[Archetype]
    cliffhanger_archetypes: list[Archetype]
    other_patterns: list[str]


WATCH_PROMPT = """This is a promo Short for a vertical microdrama app ({platform}).
Watch the whole video and describe how it is built as a promo:
- duration_seconds: its length.
- is_one_continuous_stretch / number_of_segments: is it one unbroken stretch of a scene,
  or does it join moments from different scenes or episodes? Ignore title cards.
- hook_first_3_seconds and hook_type: what in the first 3 seconds makes a scroller stop.
- cliffhanger_final_moment and cliffhanger_type: where it cuts and what is left unresolved.
- has_text_overlay_or_captions, has_narration_or_voiceover.
- why_it_works: one line.
Describe only what is in the video. Keep hook_type and cliffhanger_type to 2-4 words."""

GROUP_PROMPT = """Below are {n} analyses of real microdrama promo Shorts.
Group them into the 4-6 most common HOOK patterns and the 4-6 most common CLIFFHANGER
patterns. For each pattern: a short name, a one-line description a script reader can apply
to a SCREENPLAY (no reliance on music or captions), one real example from the list, and how
many of the {n} promos use it. Then list up to 5 other_patterns you see across many promos
(e.g. how they open, pacing, what they never show).

ANALYSES
{rows}"""


def read_links(path: str) -> list[dict]:
    links, platform = [], "unknown"
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if "http" not in line:
            low = line.lower()
            platform = ("ReelShort" if "reel" in low else "DramaBox" if "drama" in low
                        else line)
            continue
        m = re.search(r"(?:shorts/|v=|youtu\.be/)([\w-]{11})", line)
        if m:
            links.append({"platform": platform, "video_id": m.group(1),
                          "url": f"https://www.youtube.com/watch?v={m.group(1)}"})
    return links


def stats(rows: list[dict]) -> dict:
    d = sorted(r["duration_seconds"] for r in rows if r["duration_seconds"] > 0)
    share = lambda k: round(sum(bool(r[k]) for r in rows) / len(rows), 2)
    q = statistics.quantiles(d, n=4) if len(d) >= 4 else [d[0], d[len(d) // 2], d[-1]] if d else [0, 0, 0]
    return {"promos_analysed": len(rows),
            "duration_median": round(statistics.median(d), 1) if d else None,
            "duration_p25": round(q[0], 1), "duration_p75": round(q[-1], 1),
            "duration_min": d[0] if d else None, "duration_max": d[-1] if d else None,
            "share_between_20_and_60s": round(sum(20 <= x <= 60 for x in d) / max(1, len(d)), 2),
            "share_one_continuous_stretch": share("is_one_continuous_stretch"),
            "share_text_overlay": share("has_text_overlay_or_captions"),
            "share_narration": share("has_narration_or_voiceover")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("links", help="text file of YouTube links, with platform header lines")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model")
    ap.add_argument("--provider")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="rules.json")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    if a.model:
        cfg["llm"]["model"] = a.model
    if a.provider:
        cfg["llm"]["provider"] = a.provider
    llm = LLM(cfg["llm"], cfg.get("cache_dir", ".cache"))
    links = read_links(a.links)[: a.limit]
    print(f"Watching {len(links)} promos with {cfg['llm']['provider']} / {llm.model}")

    rows, failed = [], []
    for i, l in enumerate(links, 1):
        mock = lambda: PromoAnalysis(
            duration_seconds=30 + i % 25, is_one_continuous_stretch=i % 3 != 0,
            number_of_segments=1 if i % 3 else 2, hook_first_3_seconds="[MOCK] slap",
            hook_type=["public slap", "betrayal line", "identity reveal"][i % 3],
            cliffhanger_final_moment="[MOCK] question", cliffhanger_type=
            ["unanswered question", "threat", "reveal mid-sentence"][i % 3],
            has_text_overlay_or_captions=True, has_narration_or_voiceover=i % 4 == 0,
            why_it_works="[MOCK]")
        r = llm.structured(WATCH_PROMPT.format(platform=l["platform"]), PromoAnalysis,
                           f"watch-{l['video_id']}", mock=mock, video_url=l["url"])
        if r is None:
            failed.append(l)
            print(f"  {i:>2}/{len(links)} {l['video_id']}: FAILED (private, removed or quota?)")
            continue
        rows.append({**l, **r.model_dump()})
        print(f"  {i:>2}/{len(links)} {l['platform']:<9} {r.duration_seconds:>5.0f}s  "
              f"hook: {r.hook_type:<24} cliff: {r.cliffhanger_type}")

    if not rows:
        raise SystemExit("No promo could be analysed - check the model supports video and the links are public.")

    with open("reference_analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = [json.dumps({k: r[k] for k in ("platform", "duration_seconds", "hook_type",
                                             "hook_first_3_seconds", "cliffhanger_type",
                                             "cliffhanger_final_moment", "why_it_works")})
               for r in rows]
    pats = llm.structured(GROUP_PROMPT.format(n=len(rows), rows="\n".join(summary)), Patterns,
                          "group", mock=lambda: Patterns(
                              hook_archetypes=[Archetype(name="[MOCK] public slap", description="d",
                                                         example="e", count=len(rows))],
                              cliffhanger_archetypes=[Archetype(name="[MOCK] question", description="d",
                                                                example="e", count=len(rows))],
                              other_patterns=[]))
    if pats is None:
        raise SystemExit("Grouping call failed; reference_analysis.csv is saved - rerun to retry.")

    rules = {"source": a.links, "model": llm.model,
             "stats": {"all": stats(rows),
                       **{p: stats([r for r in rows if r["platform"] == p])
                          for p in sorted({r["platform"] for r in rows})}},
             **pats.model_dump(), "failed_links": [l["url"] for l in failed]}
    Path(a.out).write_text(json.dumps(rules, indent=2))

    s = rules["stats"]["all"]
    print(f"\nAnalysed {s['promos_analysed']}/{len(links)} promos. Median length "
          f"{s['duration_median']}s (middle half {s['duration_p25']}-{s['duration_p75']}s); "
          f"{s['share_one_continuous_stretch']:.0%} are one continuous stretch.")
    print("Hook patterns:", ", ".join(f"{h.name} ({h.count})" for h in pats.hook_archetypes))
    print("Cliffhanger patterns:", ", ".join(f"{c.name} ({c.count})" for c in pats.cliffhanger_archetypes))
    print(f"Saved {a.out} (used automatically by run.py) and reference_analysis.csv")
    print(f"Model usage: {llm.stats}")


if __name__ == "__main__":
    main()
