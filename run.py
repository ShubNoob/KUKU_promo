"""
Run the whole Promo Extractor on a show.

    python run.py scripts/                      # folder of episode .txt files
    python run.py scripts.zip --out outputs/    # or a zip
    python run.py scripts/ --episodes 1-5       # only some episodes
    python run.py scripts/ --provider mock      # offline test run, no API key
    python run.py scripts/ --model gemini-2.5-flash   # swap model without editing config
    python run.py --score outputs/hand_check.csv      # after labelling the hand-check sheet

Phases: 1 parse (Python) -> 2 Hook Hunter (LLM, 1 call/episode) -> 3 frame T..T+60s
(Python) -> 4 Judge & Trimmer (LLM, 1 call/clip) + timestamp checks (Python)
-> 5 tiers (Python) + Ranker (LLM, 1 call) + hand-check sheet.
"""

from __future__ import annotations

import argparse
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

import yaml

from framer import apply_cut, frame
from llm_nodes import LLM, hook_hunter, judge, load_rubric, ranker
from parser import parse_folder
from ranker import (assign_tier, build_record, mark_overlaps, merge_ranking, score,
                    score_hand_check, write_outputs)


def episode_filter(spec: str | None):
    if not spec:
        return lambda n: True
    keep = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        keep.update(range(int(a), int(b or a) + 1))
    return lambda n: n in keep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scripts", nargs="?", help="folder or .zip of episode .txt files")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--episodes", help="e.g. 1-10 or 3,7,12")
    ap.add_argument("--provider", help="override llm.provider (gemini | openai_compatible | mock)")
    ap.add_argument("--model", help="override llm.model")
    ap.add_argument("--show", help="show name for the report (default: folder/zip name)")
    ap.add_argument("--score", metavar="HAND_CHECK_CSV", help="score a labelled hand-check sheet")
    a = ap.parse_args()

    if a.score:
        return score_hand_check(a.score)
    if not a.scripts:
        ap.error("give a scripts folder or zip")

    cfg = yaml.safe_load(open(a.config))
    if a.provider:
        cfg["llm"]["provider"] = a.provider
    if a.model:
        cfg["llm"]["model"] = a.model
    pcfg, tcfg = cfg["promo"], cfg["tiers"]
    t0 = time.time()

    # ---- Phase 1: parse ----------------------------------------------------
    src = Path(a.scripts)
    if src.suffix == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="scripts_"))
        zipfile.ZipFile(src).extractall(tmp)
        src = tmp
    keep = episode_filter(a.episodes)
    episodes = [e for e in parse_folder(src) if keep(e["episode"])]
    if not episodes:
        raise SystemExit(f"No episode .txt files found in {a.scripts}")
    parse_warnings = sum(len(e["warnings"]) for e in episodes)
    print(f"Phase 1: parsed {len(episodes)} episodes, "
          f"{sum(len(e['events']) for e in episodes)} events, {parse_warnings} warnings")

    llm = LLM(cfg["llm"], cfg.get("cache_dir", ".cache"))
    print(f"Model: {cfg['llm']['provider']} / {llm.model}")
    rules_file = cfg.get("rules_file", "rules.json")
    used_rules = load_rubric(rules_file) if rules_file else False
    print(f"Rubric: {'rules.json from reference promos' if used_rules else 'brief only (no rules.json)'}")

    records, rejected, hooks_found = [], [], 0
    for ep in episodes:
        # ---- Phase 2: Hook Hunter ------------------------------------------
        hooks = hook_hunter(llm, ep, pcfg["hooks_per_episode"])[: pcfg["hooks_per_episode"]]
        hooks_found += len(hooks)
        if not hooks:
            rejected.append({"stage": "hook hunter", "clip_id": f"E{ep['episode']:02d}",
                             "episode": ep["episode"], "reason": "no hooks returned"})

        # ---- Phase 3: frame T .. T+60s -------------------------------------
        clips, starts = [], set()
        for i, h in enumerate(hooks, 1):
            clip, why = frame(ep, h, i, pcfg)
            if clip and clip["start"] in starts:
                clip, why = None, "duplicate of another hook at the same moment"
            if not clip:
                rejected.append({"stage": "framer", "clip_id": f"E{ep['episode']:02d}-H{i}",
                                 "episode": ep["episode"], "reason": why})
                continue
            starts.add(clip["start"])
            clips.append(clip)

        # ---- Phase 4: Judge & Trimmer + timestamp check ---------------------
        for clip in clips:
            v = judge(llm, clip)
            if v is None:
                rejected.append({"stage": "judge", "clip_id": clip["clip_id"],
                                 "episode": ep["episode"], "reason": "model call failed"})
                continue
            end, note = apply_cut(clip, v.recommended_end_timestamp, ep, pcfg)
            if end is None:
                rejected.append({"stage": "timestamp check", "clip_id": clip["clip_id"],
                                 "episode": ep["episode"], "reason": note})
                continue
            r = build_record(clip, v, end, note)
            assign_tier(r, tcfg)
            records.append(r)
        print(f"  Ep {ep['episode']:>2}: {len(hooks)} hooks -> {len(clips)} clips judged "
              f"(api calls so far {llm.stats['api_calls']}, cache hits {llm.stats['cache_hits']})")

    # ---- Phase 5: tiers + ranker -------------------------------------------
    mark_overlaps(records)
    tier1 = [r for r in records if r["tier"] == 1]
    top = sorted(tier1, key=score, reverse=True)[: tcfg["top_n_for_ranker"]]
    ranking = ranker(llm, top) if top else []
    merge_ranking(tier1, ranking)

    meta = {
        "show": a.show or Path(a.scripts).stem,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "provider": cfg["llm"]["provider"], "model": llm.model,
        "rubric": rules_file if used_rules else "brief only",
        "episodes": len(episodes), "parse_warnings": parse_warnings,
        "hooks_found": hooks_found, "clips_judged": len(records),
        "rejected_before_judging": len(rejected),
        "tier_counts": {t: sum(r["tier"] == t for r in records) for t in (1, 2, 3)},
        "end_timestamps_adjusted": sum(bool(r["timing_note"]) for r in records),
        "llm_stats": llm.stats, "seconds_total": round(time.time() - t0, 1),
        "config": cfg,
    }
    write_outputs(Path(a.out), records, rejected, meta, tcfg["hand_check_sample"])
    print(f"\nDone in {meta['seconds_total']}s. Tiers {meta['tier_counts']}, "
          f"{len(rejected)} rejected, {meta['end_timestamps_adjusted']} end timestamps adjusted.")
    print(f"Model usage: {llm.stats}")
    print(f"Outputs in {a.out}/: shortlist.md, promos.json, promos.csv, failures.csv, "
          "hand_check.csv, run_log.json")


if __name__ == "__main__":
    main()
