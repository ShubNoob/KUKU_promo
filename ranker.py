"""
Phase 5 - Sorter, Ranker merge, Hand-Check and output files.

Tiers (thresholds in config.yaml -> tiers):
  Tier 1  Ready to post  valid promo, passes the amnesia test, hook and cliffhanger
                         both strong, middle holds attention
  Tier 2  Editor splice  same as Tier 1 but the middle is slow: an editor should cut it
  Tier 3  Bad picks      anything else, with the reason it failed

Only Tier 1 goes to the LLM Ranker. The ranker's answer is checked: unknown clip IDs
are ignored and any clip it forgot is appended by score, so the ranking is always complete.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from framer import overlap
from llm_nodes import fmt


def build_record(clip: dict, v, end: float, note: str) -> dict:
    return {
        "clip_id": clip["clip_id"],
        "episode": clip["episode"],
        # Brief: "the episode and the start and end timestamps of every segment, in order"
        "segments": [{"episode": clip["episode"], "start": round(clip["start"], 2),
                      "end": round(end, 2), "start_ts": fmt(clip["start"]),
                      "end_ts": fmt(end)}],
        "start": clip["start"], "end": end, "duration": round(end - clip["start"], 2),
        "hook_event_id": clip["hook_event_id"], "hook_type": clip["hook_type"],
        "hook_at_second": round(clip["hook_at"] - clip["start"], 2),
        "title": v.title, "hook_text": v.hook_text, "cliffhanger_text": v.cliffhanger_text,
        "why_it_works": v.why_it_works,
        "hook_archetype": v.hook_archetype, "cliffhanger_archetype": v.cliffhanger_archetype,
        "hook_intensity_score": _score(v.hook_intensity_score),
        "cliffhanger_intensity_score": _score(v.cliffhanger_intensity_score),
        "middle_story_engagement_score": _score(v.middle_story_engagement_score),
        "is_valid_promo": v.is_valid_promo,
        "is_self_understandable": v.is_self_understandable,
        "missing_context_explanation": v.missing_context_explanation,
        "model_end_timestamp": v.recommended_end_timestamp,
        "timing_note": note,
    }


def _score(x) -> int:
    try:
        return max(1, min(10, int(x)))
    except (TypeError, ValueError):
        return 1


def assign_tier(r: dict, t: dict) -> None:
    reasons = []
    if not r["is_valid_promo"]:
        reasons.append("model marked it not a valid promo")
    if not r["is_self_understandable"]:
        reasons.append("fails amnesia test: " + (r["missing_context_explanation"] or "needs backstory"))
    if r["hook_intensity_score"] < t["tier1_min_hook"]:
        reasons.append(f"weak hook ({r['hook_intensity_score']}/10)")
    if r["cliffhanger_intensity_score"] < t["tier1_min_cliffhanger"]:
        reasons.append(f"weak cliffhanger ({r['cliffhanger_intensity_score']}/10)")
    if reasons:
        r["tier"], r["tier_reason"] = 3, "; ".join(reasons)
    elif r["middle_story_engagement_score"] <= t["splice_max_middle"]:
        r["tier"] = 2
        r["tier_reason"] = (f"slow middle ({r['middle_story_engagement_score']}/10): "
                            "editor should cut the dull stretch")
    else:
        r["tier"], r["tier_reason"] = 1, "ready to post"


def score(r: dict) -> int:
    return (r["hook_intensity_score"] + r["cliffhanger_intensity_score"]
            + r["middle_story_engagement_score"])


def mark_overlaps(records: list[dict], threshold: float = 0.5) -> None:
    """Flag near-duplicates (same episode, >50% overlap) so the human sees them."""
    for i, a in enumerate(records):
        a["overlaps_with"] = [b["clip_id"] for j, b in enumerate(records)
                              if i != j and overlap(a, b) > threshold]


def merge_ranking(tier1: list[dict], ranking) -> None:
    by_id = {r["clip_id"]: r for r in tier1}
    order, seen = [], set()
    for item in ranking:
        if item.clip_id in by_id and item.clip_id not in seen:
            by_id[item.clip_id]["rank_reason"] = item.reason
            by_id[item.clip_id]["rank_source"] = "llm"
            order.append(by_id[item.clip_id])
            seen.add(item.clip_id)
    for r in sorted(tier1, key=score, reverse=True):      # anything the model skipped
        if r["clip_id"] not in seen:
            r["rank_reason"], r["rank_source"] = "not ranked by model; placed by score", "score"
            order.append(r)
    for n, r in enumerate(order, 1):
        r["rank"] = n


# ----------------------------------------------------------------------------- outputs

CSV_COLS = ["rank", "tier", "clip_id", "episode", "start_ts", "end_ts", "duration", "title",
            "hook_text", "cliffhanger_text", "why_it_works", "hook_archetype",
            "cliffhanger_archetype", "hook_intensity_score",
            "cliffhanger_intensity_score", "middle_story_engagement_score",
            "is_self_understandable", "tier_reason", "rank_reason", "timing_note",
            "overlaps_with"]


def _row(r: dict) -> dict:
    row = {k: r.get(k, "") for k in CSV_COLS}
    row["start_ts"], row["end_ts"] = r["segments"][0]["start_ts"], r["segments"][0]["end_ts"]
    row["overlaps_with"] = " ".join(r.get("overlaps_with", []))
    return row


def write_outputs(out: Path, records: list[dict], rejected: list[dict], meta: dict,
                  sample_n: int, seed: int = 7) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tier1 = sorted([r for r in records if r["tier"] == 1], key=lambda r: r["rank"])
    others = sorted([r for r in records if r["tier"] != 1], key=lambda r: (r["tier"], -score(r)))
    ordered = tier1 + others

    (out / "promos.json").write_text(json.dumps(
        {"meta": meta, "promos": ordered, "rejected_hooks": rejected}, indent=2))

    with open(out / "promos.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(_row(r) for r in ordered)

    with open(out / "failures.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "clip_id", "episode", "reason"])
        w.writeheader()
        for r in rejected:
            w.writerow(r)
        for r in records:
            if r["tier"] == 3:
                w.writerow({"stage": "judge (tier 3)", "clip_id": r["clip_id"],
                            "episode": r["episode"], "reason": r["tier_reason"]})
            if r["timing_note"]:
                w.writerow({"stage": "timestamp check", "clip_id": r["clip_id"],
                            "episode": r["episode"], "reason": r["timing_note"]})

    # Hand-check sheet: the top-ranked clips, WITHOUT the model's scores, so the human
    # judges first. Fill human_good (y/n) + human_reason, then: python run.py --score <file>
    sample = tier1[:sample_n]
    rng = random.Random(seed)
    rng.shuffle(sample)
    with open(out / "hand_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "episode", "start_ts", "end_ts",
                                          "hook_text", "cliffhanger_text", "human_good",
                                          "human_reason"])
        w.writeheader()
        for r in sample:
            w.writerow({"clip_id": r["clip_id"], "episode": r["episode"],
                        "start_ts": r["segments"][0]["start_ts"],
                        "end_ts": r["segments"][0]["end_ts"], "hook_text": r["hook_text"],
                        "cliffhanger_text": r["cliffhanger_text"], "human_good": "",
                        "human_reason": ""})

    _write_shortlist(out / "shortlist.md", tier1, others, rejected, meta)
    (out / "run_log.json").write_text(json.dumps(meta, indent=2))


def _write_shortlist(path: Path, tier1, others, rejected, meta) -> None:
    L = [f"# Promo shortlist - {meta['show']}", "",
         f"Model: `{meta['model']}` · episodes: {meta['episodes']} · "
         f"hooks found: {meta['hooks_found']} · clips judged: {meta['clips_judged']} · "
         f"Tier 1: {len(tier1)} · Tier 2: {sum(r['tier'] == 2 for r in others)} · "
         f"Tier 3: {sum(r['tier'] == 3 for r in others)} · rejected before judging: {len(rejected)}",
         "", "## Tier 1 - post in this order", ""]
    for r in tier1:
        s = r["segments"][0]
        L += [f"### {r['rank']}. {r['title']}  ({r['clip_id']})",
              f"- **Episode {s['episode']}, {s['start_ts']} → {s['end_ts']}** ({r['duration']:.0f}s)",
              f"- **Hook:** {r['hook_text']}",
              f"- **Cliffhanger:** {r['cliffhanger_text']}",
              f"- **Why it works:** {r['why_it_works']}",
              f"- **Why this rank:** {r.get('rank_reason', '')}",
              f"- Patterns: {r['hook_archetype']} → {r['cliffhanger_archetype']}",
              f"- Scores hook/cliff/middle: {r['hook_intensity_score']}/"
              f"{r['cliffhanger_intensity_score']}/{r['middle_story_engagement_score']}"
              + (f" · overlaps {', '.join(r['overlaps_with'])}" if r.get("overlaps_with") else ""),
              ""]
    t2 = [r for r in others if r["tier"] == 2]
    if t2:
        L += ["## Tier 2 - needs an editor's splice", ""]
        for r in t2:
            s = r["segments"][0]
            L.append(f"- {r['clip_id']} · Ep {s['episode']} {s['start_ts']}→{s['end_ts']} · "
                     f"{r['title']} · {r['tier_reason']}")
        L.append("")
    t3 = [r for r in others if r["tier"] == 3]
    L += ["## Tier 3 - bad picks (and why)", ""]
    L += [f"- {r['clip_id']} · {r['title']} · {r['tier_reason']}" for r in t3] or ["- none"]
    if rejected:
        L += ["", "## Rejected before judging (timestamp / length checks)", ""]
        L += [f"- {r['clip_id']} · {r['reason']}" for r in rejected]
    path.write_text("\n".join(L) + "\n")


def score_hand_check(path: str) -> None:
    rows = list(csv.DictReader(open(path)))
    done = [r for r in rows if r["human_good"].strip().lower() in ("y", "n", "yes", "no")]
    if not done:
        print("No rows labelled yet: fill human_good with y or n.")
        return
    good = sum(r["human_good"].strip().lower().startswith("y") for r in done)
    print(f"Labelled {len(done)}/{len(rows)} top-ranked clips.")
    print(f"Human agreed with the tool on {good}/{len(done)} = {good / len(done):.0%} "
          "(precision of Tier 1 picks).")
    bad = [r for r in done if r["human_good"].strip().lower().startswith("n")]
    if bad:
        print("\nWhere the tool picked badly:")
        for r in bad:
            print(f"  {r['clip_id']} (ep {r['episode']} {r['start_ts']}-{r['end_ts']}): "
                  f"{r['human_reason'] or '(no reason given)'}")
