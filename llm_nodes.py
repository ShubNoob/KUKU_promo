"""
LLM nodes - Phase 2 (Hook Hunter), Phase 4 (Judge & Trimmer), Phase 5 (Ranker).

All model calls go through one `LLM` class, so the model is set in config.yaml:
  provider: gemini             -> Google Gemini (google-genai SDK)
  provider: openai_compatible  -> Groq, OpenRouter, OpenAI, Ollama, ... (openai SDK)
  provider: mock               -> offline keyword heuristics, no API key (for testing)

Every response is cached on disk (keyed by model + prompt), so a rerun of the same
show costs zero calls. Calls are paced to `requests_per_minute` and retried with
exponential backoff on rate-limit / server errors.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, ValidationError

from parser import to_seconds


# =============================================================================
# Schemas (what the model must return)
# =============================================================================

class Hook(BaseModel):
    start_timestamp: str          # copied from the script, "MM:SS.ss"
    hook_type: str                # e.g. slap, public humiliation, identity reveal
    hook_description: str         # the moment in a few words


class HookList(BaseModel):
    hooks: list[Hook]


class PromoJudgeEvaluation(BaseModel):
    """The locked V3 schema, plus the 4 text fields marked [brief] that Kuku's brief
    requires for every promo (the hook, the cliffhanger, one line on why it works)."""
    is_valid_promo: bool
    # THE DYNAMIC CUT
    recommended_end_timestamp: str        # must be between +20s and +60s (Python checks)
    # NARRATIVE GRADING (1-10)
    hook_intensity_score: int
    cliffhanger_intensity_score: int
    middle_story_engagement_score: int
    # CONTEXT & QUALITY CONTROL
    is_self_understandable: bool          # the Amnesia Test
    missing_context_explanation: str
    # [brief] required outputs
    title: str
    hook_text: str
    cliffhanger_text: str
    why_it_works: str


class RankedClip(BaseModel):
    clip_id: str
    reason: str


class Ranking(BaseModel):
    ranking: list[RankedClip]


# =============================================================================
# Model wrapper
# =============================================================================

class LLM:
    def __init__(self, cfg: dict, cache_dir: str = ".cache"):
        self.cfg = cfg
        self.provider = cfg["provider"]
        self.model = cfg["model"] if self.provider != "mock" else "mock"
        self.min_interval = 60.0 / max(1, cfg.get("requests_per_minute", 8))
        self.max_retries = cfg.get("max_retries", 5)
        self.cache = Path(cache_dir) / self.provider / re.sub(r"[^\w.-]", "_", self.model)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._last = 0.0
        self._client = None
        self.stats = {"api_calls": 0, "cache_hits": 0, "failed_calls": 0, "retries": 0,
                      "input_tokens": 0, "output_tokens": 0, "seconds_waiting": 0.0}

    # ---- public --------------------------------------------------------------
    def structured(self, prompt: str, schema: type[BaseModel], tag: str,
                   mock: Optional[Callable[[], BaseModel]] = None) -> Optional[BaseModel]:
        """Return an instance of `schema`, or None if the model failed every retry."""
        if self.provider == "mock":
            return mock() if mock else None

        key = hashlib.sha256(f"{self.model}|{schema.__name__}|{prompt}".encode()).hexdigest()[:24]
        path = self.cache / f"{tag}-{key}.json"
        if path.exists():
            try:
                self.stats["cache_hits"] += 1
                return schema.model_validate_json(path.read_text())
            except ValidationError:
                path.unlink()                          # stale cache -> call again

        for attempt in range(self.max_retries):
            self._pace()
            try:
                text = self._call(prompt, schema)
                obj = schema.model_validate_json(_strip_fences(text))
                path.write_text(obj.model_dump_json(indent=2))
                return obj
            except ValidationError as e:               # model returned bad JSON
                last = f"invalid JSON: {str(e)[:200]}"
            except Exception as e:                     # rate limit, 5xx, network
                last = f"{type(e).__name__}: {str(e)[:200]}"
            self.stats["retries"] += 1
            wait = min(60, 2 ** attempt * 3) + random.random()
            if "429" in last or "RESOURCE_EXHAUSTED" in last or "rate" in last.lower():
                wait = max(wait, 30)                   # free-tier per-minute quota
            print(f"   [{tag}] attempt {attempt + 1} failed ({last}); waiting {wait:.0f}s")
            self.stats["seconds_waiting"] += wait
            time.sleep(wait)
        self.stats["failed_calls"] += 1
        return None

    # ---- internals -----------------------------------------------------------
    def _pace(self):
        gap = time.time() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
            self.stats["seconds_waiting"] += self.min_interval - gap
        self._last = time.time()

    def _key(self) -> str:
        env = self.cfg.get("api_key_env") or "GEMINI_API_KEY"
        key = os.environ.get(env)
        if not key:
            raise RuntimeError(f"Set the {env} environment variable (your API key).")
        return key

    def _call(self, prompt: str, schema: type[BaseModel]) -> str:
        self.stats["api_calls"] += 1
        if self.provider == "gemini":
            from google import genai
            from google.genai import types
            if self._client is None:
                self._client = genai.Client(api_key=self._key())
            r = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.cfg.get("temperature", 0.2),
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            u = getattr(r, "usage_metadata", None)
            if u:
                self.stats["input_tokens"] += u.prompt_token_count or 0
                self.stats["output_tokens"] += u.candidates_token_count or 0
            return r.text

        if self.provider == "openai_compatible":
            from openai import OpenAI
            if self._client is None:
                self._client = OpenAI(base_url=self.cfg.get("base_url"), api_key=self._key())
            r = self._client.chat.completions.create(
                model=self.model,
                temperature=self.cfg.get("temperature", 0.2),
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Reply with ONE JSON object matching this "
                     f"JSON schema, nothing else:\n{json.dumps(schema.model_json_schema())}"},
                    {"role": "user", "content": prompt},
                ],
            )
            if r.usage:
                self.stats["input_tokens"] += r.usage.prompt_tokens or 0
                self.stats["output_tokens"] += r.usage.completion_tokens or 0
            return r.choices[0].message.content

        raise ValueError(f"unknown provider: {self.provider}")


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return text


# =============================================================================
# Helpers to show the script to the model
# =============================================================================

def fmt(sec: float) -> str:
    m, s = divmod(sec, 60)
    return f"{int(m):02d}:{s:05.2f}"


def render(events: list[dict]) -> str:
    """One line per event, with the exact timestamps the model must copy."""
    out = []
    for e in events:
        span = f"[{fmt(e['start'])}-{fmt(e['end'])}]"
        if e["type"] == "dialogue":
            who = e["speaker"] + (" (off-screen)" if e["offscreen"] else "")
            emo = f" ({e['emotion']})" if e["emotion"] else ""
            out.append(f'{span} {who}{emo}: "{e["text"]}"')
        else:
            out.append(f"{span} {e['type'].upper()}: {e['text']}")
    return "\n".join(out)


# =============================================================================
# Phase 2 - Hook Hunter (one call per episode)
# =============================================================================

HOOK_PROMPT = """You find promo hooks in a vertical microdrama for a US audience on TikTok,
Instagram Reels and YouTube Shorts.

A HOOK is a moment that makes a scrolling stranger stop within 3 seconds:
a slap or physical attack, public humiliation, a betrayal line, an identity reveal,
a shocking accusation, someone being dragged/seized, a proposal or kiss out of nowhere.
Calm establishing shots, slow walking and small talk are NOT hooks.

Rules:
- Return {n_min} to {n_max} hooks, best first. Fewer is fine if the episode is weak.
- start_timestamp MUST be copied exactly from the START time of one line below (format MM:SS.ss).
- Pick the line or action that CAUSES the moment (the slap, the line), not the reaction shot after it.
- Each hook needs at least 20 seconds of story after it inside this episode
  (the episode ends at {end}), so do not pick anything after {latest}.
- Hooks should be at least 10 seconds apart.

EPISODE {episode}
Logline: {logline}

SCRIPT (each line: [start-end] who/what: text)
{script}
"""


def hook_hunter(llm: LLM, ep: dict, n_max: int = 5) -> list[Hook]:
    latest = max(0.0, ep["duration"] - 20)
    prompt = HOOK_PROMPT.format(
        n_min=min(3, n_max), n_max=n_max, episode=ep["episode"], logline=ep["logline"],
        end=fmt(ep["duration"]), latest=fmt(latest), script=render(ep["events"]))
    res = llm.structured(prompt, HookList, f"hooks-ep{ep['episode']:03d}",
                         mock=lambda: _mock_hooks(ep, n_max))
    return res.hooks if res else []


# =============================================================================
# Phase 4 - Promo Judge & Trimmer (one call per candidate clip)
# =============================================================================

JUDGE_PROMPT = """You are a promo editor for a vertical microdrama app. Below is a
{window:.0f}-second stretch of one episode. A promo clip STARTS at {start}
(the hook: {hook}). You decide where it ENDS and grade it.

1. THE CUT: choose recommended_end_timestamp = the END time of one line below, between
   {min_end} and {max_end} (20 to 60 seconds after the start). Cut at the most
   unresolved moment: a question with no answer, a threat, a reveal, a decision not yet
   made. Never cut right after the conflict is resolved.
2. GRADE 1-10 with these anchors:
   hook_intensity_score: 9-10 physical shock, public humiliation, identity reveal or
     betrayal in the first 3s; 5-6 tense talk but no shock; 1-3 calm/establishing.
   cliffhanger_intensity_score: 9-10 viewer NEEDS the next second (proposal before the
     answer, a threat, a reveal); 5-6 some tension left; 1-3 resolved, calm or mid-sentence.
   middle_story_engagement_score: 9-10 something new every few seconds, escalating;
     1-3 long repetitive or slow stretch a viewer would swipe away from.
3. AMNESIA TEST: the viewer has NEVER seen this show and knows only what is on screen
   in this clip. Speaker labels like "BLOND NOBLEMAN" are what the viewer sees.
   is_self_understandable = false if the clip depends on backstory, unexplained names
   or earlier events; say what is missing in missing_context_explanation ("" if none).
4. is_valid_promo = true only if the hook lands in the first 3 seconds, the ending is
   unresolved, and the amnesia test passes.
5. Write: title (2-5 words), hook_text (what the viewer sees/hears in the first 3s),
   cliffhanger_text (the final moment), why_it_works (ONE line).

EPISODE {episode} - clip window
{script}
"""


def judge(llm: LLM, clip: dict) -> Optional[PromoJudgeEvaluation]:
    s = clip["start"]
    prompt = JUDGE_PROMPT.format(
        window=clip["window_end"] - s, start=fmt(s), hook=clip["hook_description"],
        min_end=fmt(s + 20), max_end=fmt(min(s + 60, clip["window_end"])),
        episode=clip["episode"], script=render(clip["events"]))
    return llm.structured(prompt, PromoJudgeEvaluation, f"judge-{clip['clip_id']}",
                          mock=lambda: _mock_judge(clip))


# =============================================================================
# Phase 5 - Ranker (one call for the whole show)
# =============================================================================

RANK_PROMPT = """You decide which promo clips a microdrama app posts FIRST on TikTok,
Reels and Shorts to pull in NEW viewers who have never seen the show.

Order ALL {n} clips below from post-first to post-last. Prefer: the strongest first
3 seconds, an ending that forces a click, clips a stranger fully understands, early-story
setups over late twists that spoil the ending, and variety (do not stack near-identical
moments next to each other). Give each a one-line reason. Use the exact clip_id values.

CLIPS
{clips}
"""


def ranker(llm: LLM, clips: list[dict]) -> list[RankedClip]:
    rows = [json.dumps({
        "clip_id": c["clip_id"], "episode": c["episode"], "seconds": round(c["duration"], 1),
        "title": c["title"], "hook": c["hook_text"], "cliffhanger": c["cliffhanger_text"],
        "why": c["why_it_works"], "scores": [c["hook_intensity_score"],
                                             c["cliffhanger_intensity_score"],
                                             c["middle_story_engagement_score"]]})
        for c in clips]
    prompt = RANK_PROMPT.format(n=len(clips), clips="\n".join(rows))
    res = llm.structured(prompt, Ranking, "rank", mock=lambda: _mock_rank(clips))
    return res.ranking if res else []


# =============================================================================
# Mock model: keyword heuristics so the whole pipeline runs offline.
# Its output is only for testing the plumbing - it is NOT a real judgement.
# =============================================================================

HOT_EMOTIONS = ("shock", "furious", "scream", "outrag", "hysteric", "gasp", "betray",
                "devastat", "venom", "menac", "threat", "cruel", "vengeful", "desperate",
                "horrif", "accus", "defian", "deadly", "brutal")
HOT_ACTIONS = ("slap", "strike", "punch", "grab", "seize", "drag", "collapse", "kneel",
               "knee", "shatter", "gasp", "smash", "ring", "kiss", "tear", "reveal")


def _heat(e: dict) -> int:
    t = (e["emotion"] + " " + e["text"]).lower()
    h = sum(w in t for w in HOT_EMOTIONS) * 2 + sum(w in t for w in HOT_ACTIONS) * 2
    h += e["text"].count("!") + (e["type"] == "sfx")
    return h


def _mock_hooks(ep: dict, n: int) -> HookList:
    latest = ep["duration"] - 20
    pool = sorted((e for e in ep["events"] if e["type"] in ("dialogue", "action", "sfx")
                   and e["start"] <= latest), key=_heat, reverse=True)
    picked: list[dict] = []
    for e in pool:
        if _heat(e) and all(abs(e["start"] - p["start"]) >= 10 for p in picked):
            picked.append(e)
        if len(picked) == n:
            break
    return HookList(hooks=[Hook(start_timestamp=fmt(e["start"]), hook_type="[MOCK] keyword",
                                hook_description=e["text"][:80]) for e in picked])


def _mock_judge(clip: dict) -> PromoJudgeEvaluation:
    s = clip["start"]
    lines = [e for e in clip["events"] if e["type"] == "dialogue" and s + 20 <= e["end"] <= s + 60]
    if not lines:
        lines = [e for e in clip["events"] if s + 20 <= e["end"] <= s + 60] or clip["events"][-1:]
    end_ev = max(lines, key=lambda e: (_heat(e) + ("?" in e["text"]) * 2
                                        + ("..." in e["text"]), e["end"]))
    first = [e for e in clip["events"] if e["start"] <= s + 3]
    mid = [e for e in clip["events"] if s + 3 < e["start"] < end_ev["start"]]
    hook = min(10, 3 + sum(map(_heat, first)))
    cliff = min(10, 3 + _heat(end_ev) + ("?" in end_ev["text"]) * 2)
    middle = min(10, 2 + sum(map(_heat, mid)) // 3)
    return PromoJudgeEvaluation(
        is_valid_promo=hook >= 5 and cliff >= 5, recommended_end_timestamp=fmt(end_ev["end"]),
        hook_intensity_score=hook, cliffhanger_intensity_score=cliff,
        middle_story_engagement_score=middle, is_self_understandable=True,
        missing_context_explanation="", title="[MOCK] " + clip["hook_description"][:30],
        hook_text=clip["hook_description"], cliffhanger_text=end_ev["text"][:100],
        why_it_works="[MOCK] keyword heuristic, not a real judgement")


def _mock_rank(clips: list[dict]) -> Ranking:
    order = sorted(clips, key=lambda c: -(c["hook_intensity_score"]
                                          + c["cliffhanger_intensity_score"]))
    return Ranking(ranking=[RankedClip(clip_id=c["clip_id"], reason="[MOCK] score order")
                            for c in order])
