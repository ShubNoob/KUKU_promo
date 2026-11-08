# Promo Extractor — Kuku AI Generalist Assignment, Problem 1

Reads a microdrama screenplay and finds 20–60 second promo clips that open on a hook
(first 3 seconds), cut on a cliffhanger, and make sense to someone who has never seen
the show. Outputs every promo's episode, start/end timestamps, hook, cliffhanger, a
one-line reason, a tier, and a posting order.

## How it works

| Phase | Who | What |
|---|---|---|
| 1 Parse | Python | `.txt` script → timeline of camera / action / dialogue / SFX events, times in seconds, shot cut points |
| 2 Hook Hunter | LLM, 1 call per episode | reads the whole episode, returns 3–5 hook start times |
| 3 Framer | Python | hook time T → window T..T+60s (never past the episode end); rejects timestamps that match no line |
| 4 Judge & Trimmer | LLM, 1 call per clip | picks the cut (20–60s), scores hook / cliffhanger / middle, runs the amnesia test, writes hook, cliffhanger, why |
| 4b Timestamp check | Python | snaps the chosen end to a real line end or shot cut inside 20–60s; logs every adjustment |
| 5 Sorter + Ranker | Python + LLM, 1 call | Tier 1 ready / Tier 2 editor splice / Tier 3 bad picks; LLM orders Tier 1 for posting |
| 5b Hand check | Human | labels the top picks blind; `--score` reports how often the tool was right |

The model never owns time: every timestamp it returns is checked against the script,
and clip length is enforced in code.

## Run it

**Colab (recommended):** open `notebook.ipynb` in Colab, add `GEMINI_API_KEY` in the
Secrets panel, upload the scripts zip, run all cells.

**Locally:**
```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...
python tests/test_checks.py                       # 9 checks, no API calls
python run.py scripts.zip --provider mock         # offline dry run
python run.py scripts.zip --episodes 1-3          # quick real run
python run.py scripts.zip --out outputs           # full show
python run.py --score outputs/hand_check.csv      # after labelling the sheet
streamlit run app.py                              # dashboard
```

## Swap the model

Edit the `llm:` block in `config.yaml`, or pass `--model` / `--provider`:
- `provider: gemini`, `model:` any Gemini model ID your key can use
- `provider: openai_compatible` with `base_url` + `api_key_env` → Groq, OpenRouter,
  OpenAI, Ollama, or any API that speaks the OpenAI chat format
- `provider: mock` → offline keyword heuristics, for testing only

## Outputs (`outputs/`)

| File | What |
|---|---|
| `shortlist.md` | Tier 1 in posting order, then Tier 2 and Tier 3 with reasons |
| `promos.json` | every judged clip with segments, scores, notes, and the run metadata |
| `promos.csv` | the same, one row per clip |
| `failures.csv` | rejected hooks, Tier 3 reasons, adjusted timestamps |
| `hand_check.csv` | top picks without model scores, for blind human labelling |
| `run_log.json` | counts, API calls, cache hits, tokens, time taken, config used |

## Cost and scale

One show of 50 episodes ≈ 50 Hook Hunter calls + up to 250 Judge calls + 1 Ranker call,
each prompt ~1–2k tokens, so well under 1M tokens. On the free tier the run is limited
by requests per minute (set `requests_per_minute` in `config.yaml`). Every response is
cached in `.cache/`, so re-runs and interrupted runs don't repeat calls.

## Known limits

- Works from the script, not the video: it can't judge acting, pacing on screen or audio.
- One continuous segment per promo; it doesn't stitch moments across episodes yet.
- Clips never cross episode boundaries; a hook in the last 20s of an episode is dropped.
- The amnesia test and scores are the model's judgement, so check them with the hand-check sheet.

## Confidentiality

Kuku's scripts are confidential. `.gitignore` keeps scripts, outputs and the cache out of
the repo; upload the zip at run time instead.
