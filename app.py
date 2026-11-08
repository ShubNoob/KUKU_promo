"""
Streamlit dashboard for the Promo Extractor.

    streamlit run app.py

- View results: upload a promos.json from a run (or it loads outputs/promos.json).
- Run engine: upload a zip of episode scripts and run the pipeline on a few episodes.
  A full 50-episode run takes a while on the free tier - use the Colab notebook for that.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Promo Extractor", page_icon="🎬", layout="wide")
HERE = Path(__file__).parent


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("🎬 Promo Extractor")
mode = st.sidebar.radio("Mode", ["View results", "Run engine"])
data = None

if mode == "View results":
    up = st.sidebar.file_uploader("promos.json from a run", type="json")
    if up:
        data = json.load(up)
    elif (HERE / "outputs" / "promos.json").exists():
        data = json.loads((HERE / "outputs" / "promos.json").read_text())
        st.sidebar.caption("Showing outputs/promos.json")

else:
    zip_up = st.sidebar.file_uploader("Episode scripts (.zip of .txt files)", type="zip")
    episodes = st.sidebar.text_input("Episodes", "1-3", help="e.g. 1-3 or 1,5,9. Keep it small here.")
    provider = st.sidebar.selectbox("Provider", ["gemini", "openai_compatible", "mock"])
    model = st.sidebar.text_input("Model", "gemini-flash-latest")
    try:                                    # Streamlit Cloud: put the key in app secrets
        default_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:                       # no secrets file when running locally
        default_key = ""
    key = st.sidebar.text_input("API key", type="password", value=default_key)
    if st.sidebar.button("RUN ENGINE", type="primary", disabled=zip_up is None):
        work = Path(tempfile.mkdtemp())
        zpath = work / "scripts.zip"
        zpath.write_bytes(zip_up.getvalue())
        env = {**os.environ, "GEMINI_API_KEY": key} if key else dict(os.environ)
        cmd = [sys.executable, str(HERE / "run.py"), str(zpath), "--out", str(work / "out"),
               "--provider", provider, "--model", model, "--show", Path(zip_up.name).stem]
        if episodes.strip():
            cmd += ["--episodes", episodes.strip()]
        log = st.empty()
        lines = []
        with st.spinner("Running... (free-tier pacing means ~8 calls a minute)"):
            p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                lines.append(line.rstrip())
                log.code("\n".join(lines[-15:]))
            p.wait()
        if p.returncode == 0 and (work / "out" / "promos.json").exists():
            st.session_state["result"] = json.loads((work / "out" / "promos.json").read_text())
        else:
            st.error("Run failed - see the log above.")
    data = st.session_state.get("result")


# ----------------------------------------------------------------------------- main
if not data:
    st.title("Promo Extractor")
    st.write("Upload a `promos.json` to view results, or switch to **Run engine**.")
    st.stop()

meta, promos, rejected = data["meta"], data["promos"], data.get("rejected_hooks", [])
st.title(f"Promos · {meta.get('show', '')}")
c = st.columns(5)
c[0].metric("Episodes", meta["episodes"])
c[1].metric("Clips judged", meta["clips_judged"])
c[2].metric("Tier 1", meta["tier_counts"].get("1", meta["tier_counts"].get(1, 0)))
c[3].metric("Rejected by checks", meta["rejected_before_judging"])
c[4].metric("Model", meta["model"])

st.download_button("Download promos.json", json.dumps(data, indent=2), "promos.json")


def card(r):
    s = r["segments"][0]
    head = f"#{r['rank']} · " if r.get("rank") else ""
    with st.container(border=True):
        st.markdown(f"**{head}{r['title']}** · `{r['clip_id']}` · Episode {s['episode']} · "
                    f"**{s['start_ts']} → {s['end_ts']}** ({r['duration']:.0f}s)")
        st.markdown(f"🪝 **Hook:** {r['hook_text']}  \n"
                    f"⏸️ **Cliffhanger:** {r['cliffhanger_text']}  \n"
                    f"💡 **Why it works:** {r['why_it_works']}")
        cols = st.columns(3)
        cols[0].caption(f"Hook {r['hook_intensity_score']}/10")
        cols[1].caption(f"Cliffhanger {r['cliffhanger_intensity_score']}/10")
        cols[2].caption(f"Middle {r['middle_story_engagement_score']}/10")
        if r.get("rank_reason"):
            st.caption(f"Rank reason: {r['rank_reason']}")
        if r.get("timing_note"):
            st.caption(f"⚠️ {r['timing_note']}")
        if r.get("overlaps_with"):
            st.caption(f"Overlaps with {', '.join(r['overlaps_with'])}")


t1, t2, t3 = st.tabs(["🏆 Tier 1: ready to post", "✂️ Tier 2: editor splice", "🗑️ Tier 3: bad picks"])
with t1:
    for r in sorted([r for r in promos if r["tier"] == 1], key=lambda r: r["rank"]):
        card(r)
with t2:
    for r in [r for r in promos if r["tier"] == 2]:
        st.warning(r["tier_reason"])
        card(r)
with t3:
    for r in [r for r in promos if r["tier"] == 3]:
        st.error(r["tier_reason"])
        card(r)
    if rejected:
        st.subheader("Rejected before judging")
        st.table([{k: x[k] for k in ("clip_id", "stage", "reason")} for x in rejected])
