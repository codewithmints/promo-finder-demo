"""Promo Finder — Streamlit app over the precomputed bundle (app_data/), no API key needed.

  streamlit run app.py                    # local: full screenplay excerpts + reviews feed the real feedback store
  PUBLIC_MODE=1 streamlit run app.py      # public: no screenplay excerpts; reviews go to a separate demo file
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "app_data"
PUBLIC = os.getenv("PUBLIC_MODE", "0") == "1" or (ROOT / ".public_mode").exists()   # the public bundle ships the marker
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#8a8984"
TIER_TEXT = {"A": "Post first", "B": "Good, with caveats", "C": "Post last / needs review"}
REASONS = ["weak_hook", "no_cliffhanger", "resolved", "confusing", "needs_context", "spoiler", "bad_boundary",
           "low_emotion", "boring", "duplicate", "other"]

st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1280px;}
.pf-card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 14px 16px; margin-bottom: 12px;}
.pf-rank {font-weight: 700; font-size: 1.05rem;}
.pf-meta {opacity: .7; font-size: .85rem;}
.pf-tier {display:inline-block; padding: 1px 8px; border-radius: 999px; font-size: .75rem; font-weight: 600;
          border: 1px solid rgba(128,128,128,.4); margin-left: 6px;}
.pf-label {font-size: .72rem; text-transform: uppercase; letter-spacing: .04em; opacity: .6; margin-top: 6px;}
.pf-chip {display:inline-block; padding: 1px 7px; margin: 2px 4px 2px 0; border-radius: 6px; font-size: .75rem;
          background: rgba(128,128,128,.14);}
.pf-flow {display:flex; flex-wrap: wrap; gap: 6px; align-items:center; font-size: .9rem;}
.pf-flow span.step {border: 1px solid rgba(128,128,128,.35); border-radius: 8px; padding: 4px 10px;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_data
def load_index():
    f = DATA / "shows.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


@st.cache_data
def load_show(sid: str):
    d = DATA / sid
    j = lambda n: json.loads((d / n).read_text(encoding="utf-8")) if (d / n).exists() else None
    rep = (d / "eval_report.md").read_text(encoding="utf-8") if (d / "eval_report.md").exists() else ""
    return {"promos": j("promos.json") or [], "manifest": j("manifest.json") or {}, "cuts": j("cuts.json") or {},
            "feedback": j("feedback.json") or {}, "learning": j("learning.json") or {}, "report": rep}


@st.cache_data
def load_excerpts(sid: str):
    """Full screenplay excerpts: LOCAL ONLY (never in app_data, never in public mode)."""
    if PUBLIC:
        return {}
    try:
        from promo.show import get_show
        f = get_show(sid).outputs_dir / "promos_ranked.json"
        return {p["candidate_id"]: p.get("screenplay_excerpt", []) for p in json.loads(f.read_text(encoding="utf-8"))}
    except Exception:
        return {}


def ts(t: float) -> str:
    m, s = divmod(round(t, 2), 60)
    return f"{int(m):02d}:{s:05.2f}"


def seg_str(p):
    return " + ".join(f"{s['start']}–{s['end']}" for s in p["segments"])


def spk(x):
    return (f"{x['speaker']}: " if x.get("speaker") else "") + x["text"]


def promo_card(p, compact=False):
    tier = p["tier"]
    html = (f"<div class='pf-card'><span class='pf-rank'>#{p['rank']}</span> "
            f"<span class='pf-meta'>Episode {p['episode']} · {seg_str(p)} · {p['duration_s']}s · score {p['final_score']:.3f}"
            f"{' · multi-segment' if len(p['segments']) > 1 else ''}</span>"
            f"<span class='pf-tier'>Tier {tier} · {TIER_TEXT[tier]}</span>"
            f"<div class='pf-label'>Hook ({p['hook']['time']})</div><div>{spk(p['hook'])}</div>"
            f"<div class='pf-label'>Cliffhanger ({p['cliffhanger']['time']})</div><div>{spk(p['cliffhanger'])}</div>")
    if not compact:
        html += f"<div class='pf-label'>Why it works</div><div>{p['why_it_works']}</div>"
    st.markdown(html + "</div>", unsafe_allow_html=True)


def bar(df, x, y, title=None, color=BLUE, sort="-x", height=None, tooltip=None):
    c = alt.Chart(df).mark_bar(cornerRadiusEnd=4, color=color, size=14).encode(
        x=alt.X(x, title=None, axis=alt.Axis(tickMinStep=1, format="d")), y=alt.Y(y, sort=sort, title=None),
        tooltip=tooltip or [y, x])
    if title:
        c = c.properties(title=title)
    return c.properties(height=height or alt.Step(24))


def excerpt_block(p):
    ex = load_excerpts(SID).get(p["candidate_id"])
    if PUBLIC:
        st.info("Full screenplay excerpts are hidden in the public deployment. Timestamps, hook and cliffhanger lines are "
                "shown above; the complete text is visible when the app runs locally.")
    elif ex:
        seg = None; lines = []
        for x in ex:
            if x.get("segment") != seg and len(p["segments"]) > 1:
                lines.append(f"--- segment {x.get('segment')} ---")
            seg = x.get("segment")
            lines.append(f"[{x['start']}] {x['label']}: {x['text']}")
        st.code("\n".join(lines), language=None)
    else:
        st.caption("No local excerpt available (run scripts/run_final.py for this show).")


# ---------------------------------------------------------------- sidebar
idx = load_index()
if not idx or not idx.get("shows"):
    st.error("No app data. Run `python scripts/export_app_data.py --all` first.")
    st.stop()
with st.sidebar:
    st.markdown("### 🎬 Promo Finder")
    shows = idx["shows"]
    ids = list(shows)
    SID = st.selectbox("Show", ids, index=ids.index(idx["default_show"]) if idx["default_show"] in ids else 0,
                       format_func=lambda s: shows[s]["show_name"])
    page = st.radio("Page", ["Dashboard", "Promo Explorer", "Human Review", "Evaluation", "Failure Analysis", "Feedback Learning"],
                    label_visibility="collapsed")
    st.divider()
    st.caption(("🔒 Public mode: screenplay excerpts hidden; reviews are saved to a demo file that never changes the ranking."
                if PUBLIC else "🖥️ Local mode: full excerpts; reviews feed the real feedback store."))

D = load_show(SID)
P, M = D["promos"], D["manifest"]
LRN, FB = D["learning"], D["feedback"]

# ---------------------------------------------------------------- pages
if page == "Dashboard":
    st.title(shows[SID]["show_name"])
    st.caption(f"Ranking `{M.get('ranking_version')}` · weights `{M.get('weights_version')}` · generated {M.get('generated')} · "
               "one global ranking across the whole show (#1 = post first)")
    st.markdown("<div class='pf-flow'>" + " → ".join(f"<span class='step'>{t}</span>" for t in [
        f"{M.get('episodes')} episodes", f"{M.get('continuous_candidates')} candidates + {M.get('multi_segment_candidates')} multi-segment",
        f"{M.get('duplicates_removed')} duplicates merged", f"{M.get('ranked_promos')} ranked", "human review", "feedback",
        "calibrated weights"]) + "</div>", unsafe_allow_html=True)
    st.write("")
    tiers = M.get("tiers", {})
    c = st.columns(5)
    c[0].metric("Ranked promos", M.get("ranked_promos"))
    c[1].metric("Tier A · post first", tiers.get("A", 0))
    c[2].metric("Multi-segment ranked", sum(len(p["segments"]) > 1 for p in P))
    c[3].metric("Feedback records", FB.get("summary", {}).get("records", 0))
    c[4].metric("Active weights", LRN.get("active_label", "global/v0"))
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Post first: top 10")
        for p in P[:10]:
            promo_card(p, compact=True)
    with right:
        st.subheader("Where the top promos come from")
        top = Counter(p["episode"] for p in P[:50])
        df = pd.DataFrame({"episode": [f"Ep {e}" for e in top], "promos in top 50": list(top.values())})
        st.altair_chart(bar(df.sort_values("promos in top 50", ascending=False).head(15), "promos in top 50:Q", "episode:N",
                            height=360), use_container_width=True)
        st.caption("No per-episode quota: strong episodes place several promos, quiet ones none.")
        dft = pd.DataFrame({"tier": [f"{t} · {TIER_TEXT[t]}" for t in "ABC"], "promos": [tiers.get(t, 0) for t in "ABC"]})
        st.altair_chart(bar(dft, "promos:Q", "tier:N", sort=None), use_container_width=True)

elif page == "Promo Explorer":
    st.title("Promo Explorer")
    f1, f2, f3, f4 = st.columns([1, 2, 1, 2])
    tier_f = f1.multiselect("Tier", ["A", "B", "C"], default=["A", "B", "C"])
    eps_all = sorted({p["episode"] for p in P})
    ep_f = f2.multiselect("Episodes", eps_all, placeholder="All episodes")
    type_f = f3.selectbox("Type", ["All", "Continuous", "Multi-segment"])
    q = f4.text_input("Search hook / cliffhanger", "")
    sel = [p for p in P if p["tier"] in tier_f and (not ep_f or p["episode"] in ep_f)
           and (type_f == "All" or (type_f == "Multi-segment") == (len(p["segments"]) > 1))
           and (not q or q.lower() in (p["hook"]["text"] + p["cliffhanger"]["text"]).lower())]
    st.caption(f"{len(sel)} of {len(P)} promos")
    df = pd.DataFrame([{"rank": p["rank"], "tier": p["tier"], "episode": p["episode"], "segments": seg_str(p),
                        "dur (s)": p["duration_s"], "score": round(p["final_score"], 3), "hook": spk(p["hook"]),
                        "cliffhanger": spk(p["cliffhanger"])} for p in sel])
    st.dataframe(df, hide_index=True, use_container_width=True, height=300)
    if sel:
        pick = st.selectbox("Inspect promo", [p["promo_id"] for p in sel],
                            format_func=lambda i: next(f"#{p['rank']} · Ep {p['episode']} · {p['hook']['text'][:70]}" for p in sel if p["promo_id"] == i))
        p = next(x for x in sel if x["promo_id"] == pick)
        a, b = st.columns([3, 2])
        with a:
            promo_card(p)
            st.markdown(f"**Tier {p['tier']}** — {p['tier_reason']}")
            if p["flags"]:
                st.markdown(" ".join(f"<span class='pf-chip'>{f}</span>" for f in p["flags"]), unsafe_allow_html=True)
            st.markdown("**Screenplay excerpt**")
            excerpt_block(p)
        with b:
            st.markdown("**Score breakdown**")
            parts = {**p["score_breakdown"]["generator"], **{k: v for k, v in p["score_breakdown"]["penalties"].items()}}
            dfc = pd.DataFrame([{"term": k, "contribution": v} for k, v in parts.items() if abs(v) >= 0.0005])
            dfc["effect"] = dfc.contribution.map(lambda v: "adds" if v > 0 else "subtracts")
            ch = alt.Chart(dfc).mark_bar(cornerRadiusEnd=4, size=14).encode(
                x=alt.X("contribution:Q", title=None), y=alt.Y("term:N", sort="-x", title=None),
                color=alt.Color("effect:N", scale=alt.Scale(domain=["adds", "subtracts"], range=[BLUE, ORANGE]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["term", alt.Tooltip("contribution:Q", format="+.3f")]).properties(height=alt.Step(26))
            st.altair_chart(ch, use_container_width=True)
            st.caption(p["score_breakdown_text"])
            cv = p.get("cold_viewer")
            if cv:
                st.markdown("**Cold Viewer (LLM as a first-time viewer)**")
                st.markdown(f"Verdict **{cv['verdict']}** · clarity {cv['clarity']}/5 · hook {cv['hook_3s']}/5 · "
                            f"context dependency {cv['context_dependency']}/5 · ending tension {cv['ending_tension']}/5")
                if cv.get("open_question") and cv["open_question"].lower() != "none":
                    st.caption(f"Open question: {cv['open_question']}")
                st.caption("Used only as a penalty: on human gold it rated almost every clip 'post'.")

elif page == "Human Review":
    st.title("Human Review")
    reviewer = st.text_input("Reviewer", value="reviewer" if not PUBLIC else "guest")
    if PUBLIC:
        from promo.review_public import add_public_review, load_public
        done = {r["candidate_id"] for r in load_public(SID) if r["reviewer"] == reviewer}
    else:
        from promo.show import get_show
        from promo.review import ReviewContext
        from promo import feedback as Fb

        @st.cache_resource
        def ctx_for(sid):
            return ReviewContext(get_show(sid))
        ctx = ctx_for(SID)
        done = ctx.reviewed(reviewer)
    queue = [p for p in P if p["candidate_id"] not in done]
    st.caption(f"{len(queue)} promos not yet reviewed by {reviewer} · best-ranked first")
    if not queue:
        st.success("Everything reviewed.")
        st.stop()
    pick = st.selectbox("Promo", [p["promo_id"] for p in queue[:100]],
                        format_func=lambda i: next(f"#{p['rank']} · Ep {p['episode']} · {p['hook']['text'][:70]}" for p in queue if p["promo_id"] == i))
    p = next(x for x in queue if x["promo_id"] == pick)
    a, b = st.columns([3, 2])
    with a:
        promo_card(p)
        excerpt_block(p)
    with b:
        with st.form(f"review_{pick}"):
            decision = st.radio("Decision", ["Accept", "Reject"], horizontal=True)
            quality = st.select_slider("Quality (1 = great, 3 = weak)", [1, 2, 3], value=2)
            reason = st.selectbox("Rejection reason (required to reject)", [""] + REASONS)
            note = st.text_area("Note (optional)", max_chars=200)
            st.markdown("**Boundaries** (seconds; snapped to the nearest legal cut within 2s on save)")
            corrected = []
            for i, s in enumerate(p["segments"]):
                c1, c2 = st.columns(2)
                s0 = c1.number_input(f"Seg {i + 1} start", value=float(s["start_s"]), step=0.1, format="%.2f", key=f"s{pick}{i}")
                s1 = c2.number_input(f"Seg {i + 1} end", value=float(s["end_s"]), step=0.1, format="%.2f", key=f"e{pick}{i}")
                corrected.append([s0, s1])
            total = sum(e - s for s, e in corrected)
            st.caption(f"Total {total:.1f}s (must be 20–60s) · original {seg_str(p)}")
            saved = st.form_submit_button("Save review", type="primary")
        if saved:
            changed = any(abs(c[0] - s["start_s"]) > 1e-6 or abs(c[1] - s["end_s"]) > 1e-6 for c, s in zip(corrected, p["segments"]))
            kw = dict(accepted=decision == "Accept", quality=quality if decision == "Accept" else None,
                      reason=reason if decision == "Reject" else "", note=note,
                      corrected_segments=corrected if changed else None, reviewer=reviewer)
            try:
                if PUBLIC:
                    from promo.feedback import ReviewError
                    rec = add_public_review(SID, p, D["cuts"], **kw)
                else:
                    from promo.feedback import ReviewError
                    rec = ctx.review(pick, **kw)
                msg = f"Saved: {'accepted' if rec['accepted'] else 'rejected'}"
                if rec.get("corrected_segments"):
                    msg += f" · boundaries snapped to {' + '.join(ts(s) + '–' + ts(e) for s, e in rec['corrected_segments'])}"
                if rec.get("evaluation_only") and rec.get("evaluation_only_reason", "").startswith("held-out"):
                    msg += " · held-out episode: stored for evaluation only"
                st.success(msg)
                st.cache_data.clear()
            except ReviewError as e:
                st.error(f"Not saved: {e}")
    st.divider()
    st.subheader("Recent reviews")
    recs = load_public(SID) if PUBLIC else [r for r in Fb.load_all() if r["show_id"] == SID and r["source"] == "review"]
    if recs:
        st.dataframe(pd.DataFrame([{"when": r["created_at"], "reviewer": r["reviewer"], "promo": r["promo_id"],
                                    "episode": r["episode_number"], "decision": "accept" if r["accepted"] else "reject",
                                    "quality": r["quality"], "reason": r["reason"], "corrected": bool(r["corrected_segments"]),
                                    "evaluation only": r["evaluation_only"]} for r in recs[::-1][:50]]),
                     hide_index=True, use_container_width=True)
    else:
        st.caption("No reviews yet.")

elif page == "Evaluation":
    st.title("Evaluation")
    rep = D["report"]
    if not rep:
        st.info("No evaluation report for this show (run scripts/evaluate.py).")
    else:
        parts = re.split(r"\n(?=## )", rep)
        head, secs = parts[0], parts[1:]
        st.markdown(head.replace("# ", "", 1) if head.startswith("# ") else head)
        names = [s.split("\n", 1)[0].lstrip("# ").strip() for s in secs]
        short = [re.split(r"[:(]", re.sub(r"^\d+\.\s*", "", n).replace("`", ""))[0].strip()[:30] for n in names]
        tabs = st.tabs(short)
        for t, s in zip(tabs, secs):
            with t:
                st.markdown(s.split("\n", 1)[1] if "\n" in s else s)

elif page == "Failure Analysis":
    st.title("Where the tool picks badly")
    n = st.slider("Look at the top N promos", 10, len(P), min(50, len(P)), step=10)
    fl = Counter(f for p in P[:n] for f in p["flags"])
    a, b = st.columns(2)
    with a:
        st.markdown(f"**Automatic flags in the top {n}**")
        if fl:
            st.altair_chart(bar(pd.DataFrame({"flag": list(fl), "promos": list(fl.values())}), "promos:Q", "flag:N"),
                            use_container_width=True)
        else:
            st.caption("No flags.")
    with b:
        rs = FB.get("summary", {}).get("reasons", {})
        st.markdown("**Why humans rejected promos (feedback)**")
        if rs:
            st.altair_chart(bar(pd.DataFrame({"reason": list(rs), "rejections": list(rs.values())}), "rejections:Q", "reason:N",
                                color=ORANGE), use_container_width=True)
        else:
            st.caption("No rejections recorded yet.")
    pat = FB.get("patterns")
    if pat:
        st.markdown("**How rejected promos differ from accepted ones, feature by feature**")
        rows = [{"reason": r, "count": v["count"], "largest differences vs accepted": ", ".join(f"{k} {d:+.2f}" for k, d in
                 v["largest_feature_differences_vs_accepted"])} for r, v in pat["reasons"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        fm = pd.DataFrame([{"feature": k, "accepted": v["accepted"], "rejected": v["rejected"]} for k, v in pat["feature_means"].items()
                           if v["accepted"] is not None])
        with st.expander("Feature means, accepted vs rejected"):
            st.dataframe(fm.round(3), hide_index=True, use_container_width=True)
    m = re.search(r"## 7\. Where the tool picks badly.*?(?=\n## |\Z)", D["report"], re.S)
    if m:
        st.markdown("**Concrete failure cases (from the evaluation)**")
        st.markdown(m.group(0).split("\n", 1)[1])
    st.markdown(f"**Flagged promos in the top {n}**")
    st.dataframe(pd.DataFrame([{"rank": p["rank"], "tier": p["tier"], "episode": p["episode"], "segments": seg_str(p),
                                "flags": ", ".join(p["flags"]), "tier reason": p["tier_reason"]} for p in P[:n] if p["flags"]]),
                 hide_index=True, use_container_width=True)

elif page == "Feedback Learning":
    st.title("Feedback learning")
    st.markdown("Reviews become preference pairs (accepted > rejected, quality 1 > 3, corrected > original boundaries). A small "
                "linear model re-weights the same interpretable features the ranker uses, stays close to the current weights "
                "when feedback is scarce, and is **promoted only if it passes the gate**. Held-out episodes are never used. "
                "No model fine-tuning; the LLM does not learn from reviews.")
    s = FB.get("summary", {})
    g = LRN.get("gate", {})
    recs = [r for r in FB.get("records", []) if not r["evaluation_only"]]
    acc, rej = sum(r["accepted"] for r in recs), sum(not r["accepted"] for r in recs)
    n_eps = len({r["episode_number"] for r in recs})
    c = st.columns(4)
    c[0].metric("Active weights", LRN.get("active_label", "global/v0"))
    c[1].metric("Training-usable feedback", s.get("training_usable", 0), help="held-out episodes excluded")
    c[2].metric("Evaluation-only (held-out)", s.get("evaluation_only", 0))
    c[3].metric("Accepted / rejected", f"{acc} / {rej}")
    st.caption("Sources: " + (", ".join(f"{k} {v}" for k, v in s.get("by_source", {}).items()) or "none"))
    st.markdown("**Promotion gate: data requirements**")
    for label, have, need in (("accepted", acc, g.get("min_accepted", 60)), ("rejected", rej, g.get("min_rejected", 20)),
                              ("episodes", n_eps, g.get("min_episodes", 8))):
        st.progress(min(1.0, have / need), text=f"{label}: {have} / {need}")
    st.caption(f"Then: episode-grouped cross-validated pair accuracy gain ≥ {g.get('min_auc_gain')} and out-of-fold bootstrap "
               f"win rate ≥ {g.get('min_bootstrap_win_rate')}.")
    vers = LRN.get("versions", [])
    if vers:
        st.markdown("**Weight versions**")
        st.dataframe(pd.DataFrame([{"scope": v["scope"], "version": v["version"], "status": v["status"], "created": v.get("created"),
                                    "records": (v.get("trained_from") or {}).get("records", "—"),
                                    "CV current → candidate": (f"{v['validation']['cv']['current']} → {v['validation']['cv']['candidate']}"
                                                               if v.get("validation") else "—"),
                                    "bootstrap win": (v.get("validation") or {}).get("bootstrap_win_rate", "—"),
                                    "why not promoted": "; ".join((v.get("validation") or {}).get("reasons_not_promoted", []))}
                                   for v in vers]), hide_index=True, use_container_width=True)
        latest = [v for v in vers if v["scope"] != "global"]
        if latest:
            v = latest[-1]
            w0 = LRN.get("active_weights", {})
            dfw = pd.DataFrame([{"feature": k, "active": w0.get(k), f"{v['version']} ({v['status']})": v["weights"].get(k),
                                 "change": round(v["weights"].get(k, 0) - w0.get(k, 0), 4)} for k in w0])
            with st.expander(f"Weights: active vs latest candidate {v['version']}"):
                st.dataframe(dfw, hide_index=True, use_container_width=True)
    for name, txt in (LRN.get("reports") or {}).items():
        with st.expander(f"Report: {name}"):
            st.markdown(txt)
    if not PUBLIC:
        st.divider()
        if st.button("Run the learner now (dry run: validates, writes nothing)"):
            from promo import feedback as Fb, learner as Ln, weights as Wt
            res = Ln.gate(Fb.load_training(SID), Wt.resolve(SID)[0])
            (st.success if res["promote"] else st.warning)(
                ("Would be PROMOTED." if res["promote"] else "Would NOT be promoted: " + "; ".join(res["reasons"])))
            st.json({"data": res["data"], "cv": {k: v for k, v in res["cv"].items() if k != "folds"},
                     "bootstrap_win_rate": res["bootstrap_win_rate"], "weight_change": res["weight_change"]})
    else:
        st.caption("In the public demo, reviews are stored separately and never reach the learner.")
