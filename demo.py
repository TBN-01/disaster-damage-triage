"""Local disaster triage case study, blind review workflow, and error analysis."""

from __future__ import annotations

import html
import hashlib
import json
import math
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from disaster_triage.ui_data import case_filter, ranked_queue
from disaster_triage.review_store import load_reviews, save_review
from disaster_triage.sample_demo import make_sample_demo


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.environ.get("TRIAGE_RESULTS_DIR", ROOT / "results"))
PUBLISHED_METRICS = ROOT / "assets" / "held-out-metrics.json"
PREDICTIONS_FILE = RESULTS_DIR / "predictions.csv"
METRICS_FILE = RESULTS_DIR / "metrics.json"
REVIEW_DB = Path(os.environ.get("TRIAGE_REVIEW_DB", ROOT / "results" / "reviews.sqlite3"))
CROSS_EVENT_FILE = ROOT / "assets" / "cross-event-metrics.csv"

st.set_page_config(
    page_title="Disaster Damage Triage",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="auto",
)
st.markdown(f"<style>{(ROOT / 'assets' / 'ui.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def read_metrics() -> tuple[dict, bool]:
    """Prefer a local experiment, but always show the committed case study."""
    source = METRICS_FILE if METRICS_FILE.exists() else PUBLISHED_METRICS
    return json.loads(source.read_text(encoding="utf-8")), source == METRICS_FILE


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def result_row(label: str, value: float, color: str) -> str:
    width = max(0, min(100, value * 100))
    return (
        f'<div class="result-row"><span class="name">{html.escape(label)}</span>'
        f'<div class="bar-track"><div class="bar-fill {color}" style="width:{width:.1f}%"></div></div>'
        f'<span class="number">{value:.1%}</span></div>'
    )


try:
    metrics, using_local_metrics = read_metrics()
except (OSError, ValueError) as error:
    st.error(f"Could not load evaluation results: {error}")
    st.stop()

model_metrics = metrics.get("model") or metrics.get("test_metrics")
baseline_metrics = metrics.get("baseline")
if not model_metrics:
    st.error("The metrics file does not include a model evaluation.")
    st.stop()

predictions = None
if PREDICTIONS_FILE.exists():
    try:
        predictions = pd.read_csv(PREDICTIONS_FILE)
        if predictions.empty:
            predictions = None
    except (OSError, pd.errors.ParserError) as error:
        st.warning(f"The local prediction table could not be loaded: {error}")

local_explorer = using_local_metrics and predictions is not None
sample_predictions = make_sample_demo(ROOT / "results" / "synthetic-demo")
review_predictions = predictions if local_explorer else sample_predictions
review_is_synthetic = not local_explorer
dataset_key = "synthetic-demo-v1" if review_is_synthetic else hashlib.sha256(PREDICTIONS_FILE.read_bytes()).hexdigest()[:16]
event_name = str(metrics.get("test_event", "held-out disaster")).replace("-", " ").title()
review_count = math.ceil(model_metrics["samples"] * 0.2)

with st.sidebar:
    st.markdown("### 🛰️ Disaster Damage Triage")
    st.caption("A data-science case study in prioritizing human review.")
    if local_explorer:
        st.markdown('<span class="status-pill">Local image explorer ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill warn">Published results + synthetic practice</span>', unsafe_allow_html=True)
    st.divider()
    st.markdown("**What you're seeing**")
    st.write("A retrospective xBD experiment. Damage labels are used to check the ranking, never to generate its scores.")
    st.link_button("View code and methods ↗", "https://github.com/TBN-01/disaster-damage-triage", width="stretch")
    st.caption("Real xBD imagery stays on this computer. The built-in interactive example uses synthetic images.")

st.markdown(
    '<div class="hero"><span class="eyebrow">Satellite imagery · data science · human-in-the-loop</span>'
    '<h1>Which buildings should be reviewed first?</h1>'
    '<p>After a disaster, there may be too many buildings to inspect at once. This project tests whether before-and-after imagery can help put the most urgent cases near the front of a human review queue.</p>'
    '</div>',
    unsafe_allow_html=True,
)

results_tab, scorecard_tab, review_tab, explore_tab, errors_tab, method_tab = st.tabs([
    "The result", "Across disasters", "Review desk", "Explore the queue", "Error gallery", "How it works",
])

with results_tab:
    st.subheader(f"A tougher test: {event_name}")
    st.markdown(
        '<p class="section-intro">Every building from this disaster was kept out of the model’s training data. '
        'The goal is to find severe cases within a limited review budget—not to automatically certify damage.</p>',
        unsafe_allow_html=True,
    )
    severe_count = model_metrics["severe_buildings"]
    st.markdown(
        '<div class="metric-strip">'
        f'<div class="metric-tile"><span class="value">{model_metrics["samples"]:,}</span><span class="label">buildings in the held-out event</span></div>'
        f'<div class="metric-tile"><span class="value">{severe_count:,}</span><span class="label">labeled major damage or destroyed</span></div>'
        f'<div class="metric-tile"><span class="value">{review_count:,}</span><span class="label">reviewed at a 20% budget</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Severe cases found in the first 20% reviewed")
    rows = ""
    if baseline_metrics:
        rows += result_row("Simple image change", baseline_metrics["recall_at_20_percent_reviewed"], "teal")
    rows += result_row("Trained random forest" if baseline_metrics else "Trained model", model_metrics["recall_at_20_percent_reviewed"], "amber")
    rows += result_row("Random order (expected)", 0.2, "gray")
    st.markdown(f'<div class="result-panel">{rows}</div>', unsafe_allow_html=True)

    if baseline_metrics:
        found_baseline = round(baseline_metrics["recall_at_20_percent_reviewed"] * severe_count)
        found_model = round(model_metrics["recall_at_20_percent_reviewed"] * severe_count)
        st.success(
            f"The simple change score found {found_baseline} of {severe_count} severe cases in the first "
            f"{review_count} reviews. The trained model found {found_model}. For this disaster, the simpler method is the better queue."
        )
    st.caption("These numbers describe one held-out event in a sampled research dataset. They are not field accuracy or a safety guarantee.")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="note-card"><h3>Why the trained model matters</h3><p>It is tempting to assume a more complex model will win. Here it did not. Testing on an entirely different disaster exposed a weakness that a random building split could have hidden.</p></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="note-card"><h3>What the score actually means</h3><p>A higher image-change score moves a building earlier in the queue. It is not a probability that the building is damaged, and it is never a substitute for a qualified human assessment.</p></div>', unsafe_allow_html=True)

with scorecard_tab:
    st.subheader("Does the ranking transfer to other disasters?")
    st.write("Each row holds out one entire disaster, trains the random forest on the others, and checks how many severe cases appear in the first 20% reviewed.")
    if CROSS_EVENT_FILE.exists():
        cross = pd.read_csv(CROSS_EVENT_FILE)
        cross["Event"] = cross["event"].str.replace("-", " ").str.title()
        chart = cross.pivot(index="Event", columns="method", values="recall_at_20_percent_reviewed")
        st.dataframe(chart.style.format("{:.1%}"), width="stretch")
        details = cross[cross["method"] == "Simple image change"].copy()
        details["Severe share"] = details["severe_prevalence"].map(lambda value: f"{value:.1%}")
        details["95% tile-resample range"] = details.apply(
            lambda row: f"{row['recall_ci_low']:.1%}–{row['recall_ci_high']:.1%}", axis=1,
        )
        st.dataframe(details[["Event", "buildings", "tiles", "severe", "Severe share", "95% tile-resample range"]].rename(columns={
            "buildings": "Buildings", "tiles": "Tiles", "severe": "Severe labels",
        }), width="stretch", hide_index=True)
        st.caption("The ranges resample image tiles within each event. They describe instability in this sampled dataset, not uncertainty about future field performance. Events with few severe labels are especially fragile. Image change is a ranking score, not a probability.")
        st.download_button("Download event scorecard", cross.to_csv(index=False), file_name="cross-event-metrics.csv", mime="text/csv")
    else:
        st.info("The all-event scorecard has not been generated yet. Run the cross-validation command in the README to add it.")

with review_tab:
    st.subheader("Try the review desk")
    if review_is_synthetic:
        st.warning("Interactive practice mode: these 24 image pairs, labels, and scores are synthetic. They are not part of the reported xBD evaluation.")
    else:
        st.info("Using your local xBD results. Historical labels stay hidden until a review decision is submitted.")
    review_method = st.selectbox("Queue order", ["Simple image change", "Trained model"], key="review_method")
    review_score = "baseline_score" if review_method == "Simple image change" else "model_score"
    if review_score not in review_predictions:
        st.error("This prediction table lacks the selected score.")
    else:
        review_ranked, _ = ranked_queue(review_predictions, review_score, 0.2)
        prior = load_reviews(REVIEW_DB, dataset_key)
        reviewed_ids = set(prior["sample_id"].astype(str))
        choice = st.radio("Show", ["Not yet reviewed", "Reviewed", "All"], horizontal=True, key="review_filter")
        visible = review_ranked.copy()
        if choice == "Not yet reviewed":
            visible = visible[~visible["sample_id"].astype(str).isin(reviewed_ids)]
        elif choice == "Reviewed":
            visible = visible[visible["sample_id"].astype(str).isin(reviewed_ids)]
        st.progress(min(len(reviewed_ids) / len(review_ranked), 1.0), text=f"{len(reviewed_ids):,} of {len(review_ranked):,} reviewed on this computer")
        if visible.empty:
            st.success("No buildings remain in this view. Choose another filter to revisit decisions.")
        else:
            st.dataframe(visible[["rank", "sample_id", review_score]].rename(columns={
                "rank": "Queue position", "sample_id": "Building ID", review_score: "Ranking score",
            }).head(40), width="stretch", hide_index=True, height=230)
            selected_id = st.selectbox("Building to inspect", visible["sample_id"].astype(str).tolist(), key="review_id")
            case = visible[visible["sample_id"].astype(str) == selected_id].iloc[0]
            st.caption(f"Queue position {int(case['rank']):,}. The score orders reviews; it does not diagnose damage.")
            left, right = st.columns(2)
            if Path(case["pre_crop"]).exists() and Path(case["post_crop"]).exists():
                left.image(str(case["pre_crop"]), caption="Before", width="stretch")
                right.image(str(case["post_crop"]), caption="After", width="stretch")
            else:
                st.warning("This building's image crops are unavailable on this computer.")
            previous = prior[prior["sample_id"].astype(str) == selected_id]
            if previous.empty:
                with st.form("blind_review_form", clear_on_submit=False):
                    decision = st.radio("Your assessment", ["Severe", "Not severe", "Unsure"], index=None)
                    submitted = st.form_submit_button("Save assessment and reveal historical label", type="primary")
                if submitted:
                    if decision is None:
                        st.warning("Choose an assessment first.")
                    else:
                        save_review(REVIEW_DB, dataset_key, selected_id, decision)
                        st.rerun()
            else:
                decision = str(previous.iloc[0]["decision"])
                ground_truth = "Severe" if bool(case["severe"]) else "Not severe"
                st.success(f"Your assessment: {decision} · {'Simulated' if review_is_synthetic else 'Historical xBD'} label: {ground_truth}")
                st.caption("The comparison is educational; it is not a safety verdict.")
        if not prior.empty:
            joined = prior.merge(review_ranked[["sample_id", "severe"]], on="sample_id", how="inner")
            certain = joined[joined["decision"] != "Unsure"]
            agreements = int(((certain["decision"] == "Severe") == certain["severe"].astype(bool)).sum())
            st.caption(f"Saved locally: {len(joined)} reviews · {agreements}/{len(certain)} assessments matched the {'simulated' if review_is_synthetic else 'historical'} label (excluding Unsure).")
            st.download_button("Export my review decisions", joined.to_csv(index=False), file_name="local-review-decisions.csv", mime="text/csv")

with explore_tab:
    st.subheader("Explore a human-review queue")
    if not local_explorer:
        st.info("The public case study is available without downloading imagery. To inspect individual buildings, run the xBD pipeline locally and point this app at its results folder.")
        st.code("triage-prepare --data-root C:\\path\\to\\xBD --out data\\processed --max-tiles-per-event 30 --seed 42\ntriage-train --manifest data\\processed\\manifest.csv --test-event santa-rosa-wildfire --out results\nstreamlit run demo.py", language="powershell")
        st.link_button("Read the setup guide ↗", "https://github.com/TBN-01/disaster-damage-triage#obtain-data-and-run")
    else:
        score_options = {"Simple image change": "baseline_score", "Trained model": "model_score"}
        score_options = {label: column for label, column in score_options.items() if column in predictions.columns}
        if not score_options:
            st.error("The prediction table has no ranking score column.")
            st.stop()

        method_column, budget_column = st.columns(2, gap="large")
        with method_column:
            method = st.selectbox("Ranking method", list(score_options), help="The change score is the recommended method for the reported wildfire test.")
        with budget_column:
            budget = st.slider("Human review budget", min_value=5, max_value=50, value=20, step=5, format="%d%%")

        ranked, queue = ranked_queue(predictions, score_options[method], budget / 100)
        m1, m2, m3 = st.columns(3)
        m1.metric("Buildings to review", f"{queue['review_count']:,}")
        m2.metric("Severe cases found", f"{queue['severe_found']:,} / {queue['severe_total']:,}")
        m3.metric("Severe-case recall", pct(queue["recall"]))
        st.caption("Change the budget to see the tradeoff. Historical xBD labels let us evaluate the queue; they would not be known during a real response.")

        filter_choice = st.selectbox(
            "Show cases",
            ["Priority queue", "All buildings", "Severe cases found", "Severe cases missed", "False alarms"],
            help="Found, missed, and false-alarm categories use historical labels for retrospective error analysis.",
        )
        filtered = case_filter(ranked, filter_choice)
        search = st.text_input("Find a building ID", placeholder="Type any part of an ID, such as a tile number")
        if search:
            filtered = filtered[filtered["sample_id"].astype(str).str.contains(search, case=False, regex=False)]
        st.caption(f"Showing {len(filtered):,} of {len(ranked):,} buildings")

        table_columns = ["rank", "sample_id", score_options[method], "review_priority", "subtype"]
        table_columns = [column for column in table_columns if column in filtered]
        display = filtered[table_columns].rename(columns={
            "rank": "Rank", "sample_id": "Building ID", score_options[method]: "Ranking score",
            "review_priority": "Review first", "subtype": "Historical label",
        })
        st.dataframe(display, width="stretch", hide_index=True, height=330)
        export = ranked[["rank", "sample_id", score_options[method], "review_priority", "subtype"]].to_csv(index=False)
        st.download_button("Download this ranked queue", export, file_name="building-review-queue.csv", mime="text/csv")

        if filtered.empty:
            st.info("No buildings match these filters. Clear the search or choose another category.")
        else:
            st.markdown("#### Inspect one building")
            selected_id = st.selectbox("Building ID", filtered["sample_id"].astype(str).tolist())
            selected = filtered[filtered["sample_id"].astype(str) == selected_id].iloc[0]
            selected_status = "In the review queue" if selected["review_priority"] else "Outside the current review budget"
            st.write(f"**Rank {int(selected['rank']):,}** · {selected_status} · historical label: **{selected['subtype']}**")
            pre_crop, post_crop = Path(selected["pre_crop"]), Path(selected["post_crop"])
            if pre_crop.exists() and post_crop.exists():
                before_col, after_col = st.columns(2)
                before_col.image(str(pre_crop), caption="Before", width="stretch")
                after_col.image(str(post_crop), caption="After", width="stretch")
            else:
                st.warning("The image crops for this building are missing from the local dataset.")

            scene_columns = {"pre_tile", "post_tile", "tile", "x0", "y0", "x1", "y1"}
            if scene_columns.issubset(ranked.columns):
                with st.expander("See this building in the full satellite scene"):
                    scene = ranked[ranked["tile"] == selected["tile"]]
                    pre_tile, post_tile = Path(selected["pre_tile"]), Path(selected["post_tile"])
                    if pre_tile.exists() and post_tile.exists():
                        with Image.open(post_tile) as source:
                            marked = source.convert("RGB")
                        draw = ImageDraw.Draw(marked)
                        for building in scene.itertuples(index=False):
                            color = "#ec5e4f" if building.review_priority else "#71889a"
                            draw.rectangle((building.x0, building.y0, building.x1, building.y1), outline=color, width=3)
                        scene_before, scene_after = st.columns(2)
                        scene_before.image(str(pre_tile), caption="Before · known building locations", width="stretch")
                        scene_after.image(marked, caption="After · coral: review first; slate: later", width="stretch")
                        st.caption("Box colors show ranking priority, not confirmed damage or building safety.")
                    else:
                        st.info("The original satellite tiles are not available in this local results folder.")

with errors_tab:
    st.subheader("Where the queue gets it wrong")
    if review_is_synthetic:
        st.warning("These example images and errors are synthetic. Load local xBD predictions to inspect real held-out cases.")
    else:
        st.write("These are real held-out cases from your local xBD run. The label comparison is retrospective and does not establish why an individual ranking failed.")
    error_method = st.selectbox("Ranking to inspect", ["Simple image change", "Trained model"], key="error_method")
    error_score = "baseline_score" if error_method == "Simple image change" else "model_score"
    if error_score in review_predictions:
        error_ranked, _ = ranked_queue(review_predictions, error_score, 0.2)
        error_type = st.radio("Error type", ["Severe cases missed", "False alarms"], horizontal=True)
        examples = case_filter(error_ranked, error_type)
        st.metric("Cases in this category", f"{len(examples):,}")
        if error_type == "Severe cases missed":
            st.write("These severe-label buildings fell outside the first 20% reviewed. Check whether damage is hard to see at this scale, localized, or obscured. Those are inspection prompts—not confirmed causes.")
        else:
            st.write("These buildings entered the first 20% but lack a severe label. Check for shadows, lighting or seasonal change, image alignment, and nearby changes. Those are inspection prompts—not confirmed causes.")
        if examples.empty:
            st.info("No cases in this category for the selected ranking.")
        else:
            count = min(6, len(examples))
            for case in examples.head(count).itertuples(index=False):
                with st.container(border=True):
                    st.write(f"**{case.sample_id}** · queue position {case.rank:,} · {'simulated' if review_is_synthetic else 'historical'} label: {case.subtype}")
                    before, after = st.columns(2)
                    if Path(case.pre_crop).exists() and Path(case.post_crop).exists():
                        before.image(case.pre_crop, caption="Before", width="stretch")
                        after.image(case.post_crop, caption="After", width="stretch")
                        if review_is_synthetic:
                            number = int(str(case.sample_id).split("-")[-1]) - 1
                            if error_type == "False alarms" and number in {1, 6, 15, 19}:
                                st.caption("Synthetic cause: a large dark region was added to imitate a lighting or shadow change without simulated structural damage.")
                            elif error_type == "Severe cases missed" and number in {9, 20}:
                                st.caption("Synthetic cause: the damaged area was deliberately kept small, giving this case a weaker overall change score.")
                        else:
                            st.caption("Possible checks: alignment, lighting, occlusion, building size, and whether surrounding changes dominate the crop. No cause has been verified for this case.")
                    else:
                        st.warning("This case's local image crops are missing.")
            st.caption(f"Showing the first {count} cases in ranked order. A visual pattern should be recorded as a hypothesis, not a verified explanation.")

with method_tab:
    st.subheader("What the project does—and does not do")
    st.markdown('<p class="section-intro">The research question is whether a limited human review team can see more severe cases sooner. This is a ranking problem, not a replacement for inspectors.</p>', unsafe_allow_html=True)
    steps = [
        ("1. Match images and labels", "Read xBD before/after satellite tiles and the human-labeled building polygons."),
        ("2. Make building pairs", "Crop each known building location in both images, with a little surrounding context."),
        ("3. Rank and compare", "Compare a transparent image-change score with a random forest trained on image summaries."),
        ("4. Test across events", "Keep each disaster out of training in turn, then measure severe cases found within the first 20% reviewed."),
    ]
    for title, description in steps:
        st.markdown(f'<div class="note-card" style="margin-bottom:.65rem"><h3>{title}</h3><p>{description}</p></div>', unsafe_allow_html=True)

    with st.expander("Aren't the damage labels already public?"):
        st.write("Yes—for these historical disasters. That is what makes it possible to test the ranking. After a new disaster, those reviewed building labels would not be available yet. This project studies how to prioritize that work, not how to re-publish known answers.")
    with st.expander("Can this tell me whether a building is safe?"):
        st.write("No. It assumes building locations are already known, uses satellite imagery rather than an on-site inspection, and has only been evaluated on sampled research data. Shadows, smoke, image alignment, and changing conditions can mislead it.")
    with st.expander("Why did the random forest lose?"):
        st.write("Its features may have picked up patterns specific to the training disasters. That is a hypothesis, not a proven cause. The observed result is that it ranked severe buildings much worse than simple image change on the held-out wildfire.")

    st.caption("Dataset: xBD / xView2. The portable practice images are synthetic. This app does not redistribute xBD imagery or annotations. See the repository for attribution, license terms, tests, and reproducibility instructions.")
