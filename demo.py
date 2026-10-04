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
from disaster_triage.review_store import clear_reviews, load_reviews, save_review
from disaster_triage.sample_demo import make_sample_demo
from disaster_triage.prepare import square_crop


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.environ.get("TRIAGE_RESULTS_DIR", ROOT / "results"))
PUBLISHED_METRICS = ROOT / "assets" / "held-out-metrics.json"
PREDICTIONS_FILE = RESULTS_DIR / "predictions.csv"
METRICS_FILE = RESULTS_DIR / "metrics.json"
REVIEW_DB = Path(os.environ.get("TRIAGE_REVIEW_DB", ROOT / "results" / "reviews.sqlite3"))
CROSS_EVENT_FILE = ROOT / "assets" / "cross-event-metrics.csv"
CURVES_FILE = ROOT / "assets" / "review-budget-curves.csv"

st.set_page_config(
    page_title="Disaster Damage Triage",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
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


@st.cache_data(show_spinner=False, max_entries=96)
def larger_pair(pre_tile: str, post_tile: str, box: tuple[float, float, float, float]) -> tuple[Image.Image, Image.Image]:
    """Make a clearer display crop from the original tiles; scores stay unchanged."""
    with Image.open(pre_tile) as source:
        before = square_crop(source, box, 256)
    with Image.open(post_tile) as source:
        after = square_crop(source, box, 256)
    return before, after


def preview_pair(row: dict | pd.Series) -> tuple[Image.Image | str | None, Image.Image | str | None]:
    """Prefer original tiles for display, with prepared crops as the fallback."""
    pre_crop, post_crop = Path(str(row["pre_crop"])), Path(str(row["post_crop"]))
    scene_fields = {"pre_tile", "post_tile", "x0", "y0", "x1", "y1"}
    if scene_fields.issubset(row):
        pre_tile, post_tile = Path(str(row["pre_tile"])), Path(str(row["post_tile"]))
        if pre_tile.exists() and post_tile.exists():
            try:
                box = tuple(float(row[key]) for key in ("x0", "y0", "x1", "y1"))
                return larger_pair(str(pre_tile), str(post_tile), box)
            except (OSError, ValueError, TypeError):
                pass
    if pre_crop.exists() and post_crop.exists():
        return str(pre_crop), str(post_crop)
    return None, None


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
event_name = str(metrics.get("test_event", "held-out disaster")).replace("-", " ").title()
review_count = math.ceil(model_metrics["samples"] * 0.2)

with st.sidebar:
    st.markdown("### 🛰️ Disaster Damage Triage")
    st.caption("A small project about which buildings to check first after a disaster.")
    if local_explorer:
        st.markdown('<span class="status-pill">Local image explorer ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill warn">Results + practice images</span>', unsafe_allow_html=True)
    st.divider()
    st.markdown("**What you're seeing**")
    st.write("I used past disasters to test the queue. Their damage labels let me check the results; the ranking does not get to see them.")
    st.link_button("See the project on GitHub ↗", "https://github.com/TBN-01/disaster-damage-triage", width="stretch")
    if local_explorer:
        st.caption("Real xBD image crops are loaded from this computer only. They are not uploaded or included in the GitHub repo.")
    else:
        st.caption("The practice images are made up. Real xBD images are not included in this download.")

st.markdown(
    '<div class="hero"><span class="eyebrow">Before-and-after satellite images</span>'
    '<h1>Which buildings should be reviewed first?</h1>'
    '<p>If there are too many buildings to check at once, can image changes help put the most important ones earlier in the line? That is what I tested here.</p>'
    '</div>',
    unsafe_allow_html=True,
)

story_tab, practice_tab, data_tab = st.tabs(["The story", "Try a review", "Explore the results"])
results_tab = method_tab = story_tab
review_tab = errors_tab = practice_tab
scorecard_tab = explore_tab = data_tab

with st.expander("New here? Take the 2-minute tour"):
    st.markdown(
        "1. **The story:** see the question, the Santa Rosa result, and where the approach fell short.\n"
        "2. **Try a review:** compare two images, make a call before seeing the label, then move to the next building. "
        "Choose practice images for a quick demo, or real local images if you have the xBD files on this computer.\n"
        "3. **Explore the results:** pick a disaster and change how many buildings a person could review. "
        "The chart shows how many severe cases each queue would have reached in that historical test."
    )

with results_tab:
    st.subheader(f"One close look: {event_name}")
    st.markdown(
        '<p class="section-intro">I kept this entire disaster out of the training data, then asked how many severe cases would turn up if someone checked only the first 20% of the queue.</p>',
        unsafe_allow_html=True,
    )
    severe_count = model_metrics["severe_buildings"]
    st.markdown(
        '<div class="metric-strip">'
        f'<div class="metric-tile"><span class="value">{model_metrics["samples"]:,}</span><span class="label">buildings in this disaster sample</span></div>'
        f'<div class="metric-tile"><span class="value">{severe_count:,}</span><span class="label">labeled major damage or destroyed</span></div>'
        f'<div class="metric-tile"><span class="value">{review_count:,}</span><span class="label">buildings in the first 20% of the queue</span></div>'
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
        comparison = (
            "The simple score made the better queue for this disaster."
            if found_baseline > found_model else
            "The trained model made the better queue for this disaster."
            if found_model > found_baseline else
            "They found the same number of severe cases here."
        )
        st.info(
            f"The simple change score found {found_baseline} of {severe_count} severe cases in the first "
            f"{review_count} reviews. The trained model found {found_model}. {comparison}"
        )
    st.caption("These are results from one historical sample, not a safety guarantee or a prediction of how a new disaster would go.")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="note-card"><h3>Why I tested both</h3><p>I wanted to know whether training a model actually helped. The answer changed depending on the disaster, which is why I kept both results.</p></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="note-card"><h3>What the score means</h3><p>A higher score moves a building earlier in the line. It does not tell you the chance of damage, and it cannot replace an inspection.</p></div>', unsafe_allow_html=True)

    st.markdown("#### What the other tests showed")
    st.write(
        "The Santa Rosa result looked promising on its own. Once I repeated the test on other disasters, "
        "the results were much less steady. That is why I show the misses too."
    )

with scorecard_tab:
    st.subheader("How many buildings can you review?")
    st.write("Pick a past disaster and change the review budget. The question is: how many severe cases land near the front of the line when time is limited?")
    if CROSS_EVENT_FILE.exists() and CURVES_FILE.exists():
        cross = pd.read_csv(CROSS_EVENT_FILE)
        curves = pd.read_csv(CURVES_FILE)
        event_options = sorted(curves["event"].unique())
        event_index = event_options.index("santa-rosa-wildfire") if "santa-rosa-wildfire" in event_options else 0
        select_col, budget_col = st.columns(2, gap="large")
        with select_col:
            selected_event = st.selectbox(
                "Pick a disaster", event_options, index=event_index,
                format_func=lambda event: event.replace("-", " ").title(),
            )
        with budget_col:
            selected_budget = st.slider("How much of the queue can be reviewed?", 5, 50, 20, 5, format="%d%%")

        event_curves = curves[curves["event"] == selected_event]
        selected = event_curves[event_curves["budget_percent"] == selected_budget].set_index("method")
        simple = selected.loc["Simple image change"]
        forest = selected.loc["Random forest"]
        first, second = st.columns(2, gap="large")
        first.metric("Severe found · simple image change", f"{int(simple['severe_found']):,} / {int(simple['severe']):,}")
        second.metric("Severe found · random forest", f"{int(forest['severe_found']):,} / {int(forest['severe']):,}")
        st.caption(
            f"At a {selected_budget}% budget, that means checking {int(simple['reviewed']):,} of "
            f"{int(simple['buildings']):,} buildings from this sampled event. The labels are known now because this is a historical test."
        )
        difference = int(simple["severe_found"] - forest["severe_found"])
        if difference > 0:
            st.info(f"At this budget, the simple queue reached {difference} more severe-label buildings than the trained queue.")
        elif difference < 0:
            st.info(f"At this budget, the trained queue reached {-difference} more severe-label buildings than the simple queue.")
        else:
            st.info("At this budget, both queues reached the same number of severe-label buildings.")
        plot = event_curves.pivot(index="budget_percent", columns="method", values="recall").rename(columns={
            "Simple image change": "Simple image change", "Random forest": "Trained random forest",
        })
        plot["Random order (expected)"] = plot.index.to_numpy(dtype=float) / 100
        st.line_chart(plot, height=290)
        st.caption("Horizontal axis: percent of buildings reviewed. Vertical axis: share of severe cases found. The random-order line is an average reference, not a promise for any one run.")

        if int(simple["severe"]) < 25:
            st.warning("This event has very few severe labels in the sample. A small number of buildings can swing the percentages a lot.")
        if int(simple["severe_found"]) == 0 and int(forest["severe_found"]) == 0:
            st.error("Neither method reached a severe case at this budget. This is exactly the kind of failure the project needs to show.")

        with st.expander("See the full ten-disaster comparison at 20%"):
            cross["Event"] = cross["event"].str.replace("-", " ").str.title()
            chart = cross.pivot(index="Event", columns="method", values="recall_at_20_percent_reviewed")
            st.dataframe(chart.style.format("{:.1%}"), width="stretch")
            st.write("The simple method did better on eight events, the trained model on one, and they tied on one. Individual events can still go badly.")
            st.caption("The downloaded scorecard also includes event size, severe-label prevalence, average precision, and tile-resampling ranges. Those ranges describe this sample, not future field performance.")
            st.download_button("Download the event scorecard", cross.to_csv(index=False), file_name="cross-event-metrics.csv", mime="text/csv")
        st.download_button("Download all review-budget results", curves.to_csv(index=False), file_name="review-budget-curves.csv", mime="text/csv")
    else:
        st.info("The event results are missing. The command to rebuild them is in METHODS.md.")

with review_tab:
    st.subheader("What would you decide?")
    st.write("Compare the two images, make your call, then see the saved label. Your answer stays on this computer.")
    if local_explorer:
        image_source = st.radio(
            "Which images would you like to review?",
            ["Real local images", "Practice images"],
            horizontal=True,
            help="Practice images are made up. Real local images come from your xBD files and stay on this computer.",
        )
    else:
        image_source = "Practice images"
    review_is_synthetic = image_source == "Practice images"
    review_predictions = sample_predictions if review_is_synthetic else predictions
    dataset_key = (
        "synthetic-demo-v1" if review_is_synthetic
        else hashlib.sha256(PREDICTIONS_FILE.read_bytes()).hexdigest()[:16]
    )
    if st.session_state.get("active_dataset_key") != dataset_key:
        st.session_state.pop("just_reviewed_id", None)
        st.session_state["active_dataset_key"] = dataset_key
    if review_is_synthetic:
        st.caption("Practice images are made up. Your choices do not affect the real xBD results.")
    else:
        st.caption("Using real xBD images from this computer. The historical label stays hidden until you answer.")
    order_col, view_col = st.columns(2, gap="large")
    with order_col:
        review_method = st.selectbox("Queue order", ["Simple image change", "Trained model"], key="review_method")
    review_score = "baseline_score" if review_method == "Simple image change" else "model_score"
    if review_score not in review_predictions:
        st.error("This prediction table lacks the selected score.")
    else:
        review_ranked, _ = ranked_queue(review_predictions, review_score, 0.2)
        prior = load_reviews(REVIEW_DB, dataset_key)
        reviewed_ids = set(prior["sample_id"].astype(str))
        with view_col:
            choice = st.selectbox("Which cases?", ["Not yet reviewed", "Reviewed", "All"], key="review_filter")
        just_reviewed = st.session_state.get("just_reviewed_id")
        visible = review_ranked.copy()
        if choice == "Not yet reviewed":
            visible = visible[
                ~visible["sample_id"].astype(str).isin(reviewed_ids)
                | (visible["sample_id"].astype(str) == just_reviewed)
            ]
        elif choice == "Reviewed":
            visible = visible[visible["sample_id"].astype(str).isin(reviewed_ids)]
        st.progress(min(len(reviewed_ids) / len(review_ranked), 1.0), text=f"{len(reviewed_ids):,} of {len(review_ranked):,} reviewed on this computer")
        if visible.empty:
            st.success("No buildings remain in this view. Choose another filter to revisit decisions.")
        else:
            review_ids = visible["sample_id"].astype(str).tolist()
            positions = dict(zip(review_ranked["sample_id"].astype(str), review_ranked["rank"]))
            selected_id = st.selectbox(
                "Choose a building (the next one is selected for you)", review_ids,
                index=review_ids.index(just_reviewed) if just_reviewed in review_ids else 0,
                format_func=lambda case_id: f"Building #{int(positions[case_id]):,} in this queue",
                key=f"review_id_{dataset_key}",
            )
            if selected_id != just_reviewed:
                st.session_state.pop("just_reviewed_id", None)
            case = visible[visible["sample_id"].astype(str) == selected_id].iloc[0]
            st.caption(f"Building #{int(case['rank']):,} in this queue. The score only decides review order.")
            with st.expander("Need the building ID for your notes?"):
                st.code(selected_id)
            left, right = st.columns(2)
            before_image, after_image = preview_pair(case)
            if before_image is not None and after_image is not None:
                left.image(before_image, caption="Before", width="stretch")
                right.image(after_image, caption="After", width="stretch")
            else:
                st.warning("This building's image crops are unavailable on this computer.")
            previous = prior[prior["sample_id"].astype(str) == selected_id]
            if previous.empty:
                with st.form("blind_review_form", clear_on_submit=False):
                    answer = st.radio(
                        "From these images, what do you think?",
                        ["Looks badly damaged", "Doesn't look badly damaged", "I can't tell"], index=None,
                    )
                    note = st.text_area("What stood out to you? (optional)", max_chars=500, placeholder="For example: the roof looks different, but the shadow makes it hard to tell.")
                    st.caption("For this project, 'severe' means the xBD label is major damage or destroyed. An image alone cannot tell you if a building is safe.")
                    submitted = st.form_submit_button("Save my answer and reveal the label", type="primary")
                if submitted:
                    if answer is None:
                        st.warning("Pick an answer first, even if it's 'I can't tell.'")
                    else:
                        decision = {
                            "Looks badly damaged": "Severe",
                            "Doesn't look badly damaged": "Not severe",
                            "I can't tell": "Unsure",
                        }[answer]
                        save_review(REVIEW_DB, dataset_key, selected_id, decision, note)
                        st.session_state["just_reviewed_id"] = selected_id
                        st.rerun()
            else:
                decision = str(previous.iloc[0]["decision"])
                ground_truth = "Severe" if bool(case["severe"]) else "Not severe"
                st.success(f"You chose: {decision} · {'Practice' if review_is_synthetic else 'Historical xBD'} label: {ground_truth}")
                if str(previous.iloc[0]["note"]).strip():
                    st.write(f"**Your note:** {previous.iloc[0]['note']}")
                st.caption("A label is useful for checking this exercise. It is not a safety verdict.")
                if selected_id == just_reviewed and st.button("Review the next building", type="primary"):
                    st.session_state.pop("just_reviewed_id", None)
                    st.rerun()
            with st.expander("See the queue behind this choice"):
                st.dataframe(visible[["rank", "sample_id", review_score]].rename(columns={
                    "rank": "Queue position", "sample_id": "Building ID", review_score: "Ranking score",
                }).head(40), width="stretch", hide_index=True, height=230)
        if not prior.empty:
            joined = prior.merge(review_ranked[["sample_id", "severe"]], on="sample_id", how="inner")
            certain = joined[joined["decision"] != "Unsure"]
            agreements = int(((certain["decision"] == "Severe") == certain["severe"].astype(bool)).sum())
            st.caption(f"Saved locally: {len(joined)} reviews · {agreements}/{len(certain)} assessments matched the {'simulated' if review_is_synthetic else 'historical'} label (excluding Unsure).")
            st.download_button("Export my review decisions", joined.to_csv(index=False), file_name="local-review-decisions.csv", mime="text/csv")
            if review_is_synthetic:
                with st.expander("Start the practice review over"):
                    st.write("This clears only your practice answers and notes on this computer. Real-image reviews are left alone. Export first if you want a copy.")
                    confirmed = st.checkbox("I want to clear my practice answers and notes")
                    if st.button("Clear practice answers", disabled=not confirmed):
                        clear_reviews(REVIEW_DB, dataset_key)
                        st.session_state.pop("just_reviewed_id", None)
                        st.rerun()

with explore_tab:
    st.divider()
    st.subheader(f"Inspect individual {event_name} buildings")
    if not local_explorer:
        st.info("Real building images are not bundled with the app. If you want to inspect them on your own computer, the dataset and setup steps are in METHODS.md.")
        st.link_button("Read the setup guide ↗", "https://github.com/TBN-01/disaster-damage-triage/blob/main/METHODS.md#run-the-full-pipeline")
    else:
        st.caption("This section uses the one held-out event loaded on this computer. Its controls are separate from the across-disaster comparison above.")
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
        st.caption("Move the slider to see what happens when someone can review more or fewer buildings. These historical labels are for checking the result; they would not exist yet after a new disaster.")

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
            before_image, after_image = preview_pair(selected)
            if before_image is not None and after_image is not None:
                before_col, after_col = st.columns(2)
                before_col.image(before_image, caption="Before", width="stretch")
                after_col.image(after_image, caption="After", width="stretch")
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
    st.divider()
    st.subheader("Now look at what the queue missed")
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
                    before_image, after_image = preview_pair(case._asdict())
                    if before_image is not None and after_image is not None:
                        before.image(before_image, caption="Before", width="stretch")
                        after.image(after_image, caption="After", width="stretch")
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
    st.divider()
    st.subheader("How I put this together")
    st.markdown('<p class="section-intro">The goal is to put more severe cases near the front of a limited review queue—not to replace inspectors.</p>', unsafe_allow_html=True)
    steps = [
        ("1. Pair the images", "Use the xBD labels to find the same building in its before and after images."),
        ("2. Crop each building", "Keep a little of the surrounding area so the images have context."),
        ("3. Make two queues", "Rank by simple image change, then compare it with a trained random forest."),
        ("4. Test on disasters the model hasn't seen", "Leave out one whole event at a time and check the first 20% of each queue."),
    ]
    for title, description in steps:
        st.markdown(f'<div class="note-card" style="margin-bottom:.65rem"><h3>{title}</h3><p>{description}</p></div>', unsafe_allow_html=True)

    with st.expander("Aren't the damage labels already public?"):
        st.write("Yes, for these past disasters. That's how I can check whether the ranking worked. After a new disaster, those damage labels would not exist yet; someone would still have to review the buildings.")
    with st.expander("Can this tell me whether a building is safe?"):
        st.write("No. It assumes building locations are already known, uses satellite imagery rather than an on-site inspection, and has only been evaluated on sampled research data. Shadows, smoke, image alignment, and changing conditions can mislead it.")
    with st.expander("Why did the random forest lose?"):
        st.write("It may have picked up patterns that didn't carry over to the wildfire. I can't prove that from this test alone. What I can say is that it did worse than simple image change on Santa Rosa, though it won on Hurricane Matthew.")

    st.caption("Dataset: xBD / xView2. The portable practice images are synthetic. This app does not redistribute xBD imagery or annotations. See the repository for attribution, license terms, tests, and reproducibility instructions.")
