"""Local, read-only demonstration of held-out disaster predictions."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw


st.set_page_config(page_title="Disaster Damage Triage", layout="wide")
st.title("Disaster Damage Triage")
st.caption("Building-level triage from before/after satellite imagery · research demonstration")

results_dir = Path(os.environ.get("TRIAGE_RESULTS_DIR", "results"))
predictions_file = results_dir / "predictions.csv"
metrics_file = results_dir / "metrics.json"
if not predictions_file.exists() or not metrics_file.exists():
    st.info("No evaluated model yet. Prepare the xBD data and run triage-train first; see README.md.")
    st.stop()

metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
predictions = pd.read_csv(predictions_file)
if predictions.empty:
    st.error("The predictions file is empty.")
    st.stop()

st.subheader(f"Held-out disaster: {metrics['test_event']}")
st.write("The trained model saw no buildings from this disaster. Both rankings use known building locations supplied by xBD annotations.")
model_metrics = metrics.get("model", metrics.get("test_metrics"))
baseline_metrics = metrics.get("baseline")
ranking_options = (["Simple image change"] if "baseline_score" in predictions else []) + ["Trained model"]
ranking = st.radio("Rank by", ranking_options, horizontal=True)
score_column = "baseline_score" if ranking == "Simple image change" else "model_score"
selected_metrics = baseline_metrics if ranking == "Simple image change" else model_metrics
c1, c2, c3 = st.columns(3)
c1.metric("Buildings tested", selected_metrics["samples"])
c2.metric("Severe buildings found in top 20%", f"{selected_metrics['recall_at_20_percent_reviewed']:.1%}")
c3.metric("Average precision", f"{selected_metrics['average_precision']:.3f}")
if baseline_metrics:
    st.write(
        f"Comparison on this event: image change **{baseline_metrics['recall_at_20_percent_reviewed']:.1%}** "
        f"vs. trained model **{model_metrics['recall_at_20_percent_reviewed']:.1%}** "
        "severe-damage recall in the top 20%. The simple ranking is recommended for this demonstration."
    )
predictions = predictions.sort_values(score_column, ascending=False).reset_index(drop=True)
review_count = max(1, (len(predictions) + 4) // 5)
predictions["review_priority"] = False
predictions.loc[: review_count - 1, "review_priority"] = True
st.caption(f"The red queue contains the top {review_count:,} of {len(predictions):,} buildings by the selected score. Scores are rankings, not calibrated probabilities.")

if {"pre_tile", "post_tile", "x0", "y0", "x1", "y1"}.issubset(predictions.columns):
    st.subheader("Satellite scene")
    tile_name = st.selectbox("Choose a scene", sorted(predictions["tile"].unique()))
    scene = predictions[predictions["tile"] == tile_name]
    pre_tile = Path(scene.iloc[0]["pre_tile"])
    post_tile = Path(scene.iloc[0]["post_tile"])
    if pre_tile.exists() and post_tile.exists():
        after = Image.open(post_tile).convert("RGB")
        draw = ImageDraw.Draw(after)
        for building in scene.itertuples(index=False):
            color = "#f04438" if building.review_priority else "#4bbf83"
            draw.rectangle((building.x0, building.y0, building.x1, building.y1), outline=color, width=3)
        left, right = st.columns(2)
        left.image(str(pre_tile), caption="Before disaster", width="stretch")
        right.image(after, caption="After disaster · red: top-20% review queue; green: lower priority", width="stretch")

st.subheader("Ranked human-review queue")
view = predictions[["sample_id", score_column, "review_priority", "subtype", "severe"]].rename(columns={
    "sample_id": "Building", score_column: "Ranking score", "review_priority": "Review first?",
    "subtype": "Human label", "severe": "Severe?",
})
st.dataframe(view, width="stretch", hide_index=True)

rank = st.slider("Inspect ranked building", 1, len(predictions), 1)
row = predictions.iloc[rank - 1]
st.write(f"**{row['sample_id']}** · ranking score **{row[score_column]:.2f}** · human label **{row['subtype']}**")
left, right = st.columns(2)
left.image(row["pre_crop"], caption="Before disaster", width="stretch")
right.image(row["post_crop"], caption="After disaster", width="stretch")
st.caption("These are research predictions, not an operational damage assessment. Satellite angle, shadows, debris, and label uncertainty can cause errors.")

