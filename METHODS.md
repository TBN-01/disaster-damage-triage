# Methods and reproducibility

This page has the details behind the short [README](README.md). The project is a retrospective ranking experiment, not a system for declaring buildings safe.

## Data and labels

The [xBD dataset](https://xview2.org/dataset) has before/after satellite images, building polygons, and human damage labels. The [xView2 project description](https://www.sei.cmu.edu/projects/xview-2-challenge/) gives background. For the reported run, the original Challenge training tiles and polygon JSON annotations came from a [public training-archive mirror](https://huggingface.co/datasets/DrNerd/xBD-Training-Data). Images and annotations remain under xBD's **CC BY-NC-SA 4.0** terms and are not committed here.

I call a building **severe** when its label is `major-damage` or `destroyed`. `no-damage` and `minor-damage` are the comparison group. Unclassified buildings are skipped. Building polygons provide locations; this project does not detect new building locations.

The sample uses up to 30 tiles per event with seed 42, producing **24,465 graded building crops across ten disasters**. The code reads each tile's paired images and post-disaster JSON, crops around every graded building with some surrounding context, and extracts color and change features. The baseline is mean absolute before/after pixel difference. The other ranking is a 200-tree random forest using summary and coarse spatial features. The optional frozen ResNet-18 experiment is separate from the polygon-based results reported in the README.

## Evaluation

Each test holds out one whole disaster. The random forest is trained on the other events, then both methods rank buildings from the held-out event. The main measure is **severe-case recall among the first 20% reviewed**. This is a queue measure, not classification accuracy. The [event scorecard](assets/cross-event-metrics.csv) also includes average precision, severe-label prevalence, and descriptive 95% ranges from 500 resamples of image *tiles* within each held-out event. Those ranges show how unstable this sampled result can be; they do not predict future field performance.

For Santa Rosa wildfire, 22,361 buildings from nine other events were available for training. The held-out event had 2,104 buildings, including 361 severe labels. At a 20% budget (421 reviews), image change found **288/361 (79.8%)** and the random forest found **116/361 (32.1%)**. [Additional single-event metrics](assets/held-out-metrics.json) include thresholded precision and recall; ranking is the main task.

Across ten held-out events, mean *event-level* recall at the same budget was **41.4%** for image change and **26.1%** for the random forest. Those are unweighted event averages, not one pooled building-level score. Image change won eight events, lost to the forest on Hurricane Matthew, and tied at zero on Mexico earthquake. Mexico earthquake had only seven severe labels among 8,074 sampled buildings, so its percentage is especially fragile. A random-order queue would find about 20% of severe cases on average, but that does not make either method reliable in a new event.

I did not tune the forest on the held-out event. A random building split would be easier but misleading: neighboring buildings can share imagery, geography, and disaster-specific conditions. Even event holdouts are not external validation. The forest's score is not calibrated, and a high image-change score can come from shadow, smoke, viewing angle, seasonal difference, or misalignment rather than damage. The error gallery offers things to inspect, not verified causes for individual cases.

## Run the full pipeline

Register on the [official xBD download page](https://xview2.org/dataset), download the Challenge training images and labels, and extract them locally. The code looks for sibling `images/` and `labels/` directories below your data root, with matching `*_pre_disaster.png`, `*_post_disaster.png`, and `*_post_disaster.json` files.

Windows example:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[demo,test]"
python -m disaster_triage.inspect_data --data-root C:\path\to\xBD
triage-prepare --data-root C:\path\to\xBD --out data\processed --max-tiles-per-event 30 --seed 42
triage-train --manifest data\processed\manifest.csv --test-event santa-rosa-wildfire --out results
triage-cross-validate --manifest data\processed\manifest.csv --out results\cross-event --feature-cache data\processed\features.npz
streamlit run demo.py
```

The tile cap keeps the run manageable; remove `--max-tiles-per-event` to use every available tile. Guatemala volcano has only 18 tiles in this archive. The crop manifest contains absolute local file paths and is Git-ignored. If you move the data or change crops, rerun preparation and delete the optional feature cache. Cross-validation writes `event_metrics.csv`, `fold_predictions.csv`, and `summary.json` locally. If results live in another folder, set `TRIAGE_RESULTS_DIR` before starting the app.

The portable app launcher uses a local `.venv` and needs internet only for its first package install on each computer. `run-demo.cmd` is for Windows; `sh run-demo.sh` is for macOS/Linux. The built-in synthetic image pairs and saved reviews are generated under Git-ignored `results/`. The repo is not a standalone executable and needs Python 3.10+.

## Earlier experiments

I first tried a much smaller [public Parquet mirror](https://huggingface.co/datasets/EVER-Z/torchange_xView2) that has semantic damage masks rather than original building polygons. Connected regions in those masks are only *approximate* building crops—adjacent buildings can merge. On eight tiles per hurricane, the Hurricane Michael holdout contained 707 components, 58 labeled severe. At a 20% budget, raw image difference found 37.9% of severe components, the random forest 27.6%, and a frozen image encoder 15.5%. These are exploratory mask-derived results, not directly comparable with the main polygon experiment.

To reproduce that pilot, install `pip install -e ".[mirror,demo,test]"`, then run:

```powershell
python -m disaster_triage.prepare_parquet --parquet C:\path\to\train-0000.parquet --out data\pilot --max-tiles-per-event 8
triage-train --manifest data\pilot\manifest.csv --event-prefix hurricane- --test-event hurricane-michael --out results-pilot
```

For the optional frozen-image-encoder comparison, install the `vision` extra following [PyTorch's setup guide](https://pytorch.org/get-started/locally/), then run:

```powershell
python -m disaster_triage.train_embeddings --manifest data\processed\manifest.csv --test-event YOUR_HELD_OUT_EVENT --out results-embeddings
```

## Limits and next steps

The xBD sample is historical and does not cover every tile or a separate external dataset. It also assumes building locations are already known. Before any real use, I would test on data from a different source, reserve a separate event for model selection, check calibration, investigate specific failure cases, and include a way to locate buildings. Nothing here should direct emergency response or certify structural safety.

Code: [MIT license](LICENSE). Data: xBD / xView2, Defense Innovation Unit and Carnegie Mellon Software Engineering Institute, [CC BY-NC-SA 4.0 terms](https://xview2.org/terms). No xBD imagery or annotations are redistributed in this repository.
