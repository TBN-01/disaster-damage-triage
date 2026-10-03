# Career-fair explanation

## 30-second version

“I built a disaster-response data-science project using before-and-after satellite images. The goal isn't to replace inspectors; it's to rank buildings so a person with limited time can review the most urgent cases first. I tested on an entire wildfire that the model never trained on. A simple image-change score found about 80% of the severely damaged buildings in the first 20% reviewed, while my random-forest model found only 32%. I kept that failure in the project because it shows why out-of-disaster testing matters.”

## What I actually built

1. Parsed the xBD polygon labels and matched each labeled building to the before/after satellite tile.
2. Cropped paired images around the building, including nearby visual context.
3. Defined “severe” as `major-damage` or `destroyed`; excluded unclassified buildings.
4. Built two rankings: mean absolute pixel change, and a random forest trained on color/change summaries and a coarse 4×4 spatial change grid.
5. Held out every building in the Santa Rosa wildfire during training.
6. Evaluated with a practical constraint: if reviewers can inspect only 20% of buildings, what fraction of severe cases appear in that queue?
7. Built an interactive local demo that shows the before/after scene, highlighted review queue, ranked buildings, and individual crop pairs.

## Why this is a data-science problem

The public dataset contains historical *answers* for research and evaluation. After a new real-world disaster, those building damage labels do not exist yet; inspecting imagery and checking sites takes time. The project tests whether visual information could prioritize that work. The goal is a decision-support ranking, not a mechanic's diagnosis, engineering safety assessment, or a claim that a satellite image proves a building is safe.

## Numbers to remember

- 24,465 graded building crops from ten disasters, sampled from up to 30 image tiles per event with seed 42.
- 22,361 buildings from nine events used for training; 2,104 buildings from Santa Rosa wildfire held out.
- 361 held-out buildings have severe-damage labels.
- Reviewing the top 421 buildings by simple image change found 288 severe cases: **79.8% recall at 20% reviewed**.
- The trained random forest found 116 severe cases: **32.1% recall at 20% reviewed**.
- A random queue would find about 20% of severe cases on average, but that is only a reference expectation.

## Questions you may get

**Why did the trained model lose?** It likely learned visual patterns from the training disasters that did not transfer well to the wildfire. That explanation is a hypothesis, not proven causation. The test establishes the performance gap; further per-event and error analysis would be needed to isolate why.

**Why not random train/test splitting?** Buildings from one satellite tile and disaster share lighting, geography, damage patterns, and imaging conditions. A random split would make the test unrealistically similar to training. Holding out the whole event is a tougher, more relevant check.

**How accurate is it?** I would not describe this as “80% accurate.” The specific result is that 80% of severely damaged buildings in one held-out wildfire sample appeared within the first 20% of the review queue. It doesn't measure safety certification, building detection, or performance on a future disaster.

**What would you improve next?** Repeat the leave-one-disaster-out evaluation across all events, reserve a separate event for model selection, examine false positives caused by shadows or misalignment, and validate with an external dataset before any field use.

## Resume bullet

Built a reproducible satellite-imagery triage pipeline for 24K xBD building annotations; evaluated rankings on a disaster-level holdout and surfaced 79.8% of severe wildfire cases within the first 20% of a human-review queue, outperforming a random-forest comparator.

## Responsible-use boundary

This is a retrospective research demonstration using known building outlines and licensed imagery. It should not direct emergency response or certify structural safety without independent validation and qualified human review.
