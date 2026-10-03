# Career-fair explanation

## 30-second version

“I built a disaster-response data-science project using before-and-after satellite images. It ranks buildings for human review, not safety decisions. I tested it by holding out entire disasters, because buildings from the same event are too similar for a random split to be convincing. On one wildfire, a simple image-change score put about 80% of severe cases in the first 20% reviewed. Across ten disasters the results varied a lot, including an earthquake sample where it found none. I built a local review demo so people can try the workflow and see where it fails.”

## What I actually built

1. Parsed the xBD polygon labels and matched each labeled building to the before/after satellite tile.
2. Cropped paired images around the building, including nearby visual context.
3. Defined “severe” as `major-damage` or `destroyed`; excluded unclassified buildings.
4. Built two rankings: mean absolute pixel change, and a random forest trained on color/change summaries and a coarse 4×4 spatial change grid.
5. Held out every building in each of ten disasters in turn during training.
6. Evaluated with a practical constraint: if reviewers can inspect only 20% of buildings, what fraction of severe cases appear in that queue?
7. Built an interactive local demo with a blind review queue, saved human decisions, an error gallery, and individual crop pairs. The portable example uses synthetic imagery; real xBD images remain local.

## Why this is a data-science problem

The public dataset contains historical *answers* for research and evaluation. After a new real-world disaster, those building damage labels do not exist yet; inspecting imagery and checking sites takes time. The project tests whether visual information could prioritize that work. The goal is a decision-support ranking, not a mechanic's diagnosis, engineering safety assessment, or a claim that a satellite image proves a building is safe.

## Numbers to remember

- 24,465 graded building crops from ten disasters, sampled from up to 30 image tiles per event with seed 42.
- 22,361 buildings from nine events used for training; 2,104 buildings from Santa Rosa wildfire held out.
- 361 held-out buildings have severe-damage labels.
- Reviewing the top 421 buildings by simple image change found 288 severe cases: **79.8% recall at 20% reviewed**.
- The trained random forest found 116 severe cases: **32.1% recall at 20% reviewed**.
- Across ten event holdouts, mean event recall at the same budget was **41.4% for image change** and **26.1% for the random forest**. Image change won eight events, lost one, and tied one at zero.
- Results were highly variable: the simple method found **0 of 7** severe labels in the sampled Mexico earthquake event. It is not ready for operational use.
- A random queue would find about 20% of severe cases on average, but that is only a reference expectation.

## Questions you may get

**Why did the trained model lose?** It may have learned visual patterns from training disasters that did not transfer well. That explanation is a hypothesis, not proven causation. On Hurricane Matthew it actually beat the image-change baseline, which is why the per-event scorecard matters.

**Why not random train/test splitting?** Buildings from one satellite tile and disaster share lighting, geography, damage patterns, and imaging conditions. A random split would make the test unrealistically similar to training. Holding out the whole event is a tougher, more relevant check.

**How accurate is it?** I would not describe this as “80% accurate.” The specific result is that 80% of severely damaged buildings in one held-out wildfire sample appeared within the first 20% of the review queue. It doesn't measure safety certification, building detection, or performance on a future disaster.

**What would you improve next?** Reserve a separate event for model selection, investigate specific errors rather than guessing their cause, and validate on an external dataset before any field use.

## Resume bullet

Built a reproducible satellite-imagery triage pipeline for 24K xBD building annotations, evaluated rankings across ten held-out disasters, and created a local human-review demo with saved decisions and error analysis.

## Responsible-use boundary

This is a retrospective research demonstration using known building outlines and licensed imagery. It should not direct emergency response or certify structural safety without independent validation and qualified human review.
