# Disaster Damage Triage

I wanted to see if before-and-after satellite images could help answer a practical question after a disaster: **if people can only review some buildings right away, which ones should they look at first?**

This project ranks buildings for *human review*. It does not decide whether a building is safe, and it does not find buildings from scratch—the building locations come from the xBD dataset.

![Results from the held-out Santa Rosa wildfire](assets/held-out-results.svg)

## What I built

- A pipeline that pairs before-and-after images for labeled buildings in [xBD](https://xview2.org/dataset).
- Two ways to order a review queue: a straightforward image-change score and a trained random forest.
- A test that holds out an entire disaster at a time, so the model is evaluated on events it did not train on.
- A local app where you can change the review budget, compare disasters, review image pairs before seeing their labels, and look through mistakes.

For the main test, I sampled **24,465 labeled buildings across ten disasters**.

The app includes a small **synthetic practice set** so you can try it without downloading the large xBD dataset. Those made-up images are only for the interactive demo; the results below come from real xBD labels.

If you are showing the project to someone, the app has a simple path: **read the story → try a review → explore the full results**. The app has a 2-minute tour at the top. In the review, you can write down what you noticed before the label is revealed. That note stays on your computer.

## What happened

In the Santa Rosa wildfire sample, the simple image-change ranking found **288 of 361 severely damaged buildings** in the first **421 reviews**—the top 20% of buildings. The random forest found **116 of 361** in the same number of reviews.

I then repeated the test with each of ten disasters held out in turn. The simple method did better than the random forest on eight, worse on one, and tied on one. But it was far from consistent: in the sampled Mexico earthquake event, **neither method found any of the seven severe cases** in the first 20% reviewed. The [full event-by-event results](assets/cross-event-metrics.csv) are here.

That is the main lesson of the project for me: a model looking good on one disaster is not enough. Even the simpler method, which usually did better here, would need much more testing before anyone should use it to guide a real response.

Three choices mattered: I measured **severe cases found within a review budget** rather than overall accuracy, held out **whole disasters** instead of random buildings, and kept the simple score next to the trained model. Those choices made the weak spots easier to see. The app lets you explore the [budget tradeoff](assets/review-budget-curves.csv) from 5% to 50% without sharing building-level predictions or imagery.

## Try it on your computer

1. [Download the project as a ZIP](https://github.com/TBN-01/disaster-damage-triage/archive/refs/heads/main.zip) and unzip it, or clone the repo.
2. Install Python 3.10 or newer if it is not already installed. On Windows, make sure **Add Python to PATH** is checked during installation.
3. Open the unzipped project folder. On Windows, double-click `run-demo.cmd`. On macOS/Linux, open a terminal in that folder and run `sh run-demo.sh`.
4. Wait for the local web address to appear, usually `http://localhost:8501`, and open it in your browser if it does not open automatically. Keep the launcher window open while using the app.

The first launch needs internet to install packages. After that, the app runs locally. No account or xBD download is needed to try the synthetic review demo. Your review choices are saved on your computer, not uploaded anywhere. To stop the app, close the launcher window or press Ctrl+C in it.

### A quick path through the app

1. In **The story**, read the Santa Rosa result and the limitations. You can explain the point in one sentence: *I tested whether a simple image-change ranking could help people find more severe cases when they only have time to review part of a disaster.*
2. In **Try a review**, pick **Practice images** for a quick demo. If you have already run the xBD pipeline on this computer, you can choose **Real local images** instead. These two sets keep separate answers. Choose a queue order, compare the before/after pair, make your call, and click **Save my answer and reveal the label**. Then click **Review the next building**. The building selector shows its place in the queue; the full ID is available underneath if you need it. You can export your answers as a CSV. For another practice run, use **Start the practice review over** after exporting anything you want to keep.
3. In **Explore the results**, pick a disaster and move the review-budget slider. The counts and chart change with your choice. Try Santa Rosa, then Mexico Earthquake to see why one good result is not enough. If you have local xBD results, scroll down to inspect real building images and mistakes from the held-out event.

The review exercise reveals historical or simulated labels *after* you answer. It never decides whether a building is safe.

To explore **real building images**, you need to get xBD separately and run the data pipeline. The dataset is too large to include here, and its images have a separate license. The exact commands, sampling choices, metrics, and optional experiments are in [METHODS.md](METHODS.md).

## What this does *not* prove

This is a portfolio experiment, not an emergency-response tool. It uses known building outlines and a sampled historical dataset. Lighting, shadows, smoke, image alignment, and different types of disasters can all affect the ranking. The score is **not** a damage probability or a structural-safety assessment.

The code is [MIT licensed](LICENSE). xBD imagery and annotations are **CC BY-NC-SA 4.0** and are not included in this repo. Dataset attribution and links are in [METHODS.md](METHODS.md).
