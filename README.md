# Disaster Damage Triage

I wanted to see if before-and-after satellite images could help answer a practical question after a disaster: **if people can only review some buildings right away, which ones should they look at first?**

This project ranks buildings for *human review*. It does not decide whether a building is safe, and it does not find buildings from scratch—the building locations come from the xBD dataset.

![Results from the held-out Santa Rosa wildfire](assets/held-out-results.svg)

## What I built

- A pipeline that pairs before-and-after images for labeled buildings in [xBD](https://xview2.org/dataset).
- Two ways to order a review queue: a straightforward image-change score and a trained random forest.
- A test that holds out an entire disaster at a time, so the model is evaluated on events it did not train on.
- A local app where you can explore the results, review image pairs before seeing their labels, and look through mistakes.

For the main test, I sampled **24,465 labeled buildings across ten disasters**.

The app includes a small **synthetic practice set** so you can try it without downloading the large xBD dataset. Those made-up images are only for the interactive demo; the results below come from real xBD labels.

## What happened

In the Santa Rosa wildfire sample, the simple image-change ranking found **288 of 361 severely damaged buildings** in the first **421 reviews**—the top 20% of buildings. The random forest found **116 of 361** in the same number of reviews.

I then repeated the test with each of ten disasters held out in turn. The simple method did better than the random forest on eight, worse on one, and tied on one. But it was far from consistent: in the sampled Mexico earthquake event, **neither method found any of the seven severe cases** in the first 20% reviewed. The [full event-by-event results](assets/cross-event-metrics.csv) are here.

That is the main lesson of the project for me: a model looking good on one disaster is not enough. Even the simpler method, which usually did better here, would need much more testing before anyone should use it to guide a real response.

## Try it on your computer

1. Download or clone this repo.
2. Install Python 3.10 or newer.
3. On Windows, double-click `run-demo.cmd`. On macOS/Linux, run `sh run-demo.sh` in the project folder.

The first launch needs internet to install packages. After that, the app runs locally. No account or xBD download is needed to try the synthetic review demo. Your review choices are saved on your computer, not uploaded anywhere.

To explore **real building images**, you need to get xBD separately and run the data pipeline. The dataset is too large to include here, and its images have a separate license. The exact commands, sampling choices, metrics, and optional experiments are in [METHODS.md](METHODS.md).

## What this does *not* prove

This is a portfolio experiment, not an emergency-response tool. It uses known building outlines and a sampled historical dataset. Lighting, shadows, smoke, image alignment, and different types of disasters can all affect the ranking. The score is **not** a damage probability or a structural-safety assessment.

The code is [MIT licensed](LICENSE). xBD imagery and annotations are **CC BY-NC-SA 4.0** and are not included in this repo. Dataset attribution and links are in [METHODS.md](METHODS.md).
