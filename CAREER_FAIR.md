# Notes for talking about the project

Don't try to recite the whole README. Open the app, show two images, and explain the question first.

## A short version

“I used before-and-after satellite images to test a review queue for damaged buildings. If a team only has time to check 20% right away, can the images help choose which buildings to look at first? I compared a simple image-change score with a trained model and tested them on disasters they had not seen. The simple method usually did better, but the results changed a lot by disaster. So the project is really about testing whether a ranking holds up, not claiming I built an automatic damage detector.”

## If someone wants a number

In the Santa Rosa wildfire sample, the simple method put **288 of 361 severe cases** in the first **421 reviews** (20% of the buildings). The trained random forest put **116** there. Across ten held-out disasters, image change won eight, lost one, and tied one. In the sampled Mexico earthquake event, both methods found **zero of seven** severe cases in the first 20%. Don't call the wildfire result “80% accurate”; it is **79.8% recall at a 20% review budget** on that sample.

## Why this is a data-science project

The labels for these *past* disasters are public, which lets me check the queue. After a new disaster, they would not be known yet. The data-science part is deciding how to rank limited review work, setting up a fair test, comparing a simple baseline with a model, and looking honestly at the mistakes.

## Questions I would be ready for

**Why not split the buildings randomly?** Nearby buildings often come from the same satellite tile and disaster. A random split can make the test look easier than a genuinely new event. I left out whole disasters instead.

**Why did the simpler approach win?** I don't know for certain. The forest may have learned patterns that did not transfer. It actually did better on Hurricane Matthew, so I would not claim the simple method always wins.

**Can someone use this after a real disaster?** No. It assumes building outlines are already known, and image changes can come from shadows, smoke, or alignment. A qualified person would need to review results, and the method needs external testing first.

**What would I do next?** Check the individual errors more carefully, test on imagery from another source, and reserve a separate event for choosing model settings.

## Resume version

Built a local satellite-imagery review demo and tested two building-ranking methods across ten held-out xBD disasters, showing where a simple baseline beat a trained model—and where both failed.
