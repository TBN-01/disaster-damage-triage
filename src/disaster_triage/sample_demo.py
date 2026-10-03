"""Create a tiny, explicitly synthetic image set for the portable UI demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def make_sample_demo(directory: Path) -> pd.DataFrame:
    directory.mkdir(parents=True, exist_ok=True)
    table = directory / "synthetic_predictions.csv"
    if table.exists():
        frame = pd.read_csv(table)
        if len(frame) == 24 and all(
            Path(path).exists() for column in ("pre_crop", "post_crop") for path in frame[column]
        ):
            return frame

    rng = np.random.default_rng(42)
    rows = []
    for index in range(24):
        severe = index in {2, 4, 7, 9, 13, 17, 20, 22}
        nuisance = index in {1, 6, 15, 19}
        missed = index in {9, 20}
        ground = tuple(int(value) for value in rng.integers(120, 170, size=3))
        roof = tuple(int(value) for value in rng.integers(85, 160, size=3))
        before = Image.new("RGB", (160, 160), ground)
        draw = ImageDraw.Draw(before)
        draw.rectangle((27, 25, 133, 132), fill=(95, 106, 91))
        draw.polygon([(38, 52), (80, 25), (123, 52), (123, 112), (38, 112)], fill=roof, outline=(230, 221, 190), width=3)
        draw.line((80, 25, 80, 112), fill=(60, 65, 65), width=3)
        after = before.copy()
        draw = ImageDraw.Draw(after)
        if severe:
            if missed:
                draw.polygon([(63, 59), (94, 53), (102, 79), (77, 84)], fill=(62, 57, 55))
                draw.line((55, 87, 105, 58), fill=(35, 33, 32), width=3)
            else:
                draw.polygon([(39, 53), (106, 38), (126, 70), (104, 113), (52, 99)], fill=(58, 53, 48))
                for offset in range(4):
                    draw.line((42 + offset * 16, 65, 89 + offset * 7, 104), fill=(185, 169, 140), width=3)
        elif nuisance:
            draw.polygon([(30, 20), (127, 38), (138, 130), (48, 112)], fill=(48, 55, 63))
        else:
            draw.rectangle((34, 44, 124, 116), outline=(205, 197, 182), width=2)
        pre_path = directory / f"synthetic_{index:03d}_pre.png"
        post_path = directory / f"synthetic_{index:03d}_post.png"
        before.save(pre_path)
        after.save(post_path)
        rows.append({
            "sample_id": f"DEMO-{index + 1:03d}", "event": "synthetic-demo",
            "tile": f"demo-tile-{index // 6}", "severe": int(severe),
            "subtype": "simulated severe" if severe else "simulated not severe",
            "pre_crop": str(pre_path), "post_crop": str(post_path),
            "baseline_score": float(0.32 if missed else 0.87 if severe else 0.90 if nuisance else 0.12 + index / 250),
            "model_score": float(0.30 if missed else 0.68 if severe else 0.77 if nuisance else 0.10 + index / 300),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(table, index=False)
    return frame
