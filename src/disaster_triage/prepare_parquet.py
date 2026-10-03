"""Prepare an initial study from a public xBD image/mask Parquet mirror.

The mirror contains semantic masks instead of original building polygons. This
adapter approximates buildings with connected components and excludes components
with mixed damage labels. For final evaluation, prefer official polygon labels.
"""

from __future__ import annotations

import argparse
import io
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from scipy import ndimage

from .prepare import event_from_tile, square_crop


CLASS_NAMES = {1: "no-damage", 2: "minor-damage", 3: "major-damage", 4: "destroyed"}


def prepare_parquet(parquet_path: Path, out_dir: Path, max_tiles_per_event: int | None = None,
                    crop_size: int = 96, min_area: int = 12, max_mixed_fraction: float = 0.1) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = out_dir / "crops"
    crop_dir.mkdir(exist_ok=True)
    tile_dir = out_dir / "tiles"
    tile_dir.mkdir(exist_ok=True)
    tile_counts: Counter = Counter()
    rows: list[dict] = []
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(batch_size=4):
        for record in batch.to_pylist():
            tile = Path(record["image_name"]).name
            event = event_from_tile(tile)
            if max_tiles_per_event is not None and tile_counts[event] >= max_tiles_per_event:
                continue
            pre = Image.open(io.BytesIO(record["t1_image"]["bytes"])).convert("RGB")
            post = Image.open(io.BytesIO(record["t2_image"]["bytes"])).convert("RGB")
            pre_tile = tile_dir / f"{tile}_pre.jpg"
            post_tile = tile_dir / f"{tile}_post.jpg"
            pre.save(pre_tile, quality=88)
            post.save(post_tile, quality=88)
            pre_mask = np.asarray(Image.open(io.BytesIO(record["t1_mask"]["bytes"])) )
            post_mask = np.asarray(Image.open(io.BytesIO(record["t2_mask"]["bytes"])) )
            components, count = ndimage.label(pre_mask > 0)
            objects = ndimage.find_objects(components)
            for component_id in range(1, count + 1):
                extent = objects[component_id - 1]
                if extent is None:
                    continue
                ys, xs = extent
                component = components[extent] == component_id
                area = int(component.sum())
                if area < min_area:
                    continue
                labels = post_mask[extent][component]
                labels = labels[np.isin(labels, list(CLASS_NAMES))]
                if len(labels) == 0:
                    continue
                counts = np.bincount(labels, minlength=5)
                subtype_code = int(np.argmax(counts))
                if 1 - counts[subtype_code] / len(labels) > max_mixed_fraction:
                    continue
                subtype = CLASS_NAMES[subtype_code]
                box = (float(xs.start), float(ys.start), float(xs.stop), float(ys.stop))
                sample_id = f"{tile}_{component_id:05d}"
                pre_path = crop_dir / f"{sample_id}_pre.jpg"
                post_path = crop_dir / f"{sample_id}_post.jpg"
                square_crop(pre, box, crop_size).save(pre_path, quality=90)
                square_crop(post, box, crop_size).save(post_path, quality=90)
                rows.append({
                    "sample_id": sample_id, "event": event, "tile": tile,
                    "subtype": subtype, "severe": int(subtype_code in (3, 4)),
                    "pre_crop": str(pre_path.resolve()), "post_crop": str(post_path.resolve()),
                    "pre_tile": str(pre_tile.resolve()), "post_tile": str(post_tile.resolve()),
                    "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3],
                    "source": "semantic-mask-connected-component",
                    "component_area_px": area,
                })
            tile_counts[event] += 1
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No usable building components found")
    frame.to_csv(out_dir / "manifest.csv", index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--out", default=Path("data/processed"), type=Path)
    parser.add_argument("--max-tiles-per-event", type=int)
    args = parser.parse_args()
    frame = prepare_parquet(args.parquet, args.out, args.max_tiles_per_event)
    print(frame.groupby("event")["severe"].agg(["count", "sum"]).to_string())
    print(f"Manifest: {args.out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
