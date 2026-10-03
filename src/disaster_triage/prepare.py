"""Turn xBD tiles and polygon labels into paired building crops.

The first version assumes building footprints are already known. It does not
attempt building detection. It uses post-event labels only as targets, never as
model features.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


SEVERE = {"major-damage", "destroyed"}
NOT_SEVERE = {"no-damage", "minor-damage"}
POINT = re.compile(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


def polygon_box(wkt: str) -> tuple[float, float, float, float]:
    """Return a pixel bounding box from an xBD POLYGON WKT string."""
    if not wkt.startswith("POLYGON"):
        raise ValueError(f"Unsupported geometry: {wkt[:30]}")
    points = [(float(x), float(y)) for x, y in POINT.findall(wkt)]
    if len(points) < 3:
        raise ValueError("Polygon has fewer than three points")
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)


def event_from_tile(tile: str) -> str:
    match = re.match(r"(.+)_\d+$", tile)
    return match.group(1) if match else tile


def image_paths(label_file: Path) -> tuple[Path, Path, str]:
    tile = label_file.stem.removesuffix("_post_disaster")
    image_dir = label_file.parent.parent / "images"
    return (
        image_dir / f"{tile}_pre_disaster.png",
        image_dir / f"{tile}_post_disaster.png",
        tile,
    )


def square_crop(image: Image.Image, box: tuple[float, float, float, float], size: int) -> Image.Image:
    """Crop a square around a building, retaining nearby visual context."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0, 24) * 2.5
    half = side / 2
    bounds = (round(cx - half), round(cy - half), round(cx + half), round(cy + half))
    crop = image.crop(bounds)
    return ImageOps.fit(crop.convert("RGB"), (size, size), method=Image.Resampling.BILINEAR)


def prepare(data_root: Path, out_dir: Path, crop_size: int = 96, max_tiles_per_event: int | None = None, seed: int = 42) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    crops = out_dir / "crops"
    crops.mkdir(exist_ok=True)
    labels = sorted(data_root.rglob("*_post_disaster.json"))
    if not labels:
        raise FileNotFoundError(f"No xBD post-disaster labels found under {data_root}")
    if max_tiles_per_event is not None:
        by_event: dict[str, list[Path]] = {}
        for label in labels:
            by_event.setdefault(event_from_tile(label.stem.removesuffix("_post_disaster")), []).append(label)
        rng = random.Random(seed)
        labels = sorted(
            label
            for event_labels in by_event.values()
            for label in rng.sample(event_labels, min(max_tiles_per_event, len(event_labels)))
        )

    rows: list[dict] = []
    for label_file in labels:
        pre_path, post_path, tile = image_paths(label_file)
        event = event_from_tile(tile)
        if not pre_path.exists() or not post_path.exists():
            continue
        document = json.loads(label_file.read_text(encoding="utf-8"))
        features = document.get("features", {}).get("xy", [])
        with Image.open(pre_path) as pre_source, Image.open(post_path) as post_source:
            for index, feature in enumerate(features):
                props = feature.get("properties", {})
                subtype = props.get("subtype")
                if subtype not in SEVERE | NOT_SEVERE:
                    continue
                try:
                    box = polygon_box(feature["wkt"])
                except (KeyError, ValueError):
                    continue
                sample_id = f"{tile}_{index:05d}"
                pre_crop = crops / f"{sample_id}_pre.jpg"
                post_crop = crops / f"{sample_id}_post.jpg"
                square_crop(pre_source, box, crop_size).save(pre_crop, quality=90)
                square_crop(post_source, box, crop_size).save(post_crop, quality=90)
                rows.append({
                    "sample_id": sample_id,
                    "event": event,
                    "tile": tile,
                    "subtype": subtype,
                    "severe": int(subtype in SEVERE),
                    "pre_crop": str(pre_crop.resolve()),
                    "post_crop": str(post_crop.resolve()),
                    "pre_tile": str(pre_path.resolve()),
                    "post_tile": str(post_path.resolve()),
                    "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3],
                })

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No labeled building crops were created. Check the dataset layout and labels.")
    frame.to_csv(out_dir / "manifest.csv", index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", default=Path("data/processed"), type=Path)
    parser.add_argument("--crop-size", default=96, type=int)
    parser.add_argument("--max-tiles-per-event", type=int)
    parser.add_argument("--seed", default=42, type=int, help="Reproducible tile sampling when a cap is used")
    args = parser.parse_args()
    frame = prepare(args.data_root, args.out, args.crop_size, args.max_tiles_per_event, args.seed)
    print(f"Created {len(frame)} crops from {frame['event'].nunique()} events: {args.out / 'manifest.csv'}")
    print(frame.groupby("event")["severe"].agg(["count", "sum"]).to_string())


if __name__ == "__main__":
    main()

