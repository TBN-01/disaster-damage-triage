"""Small, inspectable image features for a first reproducible model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def image_features(pre_path: str | Path, post_path: str | Path) -> tuple[np.ndarray, float]:
    with Image.open(pre_path) as image:
        pre = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    with Image.open(post_path) as image:
        post = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if pre.shape != post.shape:
        raise ValueError("Before/after crops have different shapes")

    delta = np.abs(post - pre)
    chunks = [
        pre.mean(axis=(0, 1)), post.mean(axis=(0, 1)),
        pre.std(axis=(0, 1)), post.std(axis=(0, 1)),
        delta.mean(axis=(0, 1)), delta.std(axis=(0, 1)),
        np.quantile(delta, [0.5, 0.9, 0.99], axis=(0, 1)).ravel(),
    ]
    # A coarse spatial grid keeps some information about where the change occurs.
    for patch in np.array_split(delta, 4, axis=0):
        for cell in np.array_split(patch, 4, axis=1):
            chunks.append(cell.mean(axis=(0, 1)))
    return np.concatenate(chunks).astype(np.float32), float(delta.mean())

