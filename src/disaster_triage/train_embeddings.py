"""Evaluate a frozen, pretrained image encoder on paired xBD building crops.

ImageNet features are extracted for each crop. A small logistic model learns
from before, after, and absolute feature differences. Whole disaster events
remain separated between training and testing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from .train import evaluate


class PairDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row["pre_crop"]) as source:
            pre = self.transform(source.convert("RGB"))
        with Image.open(row["post_crop"]) as source:
            post = self.transform(source.convert("RGB"))
        return pre, post


def extract_embeddings(frame: pd.DataFrame, batch_size: int = 32) -> tuple[np.ndarray, str]:
    weights = ResNet18_Weights.DEFAULT
    encoder = resnet18(weights=weights)
    encoder.fc = nn.Identity()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder.to(device).eval()
    loader = DataLoader(PairDataset(frame, weights.transforms()), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    batches = []
    with torch.inference_mode():
        for before, after in loader:
            before, after = before.to(device), after.to(device)
            a = encoder(before).cpu().numpy()
            b = encoder(after).cpu().numpy()
            batches.append(np.concatenate((a, b, np.abs(b - a)), axis=1))
    return np.concatenate(batches).astype(np.float32), device


def train_embeddings(manifest_path: Path, test_event: str, out_dir: Path,
                     event_prefix: str | None = None, batch_size: int = 32) -> dict:
    frame = pd.read_csv(manifest_path)
    if event_prefix:
        frame = frame[frame["event"].str.startswith(event_prefix)].copy()
    if test_event not in set(frame["event"]):
        raise ValueError(f"Unknown test event: {test_event}")
    train_mask = (frame["event"] != test_event).to_numpy()
    test_mask = ~train_mask
    y = frame["severe"].to_numpy(dtype=int)
    if len(set(y[train_mask])) < 2 or len(set(y[test_mask])) < 2:
        raise ValueError("Both training and test events need both classes")

    x, device = extract_embeddings(frame, batch_size)
    model = make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=42,
    ))
    model.fit(x[train_mask], y[train_mask])
    scores = model.predict_proba(x[test_mask])[:, 1]
    results = {
        "method": "Frozen ImageNet ResNet-18 embeddings from before and after crops; balanced logistic classifier",
        "device": device,
        "test_event": test_event,
        "train_events": sorted(frame.loc[train_mask, "event"].unique().tolist()),
        "train_samples": int(train_mask.sum()),
        "test_metrics": evaluate(y[test_mask], scores),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = frame.loc[test_mask].copy()
    predictions["model_score"] = scores
    predictions.sort_values("model_score", ascending=False).to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--test-event", required=True)
    parser.add_argument("--event-prefix")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(train_embeddings(args.manifest, args.test_event, args.out,
                                       args.event_prefix, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
