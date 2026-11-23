"""Download + data-parallel preprocessing for 5-grade DR training.

Datasets (public, from HuggingFace mirrors — no Kaggle token needed):
  * APTOS 2019 Blindness Detection (3,662 graded fundus images)
  * IDRiD Disease Grading (413 train / 103 test, graded by ophthalmologists)

Per image (sequential): decode -> circular crop -> pad to square -> resize 512x512
Across images (data-parallel): multiprocessing.Pool, one task per image.

Usage:
    python -m src.prepare_dr_data --data-root data --workers 8 --benchmark
Outputs:
    data/dr/images/<id>.jpg   preprocessed 512x512 images
    data/dr/splits.csv        id, label, source, split (train / val / test)
    results/preprocess_benchmark.json  (with --benchmark)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HF = "https://huggingface.co/datasets/"
IDRID = {
    "idrid_train": (HF + "amin-nejad/idrid-disease-grading/resolve/main/data/"
                    "train-00000-of-00001-d81b05cdbfbe95cd.parquet", "label", None),
    "idrid_test": (HF + "amin-nejad/idrid-disease-grading/resolve/main/data/"
                   "test-00000-of-00001-3eaaa0286c6f240e.parquet", "label", None),
}
# name -> (url or local glob, label column, remap to 0=NoDR..4=PDR or None)
VARIANTS = {
    # APTOS mirror pre-resized to 224x224 (fast download; fine for ViT @224)
    "base": {"aptos_train": (HF + "bumbledeep/aptos/resolve/main/data/"
                             "train-00000-of-00001.parquet", "label_code", None),
             **IDRID},
    # full-resolution APTOS (labels stored as Mild,Moderate,NoDR,PDR,Severe) + Messidor-2
    # (external test). Parquet shards are fetched beforehand into data/raw/.
    "hires": {"aptos_train": ("aptos_hires/*.parquet", "label", {0: 1, 1: 2, 2: 0, 3: 4, 4: 3}),
              **IDRID,
              "messidor2_external": ("messidor2/*.parquet", "label", None)},
}
OUT_SIZE = 512
VAL_FRACTION = 0.15
SEED = 42


# ----- per-image pipeline (stateless -> safe to run in parallel) -------------

def circle_crop(arr: np.ndarray, thresh: int = 15) -> np.ndarray:
    """Tight-crop the fundus disk out of the black border."""
    mask = arr.mean(axis=-1) > thresh
    if mask.sum() < 100:
        return arr
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return arr[r0:r1 + 1, c0:c1 + 1]


def pad_square(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    s = max(h, w)
    out = np.zeros((s, s, 3), dtype=arr.dtype)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = arr
    return out


def preprocess_bytes(raw: bytes, size: int = OUT_SIZE) -> Image.Image:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.draft("RGB", (size * 2, size * 2))  # fast JPEG downscale on decode
    if max(img.size) > size * 3:
        img.thumbnail((size * 3, size * 3))
    arr = pad_square(circle_crop(np.asarray(img)))
    return Image.fromarray(arr).resize((size, size), Image.BICUBIC)


def _work(job):
    raw, out_path = job
    preprocess_bytes(raw).save(out_path, quality=95)
    return out_path


# ----- driver ----------------------------------------------------------------

def download(url: str, dest: Path) -> Path:
    """Plain HTTPS download (requests ships its own CA bundle, unlike some macOS Pythons)."""
    import requests
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url.split('/datasets/')[1].split('/resolve')[0]} -> {dest}", flush=True)
    tmp = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    return dest


# multi-shard datasets for the "hires" variant: local folder -> HuggingFace dataset repo
SHARDED = {"aptos_hires": "EslamHasan/APTOS2019DiabeticRetinopathy",
           "messidor2": "sngsfydy/Messidor2"}


def fetch_shards(root: Path, folder: str) -> None:
    from huggingface_hub import HfApi
    files = [s.rfilename for s in HfApi().dataset_info(SHARDED[folder]).siblings
             if s.rfilename.endswith(".parquet")]
    for f in files:
        download(f"{HF}{SHARDED[folder]}/resolve/main/{f}", root / "raw" / folder / Path(f).name)


def stratified_val(labels: np.ndarray, frac: float, rng) -> np.ndarray:
    is_val = np.zeros(len(labels), dtype=bool)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        is_val[idx[: int(round(len(idx) * frac))]] = True
    return is_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--variant", choices=list(VARIANTS), default="base")
    ap.add_argument("--benchmark", action="store_true",
                    help="also time 1 worker vs N workers on a 400-image subset")
    args = ap.parse_args()

    root = Path(args.data_root)
    suffix = "" if args.variant == "base" else f"_{args.variant}"
    img_dir = root / "dr" / f"images{suffix}"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows, jobs = [], []
    for name, (src, label_col, remap) in VARIANTS[args.variant].items():
        if src.startswith("http"):
            df = pd.read_parquet(download(src, root / "raw" / f"{name}.parquet"))
        else:
            folder = src.split("/")[0]
            if folder in SHARDED:
                fetch_shards(root, folder)
            shards = sorted((root / "raw").glob(src))
            if not shards:
                raise FileNotFoundError(f"no shards for {name} at data/raw/{src}")
            df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
        labels = df[label_col].astype(int).to_numpy()
        if remap:
            labels = np.vectorize(remap.get)(labels)
        if name.endswith("_test"):
            split = np.array(["test"] * len(df))
        elif name.endswith("_external"):
            split = np.array(["external"] * len(df))
        else:
            # per-source seeded RNG -> the IDRiD train/val split is identical in every variant
            rng = np.random.default_rng([SEED, sum(map(ord, name))])
            split = np.where(stratified_val(labels, VAL_FRACTION, rng), "val", "train")
            base_csv = root / "dr" / "splits.csv"
            if args.variant != "base" and name.startswith("idrid") and base_csv.exists():
                # reuse the base variant's IDRiD assignment so models trained on
                # different variants share exactly the same IDRiD train/val images
                prev = pd.read_csv(base_csv).set_index("id")["split"]
                split = np.array([prev[f"{name}_{i:04d}"] for i in range(len(df))])
        for i, (cell, y, s) in enumerate(zip(df["image"], labels, split)):
            iid = f"{name}_{i:04d}"
            rows.append({"id": iid, "label": int(y), "source": name.split("_")[0], "split": s})
            jobs.append((cell["bytes"], str(img_dir / f"{iid}.jpg")))
        print(f"{name}: {len(df)} images, labels {np.bincount(labels, minlength=5).tolist()}")

    if args.benchmark:
        sub = jobs[:400]
        bench = {}
        for w in sorted({1, 2, 4, args.workers}):
            t = time.perf_counter()
            if w == 1:
                for j in sub:
                    _work(j)
            else:
                with Pool(w) as p:
                    p.map(_work, sub, chunksize=8)
            bench[w] = time.perf_counter() - t
            print(f"  benchmark: {w:2d} worker(s) -> {bench[w]:6.1f}s for {len(sub)} images "
                  f"(speedup {bench[1] / bench[w]:.2f}x)")
        Path("results").mkdir(exist_ok=True)
        json.dump({"images": len(sub), "seconds_by_workers": bench,
                   "speedup_by_workers": {w: bench[1] / s for w, s in bench.items()}},
                  open("results/preprocess_benchmark.json", "w"), indent=1)

    t = time.perf_counter()
    with Pool(args.workers) as p:
        for k, _ in enumerate(p.imap_unordered(_work, jobs, chunksize=8), 1):
            if k % 500 == 0:
                print(f"  {k}/{len(jobs)}")
    print(f"preprocessed {len(jobs)} images with {args.workers} workers "
          f"in {time.perf_counter() - t:.1f}s")

    splits = pd.DataFrame(rows)
    splits.to_csv(root / "dr" / f"splits{suffix}.csv", index=False)
    print(splits.groupby(["split", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
