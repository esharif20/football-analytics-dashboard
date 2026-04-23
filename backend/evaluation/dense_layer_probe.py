"""Dense layer-wise probing over every transformer layer in the cache.

Gaps 2B in the plan: the existing layer-wise data uses only 8 probed layers
(step=4). Hidden states are cached at every layer, so we can re-probe all 29
layers without any model reload. Writes `dense_layer_probe.json` alongside
existing artifacts.

Usage:
    python3 -m backend.evaluation.dense_layer_probe \
        --cache-dir eval_output/dissertation/probing_new_run/hidden_cache \
        --output eval_output/dissertation/probing_new_run/rigor/dense_layer_probe.json \
        --modalities d v d+v nec+v \
        --n-seeds 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from backend.evaluation.probing_rigor import _parse_task_modality

logger = logging.getLogger(__name__)


def dense(cache_dir: Path, output_path: Path, modalities: list[str],
          n_seeds: int = 5, test_frac: float = 0.2) -> None:
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    want = set(modalities)
    out: dict[str, Any] = {}
    for npz in sorted(cache_dir.glob("hidden_*.npz")):
        parsed = _parse_task_modality(npz)
        if not parsed:
            continue
        task, mod = parsed
        if mod not in want:
            continue
        data = np.load(npz, allow_pickle=True)
        hidden = data["hidden"]          # (N, L, D)
        labels = np.asarray(data["labels"])
        N, L, D = hidden.shape
        classes = sorted(set(labels.tolist()))
        if len(classes) < 2:
            logger.warning("skipping %s/%s (only one class present)", task, mod)
            continue

        le = LabelEncoder().fit(classes)
        y_all = le.transform(labels)
        min_class = min((y_all == c).sum() for c in range(len(classes)))
        n_splits = min(n_seeds, max(2, min_class))
        splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_frac,
                                          random_state=0)

        logger.info("==== %s / %s  N=%d L=%d D=%d  n_splits=%d ====",
                    task, mod, N, L, D, n_splits)
        layer_records: dict[str, dict[str, float]] = {}
        for layer in range(L):
            X = hidden[:, layer, :]
            f1s = []
            for _, (tr, te) in enumerate(splitter.split(np.zeros(N), y_all)):
                scaler = StandardScaler().fit(X[tr])
                Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
                clf = LogisticRegressionCV(
                    Cs=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5,
                    max_iter=1000, class_weight="balanced",
                    scoring="f1_macro",
                )
                clf.fit(Xtr, y_all[tr])
                pred = clf.predict(Xte)
                f1s.append(float(f1_score(y_all[te], pred,
                                           average="macro", zero_division=0)))
            layer_records[str(layer)] = {
                "mean_f1": round(float(np.mean(f1s)), 4),
                "std_f1":  round(float(np.std(f1s)),  4),
                "n_seeds": n_splits,
            }
            if layer % 4 == 0 or layer == L - 1:
                logger.info("  L%-2d mean %.3f ± %.3f",
                            layer,
                            layer_records[str(layer)]["mean_f1"],
                            layer_records[str(layer)]["std_f1"])

        out.setdefault(task, {})[mod] = layer_records
        Path(output_path).write_text(json.dumps(out, indent=2))
        logger.info("saved partial %s", output_path)

    Path(output_path).write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--modalities", nargs="+", default=["d", "v", "d+v", "nec+v"])
    ap.add_argument("--n-seeds", type=int, default=5)
    args = ap.parse_args()
    dense(Path(args.cache_dir), Path(args.output), args.modalities, args.n_seeds)
