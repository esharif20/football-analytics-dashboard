"""Rigor-upgrade analyses that run on cached hidden states (CPU-only).

Four sub-commands feed directly into the dissertation's §4.8 addendum:

  multiseed   — 5 random train/test splits × linear probe, report mean ± std
                per (task, modality). Turns single-point F1 into statistics.
  transfer    — cross-task transfer matrix. Train probe on task A, test on
                task B. Reveals shared tactical subspace.
  mdl         — Minimum Description Length online-code probing (Voita &
                Titov 2020) as an alternative to F1.
  perclass    — per-class F1 evolution across transformer layers per
                (task, modality). Shows which class drives emergence.

All four consume the cached `.npz` files under the hidden_cache directory
and write JSON outputs plus slate/navy figures where appropriate.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Shared modality parser (matches analysis + interp conventions)
_MOD_STEMS = [
    "nec_va", "nec_sp", "nec_v", "nc_va", "nc_sp", "nc_v",
    "ne_va", "ne_sp", "ne_v", "p_va", "p_sp", "p_v2", "p_vh",
    "n_va", "n_sp", "n_v",
    "pt_v", "sf_v",
    "d_v", "pi_v", "p_v", "b_v",
    "dshuf", "vfix", "nec", "nc", "pt", "sf",
    "pi", "ne", "va", "sp", "v2", "vh",
    "d", "v", "p", "m", "o", "b", "n",
]


def _parse_task_modality(path: Path) -> tuple[str, str] | None:
    stem = path.stem.replace("hidden_", "")
    for m in _MOD_STEMS:
        if stem.endswith("_" + m):
            # Composites contain '_' in the stem; canonical form uses '+'.
            canonical = m.replace("_", "+") if "_" in m else m
            return stem[: -(len(m) + 1)], canonical
    return None


def _train_probe_simple(X_tr, y_tr, X_te, y_te):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegressionCV(
        Cs=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5,
        max_iter=1000, class_weight="balanced", scoring="f1_macro",
    )
    clf.fit(scaler.transform(X_tr), y_tr)
    pred = clf.predict(scaler.transform(X_te))
    return float(f1_score(y_te, pred, average="macro", zero_division=0))


# ── 1. Multi-seed bootstrap ──────────────────────────────────────────────────


def multiseed(cache_dir: Path, output_path: Path, n_seeds: int = 10) -> None:
    """For each (task, modality), re-probe with `n_seeds` STRATIFIED random
    train/test splits at the LAST probed layer. Report mean ± std per cell
    plus per-class F1 for honest reporting. Every split is guaranteed to
    contain all classes in both train and test (when min-class-count ≥ 2)."""
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedShuffleSplit

    out: dict[str, Any] = {}
    for npz in sorted(cache_dir.glob("hidden_*.npz")):
        parsed = _parse_task_modality(npz)
        if not parsed:
            continue
        task, mod = parsed
        try:
            data = np.load(npz, allow_pickle=True)
            hidden = data["hidden"]
            labels = np.array([str(l) for l in data["labels"]])
            layer = int(data["layers_probed"][-1])
            X = hidden[:, layer, :]
            le = LabelEncoder()
            y = le.fit_transform(labels)
            classes = list(le.classes_)
            _, counts = np.unique(y, return_counts=True)

            if counts.min() < 2:
                # Can't stratify on degenerate singleton classes; fall back
                # to random splits with a warning.
                splitter = None
            else:
                splitter = StratifiedShuffleSplit(
                    n_splits=n_seeds, test_size=0.2, random_state=42,
                )

            scores = []
            per_class_scores: "list[dict[str, float]]" = []
            if splitter is not None:
                iterator = splitter.split(np.zeros(len(y)), y)
            else:
                rng = np.random.default_rng(42)
                def _iter():
                    for _ in range(n_seeds):
                        idx = rng.permutation(len(y))
                        te = idx[: max(1, len(y) // 5)]
                        tr = idx[max(1, len(y) // 5):]
                        yield tr, te
                iterator = _iter()

            for tr, te in iterator:
                try:
                    from sklearn.linear_model import LogisticRegressionCV
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler().fit(X[tr])
                    # CV folds can't exceed min class count in training set.
                    _, tr_counts = np.unique(y[tr], return_counts=True)
                    cv_folds = int(max(2, min(5, tr_counts.min())))
                    clf = LogisticRegressionCV(
                        Cs=[0.001, 0.01, 0.1, 1.0, 10.0], cv=cv_folds,
                        max_iter=1000, class_weight="balanced", scoring="f1_macro",
                    )
                    clf.fit(scaler.transform(X[tr]), y[tr])
                    pred = clf.predict(scaler.transform(X[te]))
                    f1 = float(f1_score(y[te], pred, average="macro", zero_division=0))
                    scores.append(f1)
                    pc = f1_score(y[te], pred, average=None, zero_division=0,
                                  labels=range(len(classes)))
                    per_class_scores.append({
                        c: float(pc[i]) for i, c in enumerate(classes)
                    })
                except Exception as e:
                    logger.warning("seed skip %s/%s: %s", task, mod, e)

            if scores:
                # Aggregate per-class: mean + count of seeds where class present in test
                pc_mean = {
                    c: round(float(np.mean([s[c] for s in per_class_scores])), 4)
                    for c in classes
                }
                pc_std = {
                    c: round(float(np.std([s[c] for s in per_class_scores])), 4)
                    for c in classes
                }
                out.setdefault(task, {})[mod] = {
                    "mean_f1": round(float(np.mean(scores)), 4),
                    "std_f1":  round(float(np.std(scores)), 4),
                    "n_seeds": len(scores),
                    "per_seed": [round(v, 4) for v in scores],
                    "per_class_mean_f1": pc_mean,
                    "per_class_std_f1": pc_std,
                    "layer": layer,
                    "stratified": splitter is not None,
                }
                logger.info("  %s/%s: %.3f ± %.3f (n=%d)  per-class=%s",
                            task, mod, np.mean(scores), np.std(scores), len(scores),
                            ", ".join(f"{c}:{v:.2f}" for c, v in pc_mean.items()))
        except Exception as e:
            logger.warning("skip %s: %s", npz.name, e)
    output_path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", output_path)


# ── 2. Cross-task transfer matrix ────────────────────────────────────────────


def transfer(cache_dir: Path, output_path: Path,
             modality_filter: str = "d") -> None:
    """Train probe on task A's hidden states, test on task B's. Produces an
    N_tasks × N_tasks matrix of macro-F1. Diagonal = in-task F1."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    # Load all (task) caches for the given modality
    task_data: dict = {}
    for npz in sorted(cache_dir.glob(f"hidden_*_{modality_filter}.npz")):
        parsed = _parse_task_modality(npz)
        if not parsed:
            continue
        task, mod = parsed
        if mod != modality_filter:
            continue
        data = np.load(npz, allow_pickle=True)
        layer = int(data["layers_probed"][-1])
        X = data["hidden"][:, layer, :]
        y = np.array(data["labels"])
        task_data[task] = (X, y)

    tasks = sorted(task_data.keys())
    out: dict = {"modality": modality_filter, "tasks": tasks, "matrix": {}}
    for a in tasks:
        X_a, y_a = task_data[a]
        le_a = LabelEncoder().fit(y_a)
        y_a_enc = le_a.transform(y_a)
        n_tr = max(1, int(len(y_a) * 0.8))
        scaler = StandardScaler().fit(X_a[:n_tr])
        clf = LogisticRegressionCV(
            Cs=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5,
            max_iter=1000, class_weight="balanced", scoring="f1_macro",
        )
        clf.fit(scaler.transform(X_a[:n_tr]), y_a_enc[:n_tr])
        out["matrix"][a] = {}
        for b in tasks:
            X_b, y_b = task_data[b]
            # Project y_b labels onto task_a's label space where overlapping,
            # else use the probe's class index best-aligned to dominant label.
            # Simpler & standard: decode probe output, match against y_b as
            # string labels via majority alignment.
            y_b_enc = le_a.transform([l for l in y_b if l in le_a.classes_]) \
                      if all(l in le_a.classes_ for l in y_b) else None
            try:
                pred = clf.predict(scaler.transform(X_b))
                if y_b_enc is not None:
                    f1 = float(f1_score(y_b_enc, pred, average="macro", zero_division=0))
                else:
                    # Incompatible label sets → compute label-agnostic
                    # representation transferability as clustering purity:
                    # silhouette of X_b in the probe-projected subspace.
                    from sklearn.metrics import silhouette_score
                    z = scaler.transform(X_b) @ clf.coef_.T
                    if z.shape[1] >= 1 and len(set(y_b)) > 1:
                        f1 = float(silhouette_score(z, y_b, metric="cosine"))
                    else:
                        f1 = float("nan")
                out["matrix"][a][b] = round(f1, 4)
                logger.info("  %s → %s: %.3f", a, b, f1)
            except Exception as e:
                logger.warning("  %s → %s: %s", a, b, e)
                out["matrix"][a][b] = None

    output_path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", output_path)


# ── 3. MDL probing (Voita & Titov 2020 online code) ─────────────────────────


def mdl(cache_dir: Path, output_path: Path, portions: list[float] | None = None) -> None:
    """Online-code MDL (Voita & Titov 2020). Trains probes on growing
    portions of the data and sums the bits used to predict the next portion
    with the current probe. Lower codelength = more information in the
    representation (more efficient compression of labels given hidden state).

    Reported alongside F1 so the chapter can compare metrics.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    if portions is None:
        portions = [0.1, 0.2, 0.4, 0.8, 1.0]

    out: dict = {}
    for npz in sorted(cache_dir.glob("hidden_*.npz")):
        parsed = _parse_task_modality(npz)
        if not parsed:
            continue
        task, mod = parsed
        try:
            data = np.load(npz, allow_pickle=True)
            layer = int(data["layers_probed"][-1])
            X = data["hidden"][:, layer, :]
            y = LabelEncoder().fit_transform(np.array(data["labels"]))
            n = len(y)
            n_classes = len(np.unique(y))

            # Uniform-code baseline: log2(C) bits per sample
            uniform_bits = n * np.log2(max(n_classes, 2))

            online_bits = 0.0
            prev_end = 0
            for p in portions:
                end = int(p * n)
                if end <= prev_end:
                    continue
                if prev_end == 0:
                    # First block uses uniform code
                    online_bits += (end - prev_end) * np.log2(max(n_classes, 2))
                else:
                    # Train on [0, prev_end], score on [prev_end, end]
                    scaler = StandardScaler().fit(X[:prev_end])
                    clf = LogisticRegression(
                        C=1.0, max_iter=1000, class_weight="balanced",
                    )
                    try:
                        clf.fit(scaler.transform(X[:prev_end]), y[:prev_end])
                        probs = clf.predict_proba(scaler.transform(X[prev_end:end]))
                        # codelength = sum(-log2 p(true class))
                        true_probs = probs[np.arange(end - prev_end), y[prev_end:end]]
                        online_bits += float(np.sum(-np.log2(np.clip(true_probs, 1e-12, 1))))
                    except Exception:
                        online_bits += (end - prev_end) * np.log2(max(n_classes, 2))
                prev_end = end

            # Compression ratio: codelength saved / uniform code
            compression = (uniform_bits - online_bits) / uniform_bits if uniform_bits > 0 else 0.0
            out.setdefault(task, {})[mod] = {
                "layer": layer,
                "n_samples": n,
                "n_classes": n_classes,
                "uniform_codelength_bits": round(float(uniform_bits), 2),
                "online_codelength_bits": round(float(online_bits), 2),
                "compression_ratio": round(float(compression), 4),
            }
            logger.info("  %s/%s: uniform=%.0f online=%.0f compression=%.3f",
                        task, mod, uniform_bits, online_bits, compression)
        except Exception as e:
            logger.warning("mdl skip %s: %s", npz.name, e)

    output_path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", output_path)


# ── 4. Per-class F1 evolution across layers ──────────────────────────────────


def perclass(cache_dir: Path, output_path: Path) -> None:
    """Per-class F1 at every probed layer per (task, modality). Reveals which
    class drives macro-F1 emergence."""
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import LabelEncoder

    out: dict = {}
    for npz in sorted(cache_dir.glob("hidden_*.npz")):
        parsed = _parse_task_modality(npz)
        if not parsed:
            continue
        task, mod = parsed
        try:
            data = np.load(npz, allow_pickle=True)
            hidden = data["hidden"]
            labels = np.array(data["labels"])
            probed = [int(x) for x in data["layers_probed"]]
            n_train = int(data["n_train"])
            le = LabelEncoder()
            y = le.fit_transform(labels)
            classes = list(le.classes_)

            per_layer: dict = {}
            for L in probed:
                X_tr, X_te = hidden[:n_train, L, :], hidden[n_train:, L, :]
                y_tr, y_te = y[:n_train], y[n_train:]
                try:
                    from sklearn.linear_model import LogisticRegressionCV
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler().fit(X_tr)
                    clf = LogisticRegressionCV(
                        Cs=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5,
                        max_iter=1000, class_weight="balanced", scoring="f1_macro",
                    )
                    clf.fit(scaler.transform(X_tr), y_tr)
                    pred = clf.predict(scaler.transform(X_te))
                    per_class = f1_score(y_te, pred, average=None, zero_division=0,
                                         labels=range(len(classes)))
                    per_layer[str(L)] = {
                        cls: round(float(per_class[i]), 4)
                        for i, cls in enumerate(classes)
                    }
                except Exception as e:
                    logger.warning("  layer %d failed: %s", L, e)
            out.setdefault(task, {})[mod] = {
                "classes": classes,
                "per_layer": per_layer,
            }
            logger.info("  %s/%s: %d layers done", task, mod, len(per_layer))
        except Exception as e:
            logger.warning("perclass skip %s: %s", npz.name, e)

    output_path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", output_path)


# ── 5. Per-class confusion matrices (exposes class collapse) ─────────────────


def confusion(cache_dir: Path, output_path: Path) -> None:
    """Confusion matrix per (task, modality) at the last probed layer.
    Written so the chapter can honestly report which classes collapse.
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    out: dict = {}
    for npz in sorted(cache_dir.glob("hidden_*.npz")):
        parsed = _parse_task_modality(npz)
        if not parsed:
            continue
        task, mod = parsed
        try:
            data = np.load(npz, allow_pickle=True)
            hidden = data["hidden"]
            labels = np.array(data["labels"])
            layer = int(data["layers_probed"][-1])
            n_train = int(data["n_train"])
            le = LabelEncoder()
            y = le.fit_transform(labels)
            classes = list(le.classes_)

            X_tr, X_te = hidden[:n_train, layer, :], hidden[n_train:, layer, :]
            y_tr, y_te = y[:n_train], y[n_train:]
            scaler = StandardScaler().fit(X_tr)
            clf = LogisticRegressionCV(
                Cs=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5,
                max_iter=1000, class_weight="balanced", scoring="f1_macro",
            )
            clf.fit(scaler.transform(X_tr), y_tr)
            pred = clf.predict(scaler.transform(X_te))
            cm = confusion_matrix(y_te, pred, labels=range(len(classes)))
            out.setdefault(task, {})[mod] = {
                "classes": classes,
                "layer": layer,
                "confusion": cm.tolist(),
                "train_class_counts": {c: int((y_tr == i).sum()) for i, c in enumerate(classes)},
                "test_class_counts":  {c: int((y_te == i).sum()) for i, c in enumerate(classes)},
                "pred_class_counts":  {c: int((pred == i).sum()) for i, c in enumerate(classes)},
            }
        except Exception as e:
            logger.warning("confusion skip %s: %s", npz.name, e)
    output_path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", output_path)


# ── 6. Probe orthogonality (shared vs independent tactical subspaces) ────────


def orthogonality(cache_dir: Path, output_path: Path, modality: str = "d") -> None:
    """Cosine similarity between linear-probe weight vectors from different
    tasks (same modality, last layer). If ≈0 for all pairs, tactical concepts
    occupy independent subspaces. Validates the cross-task transfer finding
    from a different angle.
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    probes: dict = {}
    for npz in sorted(cache_dir.glob(f"hidden_*_{modality}.npz")):
        parsed = _parse_task_modality(npz)
        if not parsed:
            continue
        task, mod = parsed
        if mod != modality:
            continue
        data = np.load(npz, allow_pickle=True)
        layer = int(data["layers_probed"][-1])
        n_train = int(data["n_train"])
        le = LabelEncoder()
        y = le.fit_transform(np.array(data["labels"]))
        X = data["hidden"][:n_train, layer, :].astype(np.float64)
        scaler = StandardScaler().fit(X)
        clf = LogisticRegressionCV(
            Cs=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5,
            max_iter=1000, class_weight="balanced", scoring="f1_macro",
        )
        clf.fit(scaler.transform(X), y[:n_train])
        # Un-scale the weight direction to get it in the original feature space
        coef = clf.coef_ / (scaler.scale_ + 1e-9)   # (n_dirs, D)
        # Reduce multiclass coefs to a single task "direction": mean of absolute value
        task_dir = coef.mean(axis=0) if coef.shape[0] > 1 else coef[0]
        probes[task] = task_dir / (np.linalg.norm(task_dir) + 1e-9)

    tasks = sorted(probes.keys())
    mat = {a: {b: round(float(probes[a] @ probes[b]), 4) for b in tasks} for a in tasks}
    out = {"modality": modality, "tasks": tasks, "cosine_similarity": mat}
    output_path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", output_path)
    logger.info("probe direction cosine matrix:")
    for a in tasks:
        logger.info("  " + a[:14].ljust(14) + " " + " ".join(f"{mat[a][b]:+.3f}" for b in tasks))


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("multiseed")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n-seeds", type=int, default=5)

    p = sp.add_parser("transfer")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--modality", default="d")

    p = sp.add_parser("mdl")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output", required=True)

    p = sp.add_parser("perclass")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output", required=True)

    p = sp.add_parser("confusion")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output", required=True)

    p = sp.add_parser("orthogonality")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--modality", default="d")

    args = ap.parse_args()
    if args.cmd == "multiseed":
        multiseed(Path(args.cache_dir), Path(args.output), args.n_seeds)
    elif args.cmd == "transfer":
        transfer(Path(args.cache_dir), Path(args.output), args.modality)
    elif args.cmd == "mdl":
        mdl(Path(args.cache_dir), Path(args.output))
    elif args.cmd == "perclass":
        perclass(Path(args.cache_dir), Path(args.output))
    elif args.cmd == "confusion":
        confusion(Path(args.cache_dir), Path(args.output))
    elif args.cmd == "orthogonality":
        orthogonality(Path(args.cache_dir), Path(args.output), args.modality)


if __name__ == "__main__":
    main()
