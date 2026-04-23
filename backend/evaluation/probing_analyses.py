"""Downstream analyses on cached probing hidden states.

Operates purely on the `.npz` files written by `run_layer_wise_analysis`
when `cache_dir` is set — no GPU required. Produces an `analyses.json`
that feeds `probing_figures.py`.

Analyses:
  • cross_layer_cka   — Linear CKA matrix (Kornblith et al. 2019)
  • silhouette_curve  — how cleanly classes cluster at each layer
  • intrinsic_dim     — PCA-95% variance dims per layer
  • anisotropy        — mean cosine similarity of random pairs per layer
  • selectivity       — label-shuffle control probe (Hewitt & Liang 2019)
  • modality_synergy  — d+v minus max(d, v) per layer (fusion depth)
  • extraction_gap    — layer-wise linear F1 minus prompting F1
  • prototype_dist    — inter-class centroid distances per layer

All heavy lifting is NumPy / scikit-learn on CPU.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Linear CKA ────────────────────────────────────────────────────────────────


def _center(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def linear_cka_gram(kx: np.ndarray, ky: np.ndarray) -> float:
    """Linear CKA from centered N×N Gram matrices.

    Equivalent to Kornblith et al. 2019 linear CKA but computed via the
    HSIC-style Gram-matrix form:  tr(K_X K_Y) / sqrt(tr(K_X K_X) tr(K_Y K_Y)).
    For N ≪ D (our case: N=120, D=3584) this is ~100× faster than the
    feature-space form because the heavy matmul is N×N instead of D×D.
    """
    num = float(np.sum(kx * ky))
    den = float(np.sqrt(np.sum(kx * kx) * np.sum(ky * ky)))
    return num / den if den > 0 else 0.0


def cka_matrix(hidden: np.ndarray, layer_indices: list[int] | None = None) -> tuple[list[int], np.ndarray]:
    """Return (layers, cka[L, L]) for the sampled layers.

    hidden: (n_samples, n_layers, hidden_dim). If layer_indices given, subsets.
    Pre-computes one centered Gram matrix per layer (N×N), then every pairwise
    CKA reduces to elementwise products over those tiny matrices.
    """
    n_layers = hidden.shape[1]
    layers = list(range(n_layers)) if layer_indices is None else list(layer_indices)
    n = hidden.shape[0]
    # Precompute centered N×N Gram matrices (one per layer).
    H = np.eye(n) - np.ones((n, n)) / n  # centering matrix
    grams: list[np.ndarray] = []
    for L in layers:
        x = hidden[:, L, :].astype(np.float64)
        K = x @ x.T          # (N, N)
        Kc = H @ K @ H       # centred
        grams.append(Kc)
    mat = np.zeros((len(layers), len(layers)), dtype=np.float32)
    for i in range(len(layers)):
        for j in range(i, len(layers)):
            v = linear_cka_gram(grams[i], grams[j])
            mat[i, j] = v
            mat[j, i] = v
    return layers, mat


# ── Separability / geometry ──────────────────────────────────────────────────


def silhouette_per_layer(
    hidden: np.ndarray,
    labels: np.ndarray,
    layer_indices: list[int],
) -> dict[int, float]:
    """Silhouette score per layer using (centered) raw hidden states."""
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(labels)
    out: dict[int, float] = {}
    for L in layer_indices:
        x = hidden[:, L, :]
        if len(np.unique(y)) < 2 or x.shape[0] < 3:
            out[int(L)] = float("nan")
            continue
        try:
            out[int(L)] = float(silhouette_score(x, y, metric="cosine"))
        except Exception as e:
            logger.warning("silhouette layer %d failed: %s", L, e)
            out[int(L)] = float("nan")
    return out


def intrinsic_dim_per_layer(
    hidden: np.ndarray,
    layer_indices: list[int],
    var_threshold: float = 0.95,
) -> dict[int, int]:
    """PCA 95%-variance effective dimensionality per layer."""
    from sklearn.decomposition import PCA
    out: dict[int, int] = {}
    for L in layer_indices:
        x = hidden[:, L, :].astype(np.float64)
        n = min(x.shape)
        try:
            p = PCA(n_components=min(n - 1, 256)).fit(x)
            cum = np.cumsum(p.explained_variance_ratio_)
            k = int(np.searchsorted(cum, var_threshold) + 1)
        except Exception as e:
            logger.warning("intrinsic_dim layer %d failed: %s", L, e)
            k = -1
        out[int(L)] = k
    return out


def anisotropy_per_layer(
    hidden: np.ndarray,
    layer_indices: list[int],
    n_pairs: int = 1000,
    seed: int = 42,
) -> dict[int, float]:
    """Mean cosine similarity of random pairs per layer (Ethayarajh 2019).
    High = representations collapse onto a cone ⇒ expressivity cost.
    """
    rng = np.random.default_rng(seed)
    n = hidden.shape[0]
    if n < 2:
        return {int(L): float("nan") for L in layer_indices}
    a = rng.integers(0, n, size=n_pairs)
    b = rng.integers(0, n, size=n_pairs)
    mask = a != b
    a, b = a[mask], b[mask]
    out: dict[int, float] = {}
    for L in layer_indices:
        x = hidden[:, L, :]
        norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
        xn = x / norm
        cos = np.einsum("ij,ij->i", xn[a], xn[b])
        out[int(L)] = float(cos.mean())
    return out


def prototype_distance_per_layer(
    hidden: np.ndarray,
    labels: np.ndarray,
    layer_indices: list[int],
) -> dict[int, float]:
    """Mean pairwise Euclidean distance between class centroids per layer,
    normalised by mean within-class distance (Fisher-like ratio)."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = np.unique(y)
    out: dict[int, float] = {}
    for L in layer_indices:
        x = hidden[:, L, :]
        centroids = np.stack([x[y == c].mean(axis=0) for c in classes])
        inter = 0.0; k = 0
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                inter += np.linalg.norm(centroids[i] - centroids[j])
                k += 1
        inter = inter / max(k, 1)
        intra = float(np.mean([
            np.linalg.norm(x[y == c] - centroids[ci], axis=1).mean()
            for ci, c in enumerate(classes)
            if (y == c).sum() > 0
        ]))
        out[int(L)] = float(inter / (intra + 1e-9))
    return out


# ── Selectivity (control task) ───────────────────────────────────────────────


def _train_linear_probe_simple(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(
        max_iter=1000, C=1.0, class_weight="balanced", multi_class="auto"
    )
    clf.fit(scaler.transform(X_train), y_train)
    pred = clf.predict(scaler.transform(X_test))
    return float(f1_score(y_test, pred, average="macro", zero_division=0))


def selectivity_per_layer(
    hidden: np.ndarray,
    labels: np.ndarray,
    layer_indices: list[int],
    n_train: int,
    n_shuffles: int = 5,
    seed: int = 42,
) -> dict[int, dict[str, float]]:
    """Hewitt & Liang 2019: a good probe should have low F1 on a random-label
    control. selectivity = F1(real) - mean F1(shuffled). High = probe is
    actually reading the representation, not memorising through capacity.
    """
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(labels)
    rng = np.random.default_rng(seed)

    out: dict[int, dict[str, float]] = {}
    for L in layer_indices:
        x = hidden[:, L, :]
        x_tr, x_te = x[:n_train], x[n_train:]
        y_tr, y_te = y[:n_train], y[n_train:]
        try:
            f1_real = _train_linear_probe_simple(x_tr, y_tr, x_te, y_te)
        except Exception as e:
            logger.warning("selectivity real probe layer %d failed: %s", L, e)
            out[int(L)] = {"f1_real": float("nan"), "f1_shuffle_mean": float("nan"),
                           "selectivity": float("nan")}
            continue
        shuffle_f1s: list[float] = []
        for _ in range(n_shuffles):
            y_tr_shuf = y_tr.copy(); rng.shuffle(y_tr_shuf)
            try:
                shuffle_f1s.append(
                    _train_linear_probe_simple(x_tr, y_tr_shuf, x_te, y_te)
                )
            except Exception:
                pass
        mean_shuf = float(np.mean(shuffle_f1s)) if shuffle_f1s else float("nan")
        out[int(L)] = {
            "f1_real": round(f1_real, 4),
            "f1_shuffle_mean": round(mean_shuf, 4),
            "selectivity": round(f1_real - mean_shuf, 4),
        }
    return out


# ── Narrative metrics from results JSON ──────────────────────────────────────


def modality_synergy(results: dict, task: str) -> dict[str, dict[int, float]]:
    """Per (task, layer): d+v − max(d, v). Positive ⇒ fusion helps."""
    lw_d   = results[task].get("d",   {}).get("layer_wise", {})
    lw_v   = results[task].get("v",   {}).get("layer_wise", {})
    lw_dv  = results[task].get("d+v", {}).get("layer_wise", {})

    def _f1(cell: Any) -> float:
        if isinstance(cell, dict):
            return cell.get("linear_f1", float("nan"))
        return float(cell) if cell is not None else float("nan")

    layers = sorted({int(k) for k in lw_d} | {int(k) for k in lw_v} | {int(k) for k in lw_dv})
    out = {}
    for L in layers:
        fd  = _f1(lw_d.get(str(L), lw_d.get(L)))
        fv  = _f1(lw_v.get(str(L), lw_v.get(L)))
        fdv = _f1(lw_dv.get(str(L), lw_dv.get(L)))
        synergy = fdv - max(fd, fv) if all(np.isfinite([fd, fv, fdv])) else float("nan")
        out[int(L)] = {
            "linear_d":   fd,
            "linear_v":   fv,
            "linear_d+v": fdv,
            "synergy":    synergy,
        }
    return {"per_layer": out}


def extraction_gap(results: dict, task: str, modality: str) -> dict[int, float]:
    """Layer-wise linear F1 minus prompting F1 (zero-shot generation).
    Positive ⇒ the model knows but can't say it (a.k.a. extraction gap).
    """
    lw = results[task].get(modality, {}).get("layer_wise", {})
    prompting_f1 = (
        results[task].get(modality, {}).get("prompting", {}).get("f1_macro")
    )
    if prompting_f1 is None or not isinstance(prompting_f1, (int, float)):
        return {}
    out: dict[int, float] = {}
    for k, cell in lw.items():
        try:
            L = int(k)
        except Exception:
            continue
        if isinstance(cell, dict):
            f1 = cell.get("linear_f1")
        else:
            f1 = cell
        if f1 is None:
            continue
        out[L] = float(f1) - float(prompting_f1)
    return out


# ── Orchestrator ─────────────────────────────────────────────────────────────


def compute_all(cache_dir: Path, results_path: Path, out_path: Path) -> dict:
    """Run every analysis over every cached (task, modality) file."""
    cache_dir = Path(cache_dir)
    results = json.loads(Path(results_path).read_text())
    analyses: dict[str, Any] = {"per_task_modality": {}, "per_task": {}}

    for npz in sorted(cache_dir.glob("hidden_*.npz")):
        name = npz.stem.replace("hidden_", "")
        # task name may contain underscores; match against known modality
        # suffixes, longest first so 'd_v' is preferred over 'v' when both
        # tails match the filename.
        # Longest-first to disambiguate composite suffixes like 'ne_va' vs 'va'.
        for mod_stem in ("nec_va", "nec_sp", "nec_v", "nc_va", "nc_sp", "nc_v",
                         "ne_va", "ne_sp", "ne_v", "p_va", "p_sp", "p_v2",
                         "p_vh", "n_va", "n_sp", "n_v",
                         "pt_v", "sf_v",
                         "d_v", "pi_v", "p_v", "b_v",
                         "vfix", "nec", "nc", "pt", "sf",
                         "pi", "ne", "va", "sp", "v2", "vh",
                         "d", "v", "p", "m", "o", "b", "n"):
            if name.endswith("_" + mod_stem):
                task = name[: -(len(mod_stem) + 1)]
                # Composite modalities use '+' in canonical naming (e.g. 'ne+v');
                # cache filenames use '_' separators. Translate for all composites.
                mod = mod_stem.replace("_", "+") if "_" in mod_stem else mod_stem
                break
        else:
            logger.warning("skip %s (cannot parse modality)", npz)
            continue

        data = np.load(npz, allow_pickle=True)
        hidden = data["hidden"]           # (N, L, D)
        labels = data["labels"]
        layers_probed = [int(x) for x in data["layers_probed"]]
        n_train = int(data["n_train"])
        logger.info("analysing %s / %s : hidden=%s, layers=%s",
                    task, mod, hidden.shape, layers_probed)

        all_layer_ixs = list(range(hidden.shape[1]))
        _, cka = cka_matrix(hidden, all_layer_ixs)
        sep_layers = layers_probed  # lightweight analyses only on probed layers
        sil = silhouette_per_layer(hidden, labels, sep_layers)
        idim = intrinsic_dim_per_layer(hidden, sep_layers)
        aniso = anisotropy_per_layer(hidden, sep_layers)
        proto = prototype_distance_per_layer(hidden, labels, sep_layers)
        sel = selectivity_per_layer(hidden, labels, sep_layers, n_train=n_train)
        gap = extraction_gap(results, task, mod)

        analyses["per_task_modality"].setdefault(task, {})[mod] = {
            "cka_layers": all_layer_ixs,
            "cka": cka.tolist(),
            "silhouette":      {str(k): v for k, v in sil.items()},
            "intrinsic_dim":   {str(k): v for k, v in idim.items()},
            "anisotropy":      {str(k): v for k, v in aniso.items()},
            "prototype_dist":  {str(k): v for k, v in proto.items()},
            "selectivity":     {str(k): v for k, v in sel.items()},
            "extraction_gap":  {str(k): v for k, v in gap.items()},
        }

    for task in results:
        if not isinstance(results.get(task), dict):
            continue
        try:
            analyses["per_task"][task] = {"modality_synergy": modality_synergy(results, task)}
        except Exception as e:
            logger.warning("synergy failed for %s: %s", task, e)

    out_path.write_text(json.dumps(analyses, indent=2))
    logger.info("wrote %s", out_path)
    return analyses


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--output", required=True,
                    help="Path to analyses.json output")
    args = ap.parse_args()
    compute_all(Path(args.cache_dir), Path(args.results), Path(args.output))


if __name__ == "__main__":
    main()
