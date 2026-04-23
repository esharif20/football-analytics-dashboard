"""Class-token-rank analysis across 4 tasks × 4 modalities × 8 layers.

For each (task, modality, layer):
    1. Fit a linear probe at that layer (using cached hiddens).
    2. Project probe coef_ through lm_head.T to vocab space.
    3. For each class-name token (e.g. "high", "press"), compute its RANK in the
       vocab score distribution for the class's direction.

Rank = how close is the class-name token to the top of the vocab space when
reading out that class from the hidden state? Lower is better.

Output: {task: {modality: {layer: {class_name: {token: rank}}}}}

Usage (on pod):
    python -m backend.evaluation.class_token_ranks \
        --cache-dir /workspace/probe/hidden_cache \
        --model-path "$MODEL_PATH" \
        --output results/class_token_rank_clean.json \
        --modalities d v d+v nec+v
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

TASKS_CLASSES = {
    "pressing_type":          ["high_press", "mid_block", "low_block"],
    "compactness_trend":      ["compact", "moderate", "expansive"],
    "possession_phase":       ["sustained", "transitional", "chaotic"],
    "territorial_dominance":  ["pressing_high", "balanced", "retreating"],
}


def _mod_to_filename(mod: str) -> str:
    return mod.replace("+", "_")


def run(cache_dir: Path, model_path: str, output_path: Path,
        modalities: list[str], layers: list[int]) -> None:
    import torch
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    logger.info("loading Qwen2-VL-7B from %s", model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, device_map="cpu", torch_dtype=torch.float32,
    )
    model.eval()

    # Extract lm_head weight (V, D)
    lm_head = None
    for name in ("lm_head", "language_model.lm_head"):
        obj = model
        for part in name.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "weight"):
            lm_head = obj.weight.detach().float().cpu().numpy()
            break
    if lm_head is None:
        raise RuntimeError("could not find lm_head")
    logger.info("lm_head shape %s", lm_head.shape)
    V = lm_head.shape[0]
    del model  # free memory — only need the weight matrix

    out: dict[str, Any] = {}
    for task, class_names in TASKS_CLASSES.items():
        out[task] = {}
        # Pre-tokenise each class's name parts (e.g. "high_press" → ["high", "press"])
        class_token_ids: dict[str, dict[str, int]] = {}
        for cls in class_names:
            token_map: dict[str, int] = {}
            parts = cls.split("_")
            for part in parts:
                ids = tokenizer.encode(" " + part, add_special_tokens=False)
                if ids:
                    token_map[part] = ids[0]
            class_token_ids[cls] = token_map

        for mod in modalities:
            fname = _mod_to_filename(mod)
            npz_path = cache_dir / f"hidden_{task}_{fname}.npz"
            if not npz_path.exists():
                logger.warning("MISSING %s", npz_path)
                continue
            data = np.load(npz_path, allow_pickle=True)
            hidden = data["hidden"]         # (N, L, D)
            labels = np.array(data["labels"])
            n_train = int(data["n_train"])
            probed_layers = [int(x) for x in data["layers_probed"]]

            layer_map: dict[str, Any] = {}
            for layer in layers:
                if layer not in probed_layers:
                    logger.warning("layer %d not in cache for %s/%s", layer, task, mod)
                    continue
                L_idx = probed_layers.index(layer)
                x_tr = hidden[:n_train, L_idx, :].astype(np.float64)
                y_str = labels[:n_train]
                le = LabelEncoder().fit(list(dict.fromkeys(y_str)))
                y_tr = le.transform(y_str)
                scaler = StandardScaler().fit(x_tr)
                clf = LogisticRegressionCV(
                    Cs=[0.001, 0.01, 0.1, 1.0, 10.0], cv=3,
                    max_iter=1000, class_weight="balanced",
                    scoring="f1_macro",
                )
                clf.fit(scaler.transform(x_tr), y_tr)
                W_probe = clf.coef_                       # (C, D) or (1, D) for binary
                scale = scaler.scale_ + 1e-9
                class_dirs = W_probe / scale              # (C, D)
                vocab_scores = class_dirs @ lm_head.T     # (C, V)

                class_ranks: dict[str, dict[str, int]] = {}
                actual_classes = list(le.classes_)
                for ci, cls in enumerate(actual_classes):
                    if ci >= vocab_scores.shape[0]:
                        break
                    vs = vocab_scores[ci]
                    # Rank = position when sorted DESC (1 = highest)
                    order = np.argsort(-vs)
                    rank_map = {int(order[r]): r + 1 for r in range(len(order))}
                    token_ranks: dict[str, int] = {}
                    for part, tok_id in class_token_ids.get(cls, {}).items():
                        token_ranks[part] = int(rank_map.get(tok_id, V))
                    class_ranks[cls] = token_ranks
                layer_map[str(layer)] = class_ranks

            out[task][mod] = layer_map
            output_path.write_text(json.dumps(out, indent=2))
            logger.info("done %s/%s (layers %s)", task, mod,
                        list(layer_map.keys()))

    output_path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--modalities", nargs="+",
                    default=["d", "v", "d+v", "nec+v"])
    ap.add_argument("--layers", nargs="+", type=int,
                    default=[0, 4, 8, 12, 16, 20, 24, 28])
    args = ap.parse_args()
    run(args.cache_dir, args.model_path, args.output,
        args.modalities, args.layers)
