"""Interpretability probes that run on the pod against a loaded LLM.

Three sub-commands reuse the same cached hidden states and the model's
unembedding matrix:

  logit-lens       — project hidden states through lm_head at every probed
                     layer to see which vocabulary tokens the model is
                     "thinking" of at that depth.
  probe-vocab      — train a linear probe per (task, modality, layer) on
                     cached hidden states, project its weight direction
                     through lm_head.T, and report the top-k vocabulary
                     tokens that align with each class axis.
  activation-steer — extract the linear probe direction per class, inject
                     it additively into the model's hidden state at a
                     chosen layer during zero-shot prompting, and measure
                     the change in prompting F1 (causal test of whether the
                     direction encodes the concept the interface can read).

All three gracefully skip individual (task, modality) failures so the
master run continues even if one cache file is malformed or the model
lm_head cannot be located.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load_model_and_tokenizer(model_path: str, model_type: str,
                              load_in_8bit: bool = True):
    import torch
    if model_type == "qwen2vl":
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
        if load_in_8bit:
            kwargs["load_in_8bit"] = True
            kwargs.pop("torch_dtype")
        processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
        return model, processor, processor.tokenizer
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
        if load_in_8bit:
            kwargs["load_in_8bit"] = True
            kwargs.pop("torch_dtype")
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        return model, None, tokenizer


def _get_lm_head(model):
    """Return the unembedding weight matrix (vocab_size, hidden_dim).
    For 8-bit quantised models the weight is an Int8Params object that
    cannot be used directly in a matmul; caller should fall back to
    `model.lm_head(x)` forward-pass in that case.
    """
    for attr_path in [
        ["lm_head", "weight"],
        ["model", "lm_head", "weight"],
        ["language_model", "lm_head", "weight"],
    ]:
        obj = model
        ok = True
        for a in attr_path:
            if hasattr(obj, a):
                obj = getattr(obj, a)
            else:
                ok = False
                break
        if ok:
            return obj
    raise AttributeError("Could not locate lm_head weight on model")


def _get_lm_head_module(model):
    """Return the callable lm_head module so we can call `module(x)` forward.
    This works regardless of quantisation."""
    for attr_path in [["lm_head"], ["model", "lm_head"], ["language_model", "lm_head"]]:
        obj = model
        ok = True
        for a in attr_path:
            if hasattr(obj, a):
                obj = getattr(obj, a)
            else:
                ok = False; break
        if ok and callable(obj):
            return obj
    raise AttributeError("Could not locate lm_head module on model")


def _parse_task_modality(npz_path: Path, known_mods: list[str]) -> tuple[str, str] | None:
    stem = npz_path.stem.replace("hidden_", "")
    for mod in sorted(known_mods, key=len, reverse=True):
        safe = mod.replace("+", "_")
        if stem.endswith("_" + safe):
            return stem[: -(len(safe) + 1)], mod
    return None


KNOWN_MODS = ["ne_va", "ne_sp", "ne_v", "p_va", "p_sp", "p_v2", "p_vh",
              "n_va", "n_sp", "n_v",
              "d_v", "pi_v", "p_v", "b_v",
              "pi", "ne", "va", "sp", "v2", "vh",
              "d", "v", "p", "m", "o", "b", "n"]


# ── 1. Logit Lens ────────────────────────────────────────────────────────────


def logit_lens(
    cache_dir: Path,
    model_path: str,
    model_type: str,
    output_path: Path,
    top_k: int = 10,
    target_layers: list[int] | None = None,
) -> None:
    """Project cached hidden states through lm_head at each probed layer.

    For each (task, modality, layer, class) emits the top-k vocab tokens
    most strongly activated by the class-mean hidden state. Reveals where
    the concept becomes linguistically word-accessible.
    """
    import torch

    model, _, tokenizer = _load_model_and_tokenizer(model_path, model_type)
    model.eval()
    lm_head = _get_lm_head_module(model)
    device = next(model.parameters()).device

    out: dict[str, Any] = {}
    for npz in sorted(Path(cache_dir).glob("hidden_*.npz")):
        parsed = _parse_task_modality(npz, KNOWN_MODS)
        if not parsed:
            continue
        task, mod = parsed
        try:
            data = np.load(npz, allow_pickle=True)
            hidden = data["hidden"]
            labels = np.array(data["labels"])
            probed = [int(x) for x in data["layers_probed"]]
            layers = target_layers or probed
            layers = [L for L in layers if L < hidden.shape[1]]
            classes = list(dict.fromkeys([str(l) for l in labels]))

            task_out = out.setdefault(task, {}).setdefault(mod, {})
            for L in layers:
                layer_out: dict[str, list] = {}
                for cls in classes:
                    mask = np.array([str(l) == cls for l in labels])
                    if mask.sum() == 0:
                        continue
                    mean_h_np = hidden[mask, L, :].mean(axis=0)
                    with torch.no_grad():
                        h = torch.tensor(mean_h_np, dtype=torch.float16, device=device)
                        logits_t = lm_head(h.unsqueeze(0))[0].float().cpu()
                    logits = logits_t.numpy()
                    top_idx = np.argsort(-logits)[:top_k]
                    layer_out[cls] = [
                        {"token": tokenizer.decode([int(i)]).strip(),
                         "logit": float(logits[i])}
                        for i in top_idx
                    ]
                task_out[str(L)] = layer_out
            logger.info("  %s/%s done (%d layers, %d classes)",
                        task, mod, len(layers), len(classes))
        except Exception as e:
            logger.warning("  %s: %s", npz.name, e)

    output_path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", output_path)


# ── 2. Probe-weight → vocabulary projection ──────────────────────────────────


def probe_vocab(
    cache_dir: Path,
    model_path: str,
    model_type: str,
    output_path: Path,
    top_k: int = 15,
    target_layer: int | None = None,
) -> None:
    """Train a linear probe per (task, modality, layer), project coef_ through
    lm_head.T to find the vocabulary tokens aligned with each class direction.
    """
    import torch
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    model, _, tokenizer = _load_model_and_tokenizer(model_path, model_type)
    model.eval()
    W = _get_lm_head(model)
    # Move to fp32 on CPU for the projection arithmetic
    W_np = W.detach().float().cpu().numpy()  # (V, D)

    out: dict[str, Any] = {}
    for npz in sorted(Path(cache_dir).glob("hidden_*.npz")):
        parsed = _parse_task_modality(npz, KNOWN_MODS)
        if not parsed:
            continue
        task, mod = parsed
        try:
            data = np.load(npz, allow_pickle=True)
            hidden = data["hidden"]
            labels = np.array(data["labels"])
            n_train = int(data["n_train"])
            probed = [int(x) for x in data["layers_probed"]]
            L = target_layer if (target_layer is not None and target_layer in probed) else probed[-1]

            x_tr = hidden[:n_train, L, :].astype(np.float64)
            y_tr = LabelEncoder().fit_transform(labels[:n_train])
            classes = list(dict.fromkeys(labels[:n_train]))

            scaler = StandardScaler().fit(x_tr)
            clf = LogisticRegressionCV(
                Cs=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5,
                max_iter=1000, class_weight="balanced", scoring="f1_macro",
            )
            clf.fit(scaler.transform(x_tr), y_tr)
            W_probe = clf.coef_  # (C or 1, D)

            # Un-scale: because we scaled x with mean/std, class-direction in
            # the original feature space is coef / scale (up to an intercept).
            scale = scaler.scale_ + 1e-9
            class_dirs = W_probe / scale  # (C, D)
            # Project to vocab
            vocab_scores = class_dirs @ W_np.T  # (C, V)

            task_out = out.setdefault(task, {}).setdefault(mod, {})
            task_out["layer"] = L
            task_out["classes"] = classes
            task_out["top_tokens_per_class"] = {}
            # Handle both binary and multiclass
            if vocab_scores.shape[0] == 1 and len(classes) == 2:
                # Binary: positive direction = second class
                vs = vocab_scores[0]
                pos = np.argsort(-vs)[:top_k]
                neg = np.argsort(vs)[:top_k]
                task_out["top_tokens_per_class"][classes[1]] = [
                    {"token": tokenizer.decode([int(i)]).strip(),
                     "score": float(vs[i])} for i in pos
                ]
                task_out["top_tokens_per_class"][classes[0]] = [
                    {"token": tokenizer.decode([int(i)]).strip(),
                     "score": float(vs[i])} for i in neg
                ]
            else:
                for ci, cls in enumerate(classes):
                    if ci >= vocab_scores.shape[0]:
                        break
                    vs = vocab_scores[ci]
                    top = np.argsort(-vs)[:top_k]
                    task_out["top_tokens_per_class"][cls] = [
                        {"token": tokenizer.decode([int(i)]).strip(),
                         "score": float(vs[i])} for i in top
                    ]
            logger.info("  %s/%s layer %d done", task, mod, L)
        except Exception as e:
            logger.warning("  %s: %s", npz.name, e)

    output_path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", output_path)


# ── 3. Activation steering (causal test) ─────────────────────────────────────


def activation_steer(
    cache_dir: Path,
    model_path: str,
    model_type: str,
    ground_truth_path: Path,
    output_path: Path,
    layer: int = 20,
    alpha: float = 4.0,
    task_filter: str | None = None,
) -> None:
    """Causal probing. For each task:
      1. Train a probe on cached hidden states → get class direction w_c.
      2. For each test sample, generate the prompting response WITH and
         WITHOUT injecting w_c at the target layer's output (forward hook
         adds α·w_c to the last-token hidden state).
      3. Report the change in prompting macro-F1.

    If steering raises prompting F1 meaningfully (e.g. +0.2), the direction
    CAUSALLY encodes the concept AND the interface bottleneck is bypassable.
    """
    import torch
    import asyncio
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    # Import the prompting baseline from the main probing module.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from backend.evaluation.linear_probing import (
        CLASSIFICATION_TASKS, prepare_classification_data,
        run_prompting_baseline, _build_classification_prompt,
    )

    model, processor, tokenizer = _load_model_and_tokenizer(model_path, model_type)
    model.eval()
    device = next(model.parameters()).device

    # Locate the layer to hook.
    def _get_layer(m, L):
        # Qwen2-VL has model.model.layers; causal LM has model.model.layers too
        inner = m.model if hasattr(m, "model") else m
        if hasattr(inner, "layers"):
            return inner.layers[L]
        raise AttributeError("cannot find .layers on model")

    target_module = _get_layer(model, layer)

    # Data
    gt = json.loads(Path(ground_truth_path).read_text())
    task_data = prepare_classification_data([gt])

    out: dict[str, Any] = {}

    for task_name in CLASSIFICATION_TASKS:
        if task_filter and task_filter != task_name:
            continue
        if task_name not in task_data:
            continue
        series_list, labels = task_data[task_name]
        # Find the corresponding d-modality cache for the probe direction
        cache_path = Path(cache_dir) / f"hidden_{task_name}_d.npz"
        if not cache_path.exists():
            logger.warning("no d-cache for %s, skip", task_name)
            continue

        data = np.load(cache_path, allow_pickle=True)
        hidden = data["hidden"]
        lbl = np.array(data["labels"])
        n_train = int(data["n_train"])

        # Train probe at the target layer to get the class direction(s).
        x_tr = hidden[:n_train, layer, :].astype(np.float64)
        y_tr = LabelEncoder().fit_transform(lbl[:n_train])
        classes_ordered = list(dict.fromkeys(lbl[:n_train]))
        scaler = StandardScaler().fit(x_tr)
        clf = LogisticRegressionCV(
            Cs=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5,
            max_iter=1000, class_weight="balanced", scoring="f1_macro",
        )
        clf.fit(scaler.transform(x_tr), y_tr)
        W_probe = clf.coef_ / (scaler.scale_ + 1e-9)

        # Run un-steered prompting baseline first as reference.
        try:
            base = asyncio.run(run_prompting_baseline(
                model, tokenizer, series_list, labels, task_name, "d",
            ))
            base_f1 = base.get("f1_macro", float("nan"))
        except Exception as e:
            logger.warning("  baseline prompting failed %s: %s", task_name, e)
            base_f1 = float("nan")

        # Steering hook — add α·direction to last-token hidden state.
        steered_f1s = {}
        for ci, cls in enumerate(classes_ordered):
            # Binary case: single direction, positive = class[1]
            if W_probe.shape[0] == 1:
                direction = W_probe[0] if cls == classes_ordered[1] else -W_probe[0]
            else:
                if ci >= W_probe.shape[0]:
                    continue
                direction = W_probe[ci]

            dir_t = torch.tensor(direction / (np.linalg.norm(direction) + 1e-9),
                                 dtype=torch.float16, device=device)

            def _hook(module, inputs, output, _dir=dir_t, _a=alpha):
                # output is typically a tuple (hidden_states, ...) for transformer blocks
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                # Add steering to last-token position across batch
                h[..., -1, :] = h[..., -1, :] + _a * _dir
                if isinstance(output, tuple):
                    return (h,) + output[1:]
                return h

            handle = target_module.register_forward_hook(_hook)
            try:
                steered = asyncio.run(run_prompting_baseline(
                    model, tokenizer, series_list, labels, task_name, "d",
                ))
                steered_f1s[cls] = steered.get("f1_macro", float("nan"))
            except Exception as e:
                logger.warning("  steered-%s prompting failed: %s", cls, e)
                steered_f1s[cls] = float("nan")
            finally:
                handle.remove()

        out[task_name] = {
            "baseline_f1": base_f1,
            "steered_per_class_f1": steered_f1s,
            "layer": layer,
            "alpha": alpha,
            "classes": classes_ordered,
        }
        logger.info("%s: base=%.3f steered=%s", task_name, base_f1, steered_f1s)

    output_path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", output_path)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    p_lens = sp.add_parser("logit-lens")
    p_lens.add_argument("--cache-dir", required=True)
    p_lens.add_argument("--model-path", required=True)
    p_lens.add_argument("--model-type", default="qwen2vl")
    p_lens.add_argument("--output", required=True)
    p_lens.add_argument("--top-k", type=int, default=10)
    p_lens.add_argument("--layers", nargs="+", type=int, default=None)

    p_vocab = sp.add_parser("probe-vocab")
    p_vocab.add_argument("--cache-dir", required=True)
    p_vocab.add_argument("--model-path", required=True)
    p_vocab.add_argument("--model-type", default="qwen2vl")
    p_vocab.add_argument("--output", required=True)
    p_vocab.add_argument("--top-k", type=int, default=15)
    p_vocab.add_argument("--layer", type=int, default=None)

    p_steer = sp.add_parser("activation-steer")
    p_steer.add_argument("--cache-dir", required=True)
    p_steer.add_argument("--model-path", required=True)
    p_steer.add_argument("--model-type", default="qwen2vl")
    p_steer.add_argument("--ground-truth", required=True)
    p_steer.add_argument("--output", required=True)
    p_steer.add_argument("--layer", type=int, default=20)
    p_steer.add_argument("--alpha", type=float, default=4.0)
    p_steer.add_argument("--task", default=None,
                         help="If set, only steer this task")

    args = ap.parse_args()
    if args.cmd == "logit-lens":
        logit_lens(Path(args.cache_dir), args.model_path, args.model_type,
                   Path(args.output), top_k=args.top_k,
                   target_layers=args.layers)
    elif args.cmd == "probe-vocab":
        probe_vocab(Path(args.cache_dir), args.model_path, args.model_type,
                    Path(args.output), top_k=args.top_k,
                    target_layer=args.layer)
    elif args.cmd == "activation-steer":
        activation_steer(
            Path(args.cache_dir), args.model_path, args.model_type,
            Path(args.ground_truth), Path(args.output),
            layer=args.layer, alpha=args.alpha, task_filter=args.task,
        )


if __name__ == "__main__":
    main()
