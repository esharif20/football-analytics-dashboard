"""Schumacher-style prompt stability study for Qwen2-VL-7B-Instruct.

Produces per (task, modality) cell:
  - Orange columns: variant F1 distribution under 10 meaning-preserving prompt
    variants, greedy decoding (prompt-wording sensitivity).
  - Blue columns: sampling-based Pass@K via Kulal et al.'s unbiased estimator
    (20 draws at T=0.7, top_k=0, top_p=1.0) — sampling-output consistency.
  - Green column: logit-based class-argmax accuracy at the first decoded
    position, measuring the upper bound on constrained-decoding F1.

Modalities:
  d     — digit-space text (Gruver et al. 2023 tokenisation), no image.
  v     — vision-only: chart image + minimal text referencing it, no numeric
          series in the text.
  d+v   — fusion: digit-space text + chart image.

Fixes relative to earlier runs:
  B1. Vision modalities now actually pass pixel_values via
      ``processor(text=..., images=...)`` with Qwen2VL chat template.
  B2. ``v`` prompt no longer embeds numeric series in text — chart-only.
  B3. Qwen2VL chat template with image placeholder used for v / d+v.
  F2. Green column caption clarifies the comparator to P@1 (sampling).
  H1. ``samples_per_variant`` default 30 (matches Schumacher).
  H2. ``passk_samples`` default 30 (matches Schumacher's full-test-set intent).
  H3. ``top_p=1.0`` (Schumacher specifies only T; no nucleus truncation).
  H4. ``torch.manual_seed(0)`` set once at load — reproducible across runs.
  Plus the earlier ``top_k=0`` override of Qwen's ``top_k=1`` default so
  sampling actually diversifies.

Usage:
  python -m backend.evaluation.prompt_stability_schumacher \\
      --model-path <HF snapshot> --ground-truth <json> \\
      --output prompt_stability_final.json \\
      --modalities d v d+v \\
      --variants 10 --samples-per-variant 30 --passk-samples 30
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _pass_at_k_unbiased(n: int, c: int, K: int) -> float:
    """Schumacher A.4 Pass@K: 1 - C(n-c, K) / C(n, K).
    Conventions: Pass@K = 0 if c = 0; Pass@K = 1 if n - c < K."""
    if c == 0:
        return 0.0
    if n - c < K:
        return 1.0
    return 1.0 - math.comb(n - c, K) / math.comb(n, K)


def run(
    model_path: str, model_type: str, ground_truth_path: str, output_path: str,
    modalities: list[str], variants: int = 10, samples_per_variant: int = 30,
    passk_n: int = 20, passk_samples: int = 30, load_in_8bit: bool = True,
    seed: int = 0,
) -> None:
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/workspace")
    os.environ.setdefault("PIPELINE_SUBPROCESS", "1")
    import torch
    from PIL import Image
    from backend.evaluation.linear_probing import (
        CLASSIFICATION_TASKS, prepare_classification_data,
        _build_classification_prompt, _render_chart_for_modality,
        _format_series_string,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    assert model_type == "qwen2vl", "only Qwen2VL supported (for vision)"
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    kwargs = {"device_map": "auto"}
    if load_in_8bit:
        kwargs["load_in_8bit"] = True
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    model.eval()
    device = next(model.parameters()).device

    gt = json.loads(Path(ground_truth_path).read_text())
    task_data = prepare_classification_data([gt])

    def _parse_pred(text: str, classes: list[str]) -> str:
        low = text.lower()
        for cls in sorted(classes, key=len, reverse=True):
            if cls.lower() in low:
                return cls
        for cls in classes:
            for w in cls.split("_"):
                if w.lower() in low:
                    return cls
        return ""

    def _class_first_token_ids(classes: list[str]) -> dict[str, int]:
        ids = {}
        for cls in classes:
            toks = tokenizer.encode(" " + cls, add_special_tokens=False)
            ids[cls] = toks[0] if toks else tokenizer.eos_token_id
        return ids

    def _render_image_for_series(series: list[float], task_name: str,
                                  fps: float = 25.0):
        """Render a single-line chart for the series. Returns PIL.Image or None."""
        img_bytes = _render_chart_for_modality(series, task_name, "v", fps)
        if not img_bytes:
            return None
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Build prompts per modality. Honest semantics:
    #   d           -> digit-space numeric text, no image
    #   v           -> chart-only: no numeric series in text, image carries
    #   d+v         -> digit-space text + chart image
    #   pt          -> patch-text only (no image)
    #   sf          -> stat-features text only (no image)
    #   nec+v, any <text>+v -> <text> via _format_series_string + chart image
    def _build_multimodal_inputs(series: list[float], task_name: str,
                                  modality: str, variant_id: int, fps: float = 25.0):
        task = CLASSIFICATION_TASKS[task_name]
        classes = task["classes"]
        class_labels = "\n".join(
            f"  [{i+1}] {cls} — {task['labels'][cls]}"
            for i, cls in enumerate(classes)
        )

        if modality == "v":
            # Schumacher-faithful v: chart attached, prompt wording varies per
            # variant_id (role / data-label / output-instruction) WITHOUT the
            # numeric series in text — the chart is the sole signal channel.
            from backend.evaluation.linear_probing import _PROMPT_VARIANTS
            vid = variant_id % len(_PROMPT_VARIANTS)
            role, _data_label, output_instr = _PROMPT_VARIANTS[vid]
            class_list = ", ".join(classes)
            output_instr = output_instr.replace("{cls0}", classes[0]).replace(
                "{class_list}", class_list
            )
            if vid == 0:
                body_text = (
                    f"You are a football tactics analyst. Classify the following "
                    f"{len(series)}-frame time series at {fps:.0f} fps based on "
                    f"the attached chart.\n\n"
                    f"Task: {task['description']}\n\n"
                    f"Choose exactly one class:\n{class_labels}\n\n"
                    f"Respond with only the class name (e.g. '{classes[0]}')."
                )
            else:
                body_text = (
                    f"{role} Classify the following {len(series)}-frame time series "
                    f"at {fps:.0f} fps based on the attached chart.\n\n"
                    f"Task: {task['description']}\n\n"
                    f"Choose exactly one class:\n{class_labels}\n\n"
                    f"{output_instr}"
                )
            pil_img = _render_image_for_series(series, task_name, fps)
        elif "+" in modality:
            # Composite <text>+<visual>: build text via _build_classification_prompt
            # using the text part (e.g. "nec" from "nec+v"), then attach the chart.
            text_mod = modality.split("+", 1)[0]
            body_text = _build_classification_prompt(
                series, task_name, text_mod, fps, variant_id=variant_id
            )
            pil_img = _render_image_for_series(series, task_name, fps)
        else:
            # Text-only modality (d, pt, sf, nec, p, pi, n, b, ne, nc):
            # defer text construction to the shared prompt builder.
            body_text = _build_classification_prompt(
                series, task_name, modality, fps, variant_id=variant_id
            )
            pil_img = None

        # Build Qwen2VL chat-template input
        content: list = []
        if pil_img is not None:
            content.append({"type": "image", "image": pil_img})
        content.append({"type": "text", "text": body_text})
        messages = [{"role": "user", "content": content}]
        formatted = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if pil_img is not None:
            inputs = processor(
                text=[formatted], images=[pil_img],
                return_tensors="pt", padding=True,
            ).to(device)
        else:
            inputs = processor(
                text=[formatted], return_tensors="pt", padding=True,
            ).to(device)
        return inputs

    def _generate_inputs(inputs, sample: bool, temp: float = 0.7,
                          max_new: int = 24) -> str:
        gen_kwargs = dict(
            max_new_tokens=max_new,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=sample,
        )
        if sample:
            # Schumacher A.4: temperature sampling. top_k=0 overrides Qwen's
            # default top_k=1 that would otherwise force greedy; top_p=1.0
            # avoids nucleus truncation not specified by Schumacher.
            gen_kwargs.update(temperature=temp, top_p=1.0, top_k=0)
        else:
            gen_kwargs.update(temperature=1.0, top_p=1.0, top_k=1)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        input_len = inputs["input_ids"].shape[1]
        text = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        return text.strip()

    def _class_argmax_correct(inputs, true_class: str,
                               class_first_ids: dict[str, int]) -> bool:
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        class_logits = {c: float(logits[tid].item())
                        for c, tid in class_first_ids.items()}
        return max(class_logits, key=class_logits.get) == true_class

    from sklearn.metrics import f1_score
    from sklearn.preprocessing import LabelEncoder

    out: dict[str, Any] = {}
    for task_name in CLASSIFICATION_TASKS:
        if task_name not in task_data:
            continue
        series_list, labels = task_data[task_name]
        classes = CLASSIFICATION_TASKS[task_name]["classes"]
        n_take = min(samples_per_variant, len(series_list))
        k_take = min(passk_samples, len(series_list))
        out[task_name] = {}

        for mod in modalities:
            logger.info("=== %s / %s ===", task_name, mod)
            cell: dict[str, Any] = {"variants": {}}

            # Orange axis — variant F1 spread (greedy).
            for v in range(variants):
                preds: list[str] = []
                for s in series_list[:n_take]:
                    inputs = _build_multimodal_inputs(
                        s, task_name, mod, variant_id=v
                    )
                    text = _generate_inputs(inputs, sample=False)
                    preds.append(_parse_pred(text, classes))
                true = list(labels[:n_take])
                seen = sorted(set(true) | {p for p in preds if p})
                if not seen:
                    seen = classes
                le = LabelEncoder().fit(seen)
                y_true = le.transform(true)
                y_pred = le.transform([p if p in seen else seen[0] for p in preds])
                f1 = float(f1_score(y_true, y_pred, average="macro",
                                     zero_division=0))
                cell["variants"][str(v)] = round(f1, 4)
                logger.info("  variant %d -> F1 %.3f", v, f1)

            vals = list(cell["variants"].values())
            cell.update(
                variant_min=round(min(vals), 4),
                variant_max=round(max(vals), 4),
                variant_mean=round(float(np.mean(vals)), 4),
                variant_median=round(float(np.median(vals)), 4),
                variant_delta=round(max(vals) - min(vals), 4),
                n_samples_per_variant=n_take,
            )

            # Blue axis — sampling-based Pass@K (Schumacher A.4).
            logger.info("  Pass@K sampling: n=%d draws x %d windows (T=0.7)",
                        passk_n, k_take)
            correct_per_window: list[int] = []
            for s, true_label in list(zip(series_list, labels))[:k_take]:
                inputs = _build_multimodal_inputs(
                    s, task_name, mod, variant_id=0
                )
                c = 0
                for _ in range(passk_n):
                    text = _generate_inputs(inputs, sample=True, temp=0.7)
                    if _parse_pred(text, classes) == true_label:
                        c += 1
                correct_per_window.append(c)

            p1 = float(np.mean([_pass_at_k_unbiased(passk_n, c, 1)
                                 for c in correct_per_window]))
            pk = float(np.mean([_pass_at_k_unbiased(passk_n, c, passk_n)
                                 for c in correct_per_window]))
            cell.update(
                p_at_1_sampling=round(p1, 4),
                p_at_20_sampling=round(pk, 4),
                delta_p_at_k_sampling=round(pk - p1, 4),
                passk_n=passk_n,
                n_passk_windows=len(correct_per_window),
                passk_method="sampling_unbiased_kulal",
            )
            logger.info("  [sampling] P@1=%.3f P@%d=%.3f dPK=%.3f",
                        p1, passk_n, pk, pk - p1)

            # Green axis — class-argmax over C class tokens at first decode pos.
            logger.info("  Logit class-argmax accuracy (C=%d)", len(classes))
            class_first_ids = _class_first_token_ids(classes)
            argmax_hits: list[int] = []
            for s, true_label in list(zip(series_list, labels))[:k_take]:
                inputs = _build_multimodal_inputs(
                    s, task_name, mod, variant_id=0
                )
                argmax_hits.append(
                    1 if _class_argmax_correct(inputs, true_label,
                                                class_first_ids) else 0
                )
            acc = float(np.mean(argmax_hits))
            cell.update(
                class_argmax_correct_rate=round(acc, 4),
                class_argmax_n_windows=len(argmax_hits),
            )
            logger.info("  [logit] class-argmax acc=%.3f", acc)

            out[task_name][mod] = cell
            Path(output_path).write_text(json.dumps(out, indent=2))

    Path(output_path).write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-type", default="qwen2vl")
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--modalities", nargs="+", default=["d", "v", "d+v"])
    ap.add_argument("--variants", type=int, default=10)
    ap.add_argument("--samples-per-variant", type=int, default=30)
    ap.add_argument("--passk-n", type=int, default=20)
    ap.add_argument("--passk-samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.model_path, args.model_type, args.ground_truth, args.output,
        args.modalities, args.variants, args.samples_per_variant,
        args.passk_n, args.passk_samples, seed=args.seed)
