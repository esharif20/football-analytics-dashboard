"""Vision-text cosine similarity grounding using SigLIP.

Computes cosine similarity between video keyframes and extracted claim text
using the SigLIP vision-language model (already cached in this project via
the team_assigner). This provides a visual grounding score separate from the
analytics-based claim verification in llm_grounding.py.

Whereas llm_grounding.py verifies: "does this claim match the analytics JSON?"
This module asks: "does this claim match what is visually visible in the video?"

SigLIP vs CLIP:
  - SigLIP uses sigmoid contrastive loss (better per-pair calibration than CLIP softmax)
  - Its vision encoder is already loaded for team classification — weights are cached
  - We load the full SiglipModel (vision + text) instead of SiglipVisionModel (vision only)

Usage:
    python -m backend.evaluation.vision_grounding \\
        --artifacts eval_output/unified/grounding/artifacts/ \\
        --video path/to/annotated_video.mp4 \\
        --output eval_output/vision_grounding/

    # Or as part of unified runner:
    python -m backend.evaluation.unified_runner \\
        --analytics eval_output/10_analytics.json \\
        --video path/to/video.mp4 \\
        --only vision_grounding \\
        --output eval_output/unified/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from ._common import ensure_output_dir, save_figure, save_latex_table

SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"
_DEFAULT_THRESHOLD = 0.20   # SigLIP similarity scores are lower than CLIP; 0.2 is a reasonable plausibility threshold
_MAX_CLAIM_LEN = 77         # SigLIP text tokenizer max tokens (same as CLIP)


# ── Device resolution (mirrors TeamClassifier._resolve_device) ────────────────


def _resolve_device(device: str | None = None) -> str:
    if device:
        return device
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# ── Model loader ──────────────────────────────────────────────────────────────


class VisionTextGrounder:
    """Compute cosine similarity between video keyframes and text claims.

    Uses the full SiglipModel (vision + text encoders). The vision weights
    are already cached locally from the TeamClassifier.
    """

    def __init__(self, device: str | None = None):
        self.device = _resolve_device(device)
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import SiglipModel, SiglipProcessor
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for vision grounding. "
                "Install with: pip install transformers torch"
            ) from e

        print(f"  Loading SigLIP model ({SIGLIP_MODEL_PATH}) on {self.device}...")
        self._processor = SiglipProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self._model = SiglipModel.from_pretrained(SIGLIP_MODEL_PATH).to(self.device)
        self._model.eval()
        self._torch = torch
        print("  SigLIP loaded.")

    def _encode_images(self, frames: list[np.ndarray]) -> np.ndarray:
        """Encode numpy BGR frames → L2-normalised embeddings (n_frames, d)."""
        import cv2
        self._load()
        # Convert BGR→RGB for SigLIP
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        inputs = self._processor(images=rgb_frames, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self._torch.no_grad():
            vision_out = self._model.get_image_features(**inputs)

        embs = vision_out.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.maximum(norms, 1e-8)

    def _encode_texts(self, claims: list[str]) -> np.ndarray:
        """Encode text claims → L2-normalised embeddings (n_claims, d)."""
        self._load()
        # Truncate long claims to max token length
        truncated = [c[:400] for c in claims]  # rough char limit before tokenisation
        inputs = self._processor(text=truncated, return_tensors="pt", padding=True,
                                 truncation=True, max_length=_MAX_CLAIM_LEN)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self._torch.no_grad():
            text_out = self._model.get_text_features(**inputs)

        embs = text_out.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.maximum(norms, 1e-8)

    def compute_similarities(
        self,
        frames: list[np.ndarray],
        claims: list[str],
    ) -> np.ndarray:
        """Return (n_claims × n_frames) cosine similarity matrix."""
        if not frames or not claims:
            return np.zeros((len(claims), len(frames)))

        frame_embs = self._encode_images(frames)   # (n_frames, d)
        claim_embs = self._encode_texts(claims)     # (n_claims, d)
        return claim_embs @ frame_embs.T            # (n_claims, n_frames)

    def ground_claims(
        self,
        frames: list[np.ndarray],
        claims: list[str],
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> list[dict]:
        """For each claim return its best-matching frame and similarity score.

        Returns list of:
            {
                "claim": str,
                "best_frame_idx": int,
                "best_similarity": float,
                "mean_similarity": float,
                "visually_grounded": bool,   # best_similarity >= threshold
            }
        """
        if not frames or not claims:
            return [{"claim": c, "best_frame_idx": -1, "best_similarity": 0.0,
                     "mean_similarity": 0.0, "visually_grounded": False} for c in claims]

        sim_matrix = self.compute_similarities(frames, claims)   # (n_claims, n_frames)

        results = []
        for i, claim in enumerate(claims):
            row = sim_matrix[i]
            best_idx = int(np.argmax(row))
            results.append({
                "claim": claim,
                "best_frame_idx": best_idx,
                "best_similarity": float(row[best_idx]),
                "mean_similarity": float(np.mean(row)),
                "visually_grounded": bool(row[best_idx] >= threshold),
            })
        return results


# ── Claim loader ──────────────────────────────────────────────────────────────


def load_claims_from_artifacts(artifacts_dir: str, provider: str = "openai") -> list[dict]:
    """Load claim texts from grounding artifact JSON files.

    Returns list of {claim_text, analysis_type, verdict}.
    """
    art_dir = Path(artifacts_dir)
    if not art_dir.exists():
        return []

    claims: list[dict] = []
    for fpath in sorted(art_dir.glob(f"{provider}_markdown_*.json")):
        try:
            data = json.loads(fpath.read_text())
        except Exception:
            continue
        atype = data.get("analysis_type", fpath.stem)
        for vr in data.get("verification_results", []):
            text = (vr.get("text") or "").strip()
            if text:
                claims.append({
                    "claim_text": text,
                    "analysis_type": atype,
                    "verdict": vr.get("verdict", "unverifiable"),
                    "source_sentence": vr.get("source_sentence", ""),
                })
    return claims


# ── Visualisation ─────────────────────────────────────────────────────────────


def _plot_heatmap(
    sim_matrix: np.ndarray,
    claim_labels: list[str],
    output_dir: str,
    threshold: float = _DEFAULT_THRESHOLD,
) -> None:
    """Plot claims × frames similarity heatmap."""
    n_claims, n_frames = sim_matrix.shape
    fig, ax = plt.subplots(figsize=(max(6, n_frames * 1.2), max(4, n_claims * 0.4)))

    im = ax.imshow(sim_matrix, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=0.4)
    plt.colorbar(im, ax=ax, label="Cosine similarity")

    ax.set_xticks(range(n_frames))
    ax.set_xticklabels([f"Frame {i}" for i in range(n_frames)], fontsize=8)
    ax.set_yticks(range(n_claims))
    ax.set_yticklabels([c[:55] + "..." if len(c) > 55 else c for c in claim_labels],
                       fontsize=7)

    # Mark cells above threshold
    for i in range(n_claims):
        for j in range(n_frames):
            if sim_matrix[i, j] >= threshold:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor="green", linewidth=1.5))

    ax.set_title(f"Vision-Text Cosine Similarity (SigLIP)\nGreen border = visually grounded (≥{threshold:.2f})")
    fig.tight_layout()
    save_figure(fig, "vision_text_heatmap", output_dir)


def _plot_grounding_bar(results: list[dict], output_dir: str) -> None:
    """Bar chart of per-claim visual grounding scores."""
    if not results:
        return
    labels = [r["claim"][:40] + "..." if len(r["claim"]) > 40 else r["claim"] for r in results]
    scores = [r["best_similarity"] for r in results]
    colors = ["#4CAF50" if r["visually_grounded"] else "#ef5350" for r in results]

    fig, ax = plt.subplots(figsize=(8, max(4, len(results) * 0.35)))
    bars = ax.barh(range(len(labels)), scores, color=colors)
    ax.axvline(x=_DEFAULT_THRESHOLD, color="gray", linestyle="--", linewidth=1, label=f"Threshold ({_DEFAULT_THRESHOLD})")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Best frame cosine similarity")
    ax.set_title("Per-Claim Visual Grounding (SigLIP)\nGreen = visually supported, Red = not supported")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, "vision_claim_scores", output_dir)


# ── Main runner ───────────────────────────────────────────────────────────────


def run(
    artifacts_dir: str,
    video_path: str,
    output_dir: str,
    provider: str = "openai",
    n_keyframes: int = 5,
    threshold: float = _DEFAULT_THRESHOLD,
    device: str | None = None,
) -> dict:
    """Run vision-text grounding and return summary dict."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    out = ensure_output_dir(output_dir)

    print(f"\n=== Vision-Text Cosine Similarity Grounding (SigLIP) ===")
    print(f"Artifacts:  {artifacts_dir}")
    print(f"Video:      {video_path}")
    print(f"Threshold:  {threshold}")

    # Load claims
    claims_data = load_claims_from_artifacts(artifacts_dir, provider)
    if not claims_data:
        print("  No claims found in artifacts dir. Run llm_grounding first.")
        return {"_status": "no_claims"}

    claim_texts = [c["claim_text"] for c in claims_data]
    print(f"  Loaded {len(claim_texts)} claims from {artifacts_dir}")

    # Extract keyframes
    print(f"  Extracting {n_keyframes} keyframes from video...")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
    try:
        from services.vision import extract_keyframes
        frames = extract_keyframes(video_path, n_keyframes)
    except Exception as e:
        print(f"  Error extracting frames: {e}")
        return {"_status": "frame_error", "_error": str(e)}

    if not frames:
        print("  No frames extracted from video.")
        return {"_status": "no_frames"}

    print(f"  Extracted {len(frames)} frames.")

    # Compute similarities
    grounder = VisionTextGrounder(device=device)
    try:
        results = grounder.ground_claims(frames, claim_texts, threshold=threshold)
    except ImportError as e:
        print(f"  {e}")
        return {"_status": "import_error", "_error": str(e)}

    # Attach metadata from claims_data
    for res, meta in zip(results, claims_data):
        res["analysis_type"] = meta["analysis_type"]
        res["analytics_verdict"] = meta["verdict"]

    # Summary stats
    n_grounded = sum(1 for r in results if r["visually_grounded"])
    visual_grounding_rate = n_grounded / len(results) if results else 0.0

    print(f"\n  Visual grounding rate: {visual_grounding_rate:.1%} ({n_grounded}/{len(results)} claims)")
    print(f"  (Threshold: {threshold})")

    # By analysis type
    by_type: dict[str, dict] = {}
    for r in results:
        atype = r.get("analysis_type", "unknown")
        by_type.setdefault(atype, {"grounded": 0, "total": 0, "scores": []})
        by_type[atype]["total"] += 1
        by_type[atype]["scores"].append(r["best_similarity"])
        if r["visually_grounded"]:
            by_type[atype]["grounded"] += 1

    for atype, stats in by_type.items():
        rate = stats["grounded"] / stats["total"] if stats["total"] else 0.0
        avg = np.mean(stats["scores"]) if stats["scores"] else 0.0
        print(f"    {atype:<25} visual grounding={rate:.1%}  avg_sim={avg:.3f}")

    # Compare analytics verdict vs visual grounding
    agreement = sum(
        1 for r in results
        if (r["analytics_verdict"] == "verified") == r["visually_grounded"]
    )
    agreement_rate = agreement / len(results) if results else 0.0

    # Save results JSON
    summary = {
        "_status": "ok",
        "n_claims": len(results),
        "n_frames": len(frames),
        "threshold": threshold,
        "visual_grounding_rate": round(visual_grounding_rate, 4),
        "analytics_visual_agreement_rate": round(agreement_rate, 4),
        "by_analysis_type": {
            k: {
                "grounded": v["grounded"],
                "total": v["total"],
                "visual_grounding_rate": round(v["grounded"] / v["total"], 4) if v["total"] else 0.0,
                "avg_similarity": round(float(np.mean(v["scores"])), 4) if v["scores"] else 0.0,
            }
            for k, v in by_type.items()
        },
        "per_claim": results,
    }
    (out / "vision_grounding_results.json").write_text(json.dumps(summary, indent=2))

    # Visualisations
    sim_matrix = np.array([[r["best_similarity"]] for r in results])  # simplified: best score only
    _plot_grounding_bar(results, str(out))

    # Full similarity matrix heatmap (recompute to get all frames × claims)
    try:
        frame_embs = grounder._encode_images(frames)
        claim_embs = grounder._encode_texts(claim_texts)
        full_matrix = claim_embs @ frame_embs.T
        _plot_heatmap(full_matrix, claim_texts, str(out), threshold)
    except Exception:
        pass  # heatmap is best-effort

    # LaTeX table
    type_rows = [
        [atype.replace("_", " ").title(),
         stats["total"],
         stats["grounded"],
         f"{stats['visual_grounding_rate']:.1%}",
         f"{stats['avg_similarity']:.3f}"]
        for atype, stats in summary["by_analysis_type"].items()
    ]
    save_latex_table(
        headers=["Analysis Type", "Claims", "Visually Grounded", "Rate", "Avg Sim"],
        rows=type_rows,
        caption=f"Vision-text cosine similarity grounding (SigLIP, threshold={threshold})",
        name="vision_grounding_by_type",
        output_dir=str(out),
        label="tab:vision_grounding",
    )

    print(f"\nOutputs saved to: {out}/")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vision-text cosine similarity grounding using SigLIP"
    )
    parser.add_argument("--artifacts", required=True,
                        help="Directory of grounding artifact JSONs (from llm_grounding.py)")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output", default="eval_output/vision_grounding")
    parser.add_argument("--provider", default="openai", choices=["openai", "gemini"])
    parser.add_argument("--n-keyframes", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=_DEFAULT_THRESHOLD)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    run(
        artifacts_dir=args.artifacts,
        video_path=args.video,
        output_dir=args.output,
        provider=args.provider,
        n_keyframes=args.n_keyframes,
        threshold=args.threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
