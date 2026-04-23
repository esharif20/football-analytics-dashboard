"""VLM A/B comparison: text-only vs text+vision grounding.

Three conditions:
  1. text_only      — structured markdown grounding (current system)
  2. text_raw       — markdown + unannotated video keyframes
  3. text_annotated — markdown + frames with tracking overlays (bboxes, team colours)

Tests the assessor's suggestion that "augmenting video data with annotations will be vital".

Usage:
    python -m backend.evaluation.vlm_comparison \\
        --analytics path/to/analytics.json \\
        --tracks path/to/tracks.json \\
        --video path/to/video.mp4 \\
        --output eval_output/vlm/
"""

import argparse
import asyncio
import base64
import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from ._common import (
    EvalConfig,
    ensure_output_dir,
    load_analytics,
    load_tracks,
    save_figure,
    save_latex_table,
)
from .llm_grounding import (
    EXTRACTION_PROMPT,
    compute_grounding_score,
    extract_claims,
    format_as_markdown,
    verify_claim,
)


# ── Keyframe extraction ────────────────────────────────────────────────────────


def extract_keyframes(video_path: str, n_frames: int = 5) -> list[np.ndarray]:
    """Extract uniformly-spaced keyframes from video."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def annotate_frame(
    frame: np.ndarray,
    tracks: list[dict],
    frame_idx: int,
) -> np.ndarray:
    """Draw bounding boxes and team colours on a frame."""
    import cv2
    annotated = frame.copy()

    # Find closest frame in tracks
    closest = min(tracks, key=lambda f: abs(f.get("frameNumber", 0) - frame_idx))
    players = closest.get("playerPositions", {})
    # playerPositions is a dict {trackId: {x, y, teamId}} from the pipeline
    if isinstance(players, dict):
        players_iter = [(tid, pdata) for tid, pdata in players.items()]
    else:
        players_iter = [(p.get("trackId", p.get("id", "?")), p) for p in players]

    team_colors_bgr = {
        0: (255, 100, 100),   # Blue-ish for team 0
        1: (100, 255, 100),   # Green-ish for team 1
        -1: (200, 200, 200),  # Grey for unknown
    }

    for track_id, p in players_iter:
        cx = int(p.get("x", p.get("pixelX", 0)))
        cy = int(p.get("y", p.get("pixelY", 0)))
        team_id = p.get("teamId", -1)
        color = team_colors_bgr.get(team_id, (200, 200, 200))

        # Draw circle at player centre + team label
        cv2.circle(annotated, (cx, cy), 15, color, 2)
        cv2.putText(
            annotated, f"#{track_id} T{team_id}",
            (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
        )

    ball = closest.get("ballPosition")
    if ball:
        pixel_pos = ball.get("pixelPos", [ball.get("pixelX", 0), ball.get("pixelY", 0)])
        bx = int(pixel_pos[0]) if isinstance(pixel_pos, (list, tuple)) else int(ball.get("pixelX", 0))
        by = int(pixel_pos[1]) if isinstance(pixel_pos, (list, tuple)) else int(ball.get("pixelY", 0))
        cv2.circle(annotated, (bx, by), 10, (0, 255, 255), -1)
        cv2.putText(annotated, "BALL", (bx + 12, by), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return annotated


def frames_to_jpeg_bytes(frames: list[np.ndarray]) -> list[bytes]:
    """Convert numpy frames to JPEG bytes for Gemini vision API."""
    import cv2
    result = []
    for frame in frames:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            result.append(bytes(buf))
    return result


# ── Gemini Vision Provider ────────────────────────────────────────────────────


class GeminiVisionProvider:
    """Gemini 2.0 Flash with image support for A/B testing."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash"):
        import os
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model
        self._client = None

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={"temperature": 0.7, "top_p": 0.9, "max_output_tokens": 4096},
            )
        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[bytes] | None = None,
    ) -> str:
        import asyncio
        import google.generativeai as genai

        client = self._get_client()
        content_parts = [f"{system_prompt}\n\n---\n\n{user_prompt}"]

        if images:
            import base64
            for img_bytes in images:
                content_parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(img_bytes).decode("utf-8"),
                    }
                })
            content_parts.append(
                "\nThe images above show keyframes from the match. "
                "Use them alongside the data above to provide a more complete analysis."
            )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: client.generate_content(content_parts)
        )
        return response.text


# ── OpenAI Vision Provider ────────────────────────────────────────────────────


class OpenAIVisionProvider:
    """OpenAI GPT-4o-mini with image support for A/B testing."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        import os
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model_name = model
        self._client = None

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[bytes] | None = None,
    ) -> str:
        client = self._get_client()
        content_parts: list[dict] = [{"type": "text", "text": user_prompt}]

        if images:
            for img_bytes in images:
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })

        response = await client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_parts},
            ],
            temperature=0.7,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""


# ── Provider Registry ─────────────────────────────────────────────────────────

VISION_PROVIDERS = {
    "gemini": GeminiVisionProvider,
    "openai": OpenAIVisionProvider,
}


def classify_spatial_claims(results: list) -> dict:
    """Classify verification results into spatial vs non-spatial claims.

    Spatial claims reference positions, zones, or locations (prone to VLM hallucination
    per CAPTURE benchmark -- 14.75% error rate on GPT-4o).

    Returns:
        {
            "spatial": {verified, refuted, unverifiable, total, grounding_rate},
            "non_spatial": {verified, refuted, unverifiable, total, grounding_rate},
        }
    """
    spatial_keywords = {
        "zone", "half", "third", "left", "right", "wing", "flank", "area",
        "position", "region", "territory", "pitch", "space", "box", "penalty",
    }

    def _is_spatial(text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in spatial_keywords)

    spatial_res, non_spatial_res = [], []
    for r in results:
        claim_text = r.get("text", "") if isinstance(r, dict) else getattr(r.claim, "text", "")
        if _is_spatial(claim_text):
            spatial_res.append(r)
        else:
            non_spatial_res.append(r)

    def _counts(rs):
        counts = {"verified": 0, "refuted": 0, "unverifiable": 0, "plausible": 0}
        for r in rs:
            v = r.get("verdict", "") if isinstance(r, dict) else r.verdict
            counts[v] = counts.get(v, 0) + 1
        denom = counts["verified"] + counts["refuted"] + counts["unverifiable"]
        counts["total"] = len(rs)
        counts["grounding_rate"] = counts["verified"] / denom if denom else 0.0
        return counts

    return {"spatial": _counts(spatial_res), "non_spatial": _counts(non_spatial_res)}


# ── Comparison runner ─────────────────────────────────────────────────────────


async def run_condition(
    condition: str,
    analytics: dict,
    markdown_context: str,
    image_bytes: list[bytes] | None,
    vision_provider: GeminiVisionProvider,
    judge_provider,
    analysis_type: str = "match_overview",
) -> dict:
    """Run a single VLM condition and return scored results."""
    from services.tactical import SYSTEM_PROMPTS
    system_prompt = SYSTEM_PROMPTS[analysis_type]

    commentary = await vision_provider.generate(
        system_prompt, markdown_context, images=image_bytes
    )
    claims = await extract_claims(commentary, judge_provider)
    results = [verify_claim(c, analytics) for c in claims]
    score = compute_grounding_score(results, commentary=commentary, analytics=analytics)
    spatial_breakdown = classify_spatial_claims(results)
    score["spatial_breakdown"] = spatial_breakdown

    return {
        "condition": condition,
        "commentary": commentary,
        "n_claims": score["total_claims"],
        "grounding_rate": score["grounding_rate"],
        "hallucination_rate": score["hallucination_rate"],
        "verified": score.get("verified", 0),
        "refuted": score.get("refuted", 0),
        "score": score,
        "spatial_breakdown": spatial_breakdown,
    }


async def run_async(
    analytics_path: str,
    tracks_path: str,
    video_path: str,
    output_dir: str,
    n_keyframes: int = 5,
    analysis_type: str = "match_overview",
    provider: str = "gemini",
) -> dict:
    from services.llm_providers import get_provider

    out = ensure_output_dir(output_dir)
    analytics = load_analytics(analytics_path)
    tracks = load_tracks(tracks_path)
    markdown_context = format_as_markdown(analytics)

    providers_to_run = list(VISION_PROVIDERS.keys()) if provider == "all" else [provider]

    print(f"\n=== VLM Grounding Comparison ({analysis_type}) ===")
    print(f"Extracting {n_keyframes} keyframes from video...")

    raw_frames = extract_keyframes(video_path, n_keyframes)
    print(f"  Extracted {len(raw_frames)} frames")

    total_frames = max((f.get("frameNumber", 0) for f in tracks), default=0)
    keyframe_indices = np.linspace(0, total_frames, n_keyframes, dtype=int)
    annotated_frames = [
        annotate_frame(frame, tracks, int(fi))
        for frame, fi in zip(raw_frames, keyframe_indices)
    ]

    raw_jpeg = frames_to_jpeg_bytes(raw_frames)
    annotated_jpeg = frames_to_jpeg_bytes(annotated_frames)

    conditions = [
        ("text_only", None),
        ("text_raw_frames", raw_jpeg),
        ("text_annotated_frames", annotated_jpeg),
    ]

    all_results: dict[str, dict] = {}

    for provider_name in providers_to_run:
        vision_provider = VISION_PROVIDERS[provider_name]()
        if not vision_provider.is_available():
            print(f"  Skipping {provider_name} (no API key)")
            continue

        judge_provider = get_provider(provider_name)
        print(f"\n--- Provider: {provider_name} ---")

        results: dict[str, dict] = {}
        for condition, images in conditions:
            print(f"  Running condition: {condition} ...", end=" ", flush=True)
            result = await run_condition(
                condition, analytics, markdown_context, images,
                vision_provider, judge_provider, analysis_type
            )
            results[condition] = result
            all_results[f"{provider_name}_{condition}"] = result
            print(
                f"grounding={result['grounding_rate']:.1%}  "
                f"claims={result['n_claims']}  "
                f"refuted={result['refuted']}"
            )

        # Per-provider artifacts
        (Path(str(out)) / f"vlm_results_{provider_name}.json").write_text(
            json.dumps(results, indent=2)
        )

        table_rows = [
            [
                cond,
                results[cond]["n_claims"],
                results[cond]["verified"],
                results[cond]["refuted"],
                f"{results[cond]['grounding_rate']:.1%}",
                f"{results[cond]['hallucination_rate']:.1%}",
            ]
            for cond in results
        ]
        save_latex_table(
            headers=["Condition", "Claims", "Verified", "Refuted", "Grounding Rate", "Hallucination Rate"],
            rows=table_rows,
            caption=f"VLM grounding comparison ({provider_name}): text-only vs text+vision ({analysis_type})",
            name=f"vlm_comparison_{provider_name}",
            output_dir=str(out),
            label=f"tab:vlm_comparison_{provider_name}",
        )

        cond_labels = list(results.keys())
        rates = [results[c]["grounding_rate"] * 100 for c in cond_labels]
        colors = ["#4f86c6", "#e07b54", "#6aab6a"]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(cond_labels, rates, color=colors[:len(cond_labels)], edgecolor="white", linewidth=0.8)
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Grounding rate (%)")
        ax.set_title(f"VLM Grounding Comparison — {provider_name}\n({analysis_type})")
        ax.set_xticklabels(cond_labels, rotation=12, ha="right")
        fig.tight_layout()
        save_figure(fig, f"vlm_grounding_comparison_{provider_name}", str(out))

    # Cross-provider comparison table
    if len(providers_to_run) > 1:
        cmp_rows = []
        for cond, _ in conditions:
            row = [cond]
            for pname in providers_to_run:
                key = f"{pname}_{cond}"
                gr = all_results[key]["grounding_rate"] if key in all_results else float("nan")
                row.append(f"{gr:.1%}" if not (gr != gr) else "n/a")
            cmp_rows.append(row)
        save_latex_table(
            headers=["Condition"] + providers_to_run,
            rows=cmp_rows,
            caption=f"Cross-provider VLM grounding comparison ({analysis_type})",
            name="vlm_provider_comparison",
            output_dir=str(out),
            label="tab:vlm_provider_comparison",
        )

    # Save sample annotated frames
    frames_dir = Path(str(out)) / "sample_frames"
    frames_dir.mkdir(exist_ok=True)
    import cv2
    for i, (raw, ann) in enumerate(zip(raw_frames, annotated_frames)):
        cv2.imwrite(str(frames_dir / f"raw_{i:02d}.jpg"), raw)
        cv2.imwrite(str(frames_dir / f"annotated_{i:02d}.jpg"), ann)

    print(f"\nOutputs saved to: {out}/")
    return all_results


def run(
    analytics_path: str,
    tracks_path: str,
    video_path: str,
    output_dir: str,
    n_keyframes: int = 5,
    provider: str = "gemini",
) -> dict:
    return asyncio.run(
        run_async(analytics_path, tracks_path, video_path, output_dir, n_keyframes, provider=provider)
    )


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=".env", override=True)
    except ImportError:
        pass
    parser = argparse.ArgumentParser(description="VLM text-only vs text+vision grounding comparison")
    parser.add_argument("--analytics", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--n-keyframes", type=int, default=5)
    parser.add_argument("--analysis-type", default="match_overview",
                        choices=["match_overview", "tactical_deep_dive", "event_analysis", "player_spotlight"])
    parser.add_argument("--provider", default="gemini", choices=["gemini", "openai", "claude", "groq", "all"])
    parser.add_argument("--output", default="eval_output/vlm")
    args = parser.parse_args()

    asyncio.run(run_async(
        args.analytics, args.tracks, args.video,
        args.output, args.n_keyframes, args.analysis_type, args.provider,
    ))


if __name__ == "__main__":
    main()
