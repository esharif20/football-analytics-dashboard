# Phase 14: Multi-Model Evaluation Comparison - Research

**Researched:** 2026-04-03
**Domain:** LLM/VLM provider abstraction, multi-model evaluation framework
**Confidence:** HIGH

## Summary

Phase 14 extends the existing evaluation framework in `backend/evaluation/` to support HuggingFace and OpenAI vision providers alongside the existing Gemini provider. The codebase already has a well-structured provider abstraction (`LLMProvider` ABC in `llm_providers.py` with `GeminiProvider`, `OpenAIProvider`, `StubProvider`) and three evaluation scripts (`tracking_quality.py`, `llm_grounding.py`, `vlm_comparison.py`). The work is primarily: (1) add a `HuggingFaceProvider` to `llm_providers.py`, (2) add an `OpenAIVisionProvider` to `vlm_comparison.py`, (3) wire both into the CLI argument parsers, and (4) run all evaluations and collect output.

The existing architecture is clean and extensible. `llm_grounding.py` already runs 3 formats x 4 analysis types per provider. `vlm_comparison.py` currently hardcodes `GeminiVisionProvider` and needs refactoring to accept multiple vision providers. `tracking_quality.py` has no LLM dependency and just needs to be run with real data.

**Primary recommendation:** Follow the existing `LLMProvider` ABC pattern for HuggingFace; create a parallel `VisionProvider` ABC (or extend `LLMProvider`) for vision models; refactor `vlm_comparison.py` to accept pluggable vision providers instead of hardcoding Gemini.

## Project Constraints (from CLAUDE.md)

- Keep files under 500 lines (all eval files currently under 500 -- maintain this)
- NEVER hardcode API keys -- use env vars (HUGGINGFACE_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY)
- NEVER commit .env files
- Use typed interfaces for all public APIs
- ALWAYS read a file before editing it
- ALWAYS run tests after making code changes
- Prefer TDD London School (mock-first) for new code
- File organization: source in `/src`, tests in `/tests` (eval scripts live in `backend/evaluation/`)

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EVAL-01 | HuggingFace provider using huggingface_hub InferenceClient (Mistral-7B-Instruct) | `HuggingFaceProvider` class using `AsyncInferenceClient.chat.completions.create()` with OpenAI-compatible interface; add `huggingface-hub>=1.0.0` to requirements |
| EVAL-02 | Multi-model grounding: 3 models x 3 formats x 4 analysis types | Extend `llm_grounding.py` CLI `--provider` choices to include "huggingface"; add `--provider all` to run gemini+openai+huggingface sequentially; existing 3x4 matrix loops already work |
| EVAL-03 | Multi-model VLM comparison: 2 models x 3 conditions | Add `OpenAIVisionProvider` using `AsyncOpenAI` with base64 image content parts; refactor `vlm_comparison.py` to accept `--provider` flag and iterate over vision providers |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| huggingface-hub | 1.9.0 | HuggingFace Inference API client | Official SDK; `AsyncInferenceClient` has OpenAI-compatible chat.completions interface |
| openai | >=1.50.0 | OpenAI API (text + vision) | Already in requirements.txt; `AsyncOpenAI` supports vision via content parts |
| google-generativeai | >=0.8.0 | Gemini API | Already in requirements.txt; existing `GeminiProvider` and `GeminiVisionProvider` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| matplotlib | 3.9.4 | Chart generation (already installed) | Comparison bar charts, histograms |
| numpy | 2.1.2 | Numeric computation (already installed) | Metrics computation |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| huggingface-hub InferenceClient | Direct HF API via httpx | InferenceClient handles auth, retries, model routing; no reason to hand-roll |
| Mistral-7B-Instruct via HF | mistralai SDK directly | HF InferenceClient provides unified interface; Mistral SDK would be a separate dep |

**Installation:**
```bash
pip install "huggingface-hub>=1.0.0"
```

**Version verification:** huggingface-hub 1.9.0 is current as of 2026-04-03 (verified via pip index).

## Architecture Patterns

### Existing Provider Architecture (DO NOT CHANGE)
```
backend/
  api/
    services/
      llm_providers.py     # LLMProvider ABC + GeminiProvider + OpenAIProvider + StubProvider
      tactical.py           # TacticalAnalyzer, GroundingFormatter, SYSTEM_PROMPTS
  evaluation/
    _common.py             # EvalConfig, load_analytics, load_tracks, save helpers
    __init__.py            # Module docstring
    llm_grounding.py       # 3 formats x 4 analysis types, uses get_provider()
    vlm_comparison.py      # 3 conditions (text_only, raw_frames, annotated_frames), hardcoded GeminiVisionProvider
    tracking_quality.py    # Proxy metrics, no LLM dependency
    team_classification.py # Manual annotation workflow
    homography_error.py    # Reprojection error
```

### Pattern 1: HuggingFaceProvider (follows existing ABC)
**What:** New `LLMProvider` subclass using `huggingface_hub.AsyncInferenceClient`
**When to use:** `--provider huggingface` in llm_grounding.py
**Example:**
```python
# Source: huggingface_hub docs + existing OpenAIProvider pattern
from huggingface_hub import AsyncInferenceClient

class HuggingFaceProvider(LLMProvider):
    def __init__(self, api_key: str | None = None, model: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY", "")
        self.model_name = model
        self._client = None

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self) -> AsyncInferenceClient:
        if self._client is None:
            self._client = AsyncInferenceClient(api_key=self.api_key)
        return self._client

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""
```

### Pattern 2: OpenAIVisionProvider (extends OpenAI with image support)
**What:** Vision-capable provider using OpenAI's content parts format with base64 images
**When to use:** `--provider openai` in vlm_comparison.py
**Example:**
```python
# Source: OpenAI vision docs
import base64
from openai import AsyncOpenAI

class OpenAIVisionProvider:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model_name = model
        self._client = None

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def generate(
        self, system_prompt: str, user_prompt: str, images: list[bytes] | None = None
    ) -> str:
        client = self._get_client()
        content_parts = [{"type": "text", "text": user_prompt}]
        if images:
            for img_bytes in images:
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
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
```

### Pattern 3: Multi-Provider CLI Pattern
**What:** `--provider all` runs evaluation across all available providers
**When to use:** Both llm_grounding.py and vlm_comparison.py
**Example:**
```python
# In argparse:
parser.add_argument("--provider", default="gemini",
                    choices=["gemini", "openai", "huggingface", "all", "stub"])

# In run logic:
if config.provider == "all":
    providers_to_run = ["gemini", "openai", "huggingface"]
else:
    providers_to_run = [config.provider]

for provider_name in providers_to_run:
    provider = get_provider(provider_name)
    if not provider.is_available():
        print(f"  Skipping {provider_name} (no API key)")
        continue
    # ... run evaluation with this provider
```

### Anti-Patterns to Avoid
- **Hardcoding provider in evaluation logic:** `vlm_comparison.py` currently hardcodes `GeminiVisionProvider()` -- refactor to accept pluggable providers
- **Separate vision provider classes per model:** Use a unified interface with `images: list[bytes] | None` parameter
- **Running evaluations without data:** All eval scripts require real pipeline output (analytics.json, tracks.json, video.mp4) from RunPod

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HuggingFace API calls | Raw HTTP requests to HF Inference API | `huggingface_hub.AsyncInferenceClient` | Handles auth, rate limiting, model routing, retries |
| OpenAI vision message format | Custom image encoding logic | OpenAI SDK content parts format | Well-documented, handles multimodal properly |
| LaTeX table generation | Custom string formatting | Existing `save_latex_table()` in `_common.py` | Already works, used by all eval scripts |
| Figure saving | Manual plt.savefig calls | Existing `save_figure()` in `_common.py` | Saves PDF + PNG, handles directory creation |

**Key insight:** The evaluation framework already has robust helpers in `_common.py` for output generation. New providers only need to implement the `generate()` interface.

## Common Pitfalls

### Pitfall 1: HuggingFace Inference API Rate Limits
**What goes wrong:** Free HF Inference API has strict rate limits; Mistral-7B may queue or timeout
**Why it happens:** Free tier has limited throughput; running 3x4=12 evaluations can hit limits
**How to avoid:** Add retry logic with exponential backoff; consider using HF Pro for higher rate limits; use `HUGGINGFACE_API_KEY` env var (not just login token)
**Warning signs:** HTTP 429 responses, long queue times, timeout errors

### Pitfall 2: HuggingFace chat_completion Deprecated Task Issue
**What goes wrong:** `chat_completion` may use deprecated "conversational" task internally, causing failures for some models
**Why it happens:** HF Inference API changed task routing; some models require "text-generation" task
**How to avoid:** Use `client.chat.completions.create()` (OpenAI-compatible interface) instead of the older `client.chat_completion()` method; specify model explicitly; use huggingface-hub >= 1.0.0
**Warning signs:** "Task not found" or "conversational task deprecated" errors

### Pitfall 3: OpenAI Vision Base64 Format
**What goes wrong:** Images not recognized when sent as base64 to GPT-4o-mini
**Why it happens:** The data URL must include proper MIME type prefix: `data:image/jpeg;base64,{b64_data}`
**How to avoid:** Always use the full data URL format, not raw base64 string
**Warning signs:** "Invalid content type" or "image_url is only supported by certain models" errors

### Pitfall 4: GeminiVisionProvider vs General Vision Interface
**What goes wrong:** `vlm_comparison.py` currently uses `GeminiVisionProvider` directly with Gemini-specific `genai.types.Part` for images
**Why it happens:** VLM comparison was built Gemini-first without abstraction
**How to avoid:** Abstract the vision interface so both Gemini and OpenAI vision providers accept `images: list[bytes]` and handle format conversion internally
**Warning signs:** Import errors, type mismatches when swapping providers

### Pitfall 5: Missing Pipeline Output Data
**What goes wrong:** Eval scripts fail immediately because analytics.json/tracks.json don't exist locally
**Why it happens:** Phase depends on RunPod pipeline output; data must be downloaded first
**How to avoid:** Document required data files; add clear error messages; potentially include sample/fixture data for development
**Warning signs:** FileNotFoundError on first run

## Code Examples

### Registering HuggingFaceProvider in get_provider()
```python
# Source: Existing pattern in llm_providers.py
providers = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "huggingface": HuggingFaceProvider,  # NEW
    "stub": StubProvider,
}
```

### Vision Provider Interface (for vlm_comparison.py refactor)
```python
# Source: Existing GeminiVisionProvider pattern
class VisionProvider(ABC):
    @abstractmethod
    async def generate(
        self, system_prompt: str, user_prompt: str, images: list[bytes] | None = None
    ) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...
```

### Multi-Provider Grounding Output Structure
```python
# Source: Existing format_type_scores structure in llm_grounding.py
# Extended to: provider -> format -> analysis_type -> score
all_scores: dict[str, dict[str, dict[str, dict]]] = {}
# all_scores["gemini"]["markdown"]["match_overview"] = {grounding_rate: 0.85, ...}
# all_scores["openai"]["markdown"]["match_overview"] = {grounding_rate: 0.82, ...}
# all_scores["huggingface"]["markdown"]["match_overview"] = {grounding_rate: 0.71, ...}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `InferenceClient.chat_completion()` | `InferenceClient.chat.completions.create()` | huggingface-hub v1.0+ | OpenAI-compatible interface, avoids deprecated task routing |
| `genai.GenerativeModel` (sync) | `genai.GenerativeModel` + `run_in_executor` | Current codebase | Already handles async via thread pool |
| Single-provider eval | Multi-provider comparison | Phase 14 | Enables cross-model grounding comparison for dissertation |

**Deprecated/outdated:**
- `InferenceClient.chat_completion()` (old method): Use `client.chat.completions.create()` instead (OpenAI-compatible)
- `conversational` task in HF API: Deprecated in favor of `text-generation` task

## Open Questions

1. **Which Mistral model variant to use?**
   - What we know: Mistral-7B-Instruct-v0.3 is the latest v0.3; there's also v0.2 and v0.1
   - What's unclear: Whether v0.3 is available via free HF Inference API or requires Pro
   - Recommendation: Default to `mistralai/Mistral-7B-Instruct-v0.3`, fall back to v0.2 if unavailable

2. **Pipeline output data availability**
   - What we know: Phase depends on RunPod pipeline output data
   - What's unclear: Whether analytics.json/tracks.json/video.mp4 exist locally yet
   - Recommendation: Check for data files first; include clear instructions for downloading from RunPod

3. **HuggingFace API key scope**
   - What we know: `HUGGINGFACE_API_KEY` is needed; free tier has rate limits
   - What's unclear: Whether a free token suffices for 12+ sequential API calls in llm_grounding
   - Recommendation: Add rate-limit handling with backoff; document API key requirements in env.example

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | All eval scripts | Yes | 3.13.0 | -- |
| openai | OpenAI providers | Yes | 2.30.0 | -- |
| google-generativeai | Gemini providers | Yes | 0.8.6 | -- |
| huggingface-hub | HF provider (EVAL-01) | No | -- | `pip install huggingface-hub>=1.0.0` |
| matplotlib | Chart generation | Yes | 3.9.4 | -- |
| numpy | Metrics computation | Yes | 2.1.2 | -- |
| cv2 (opencv) | Frame extraction in VLM | Likely (pipeline dep) | -- | Required for vlm_comparison only |

**Missing dependencies with no fallback:**
- None (huggingface-hub is a simple pip install)

**Missing dependencies with fallback:**
- `huggingface-hub`: Not installed locally -- add to requirements.txt and pip install

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.0+ with pytest-asyncio |
| Config file | `backend/pyproject.toml` ([tool.pytest.ini_options]) |
| Quick run command | `cd backend && python -m pytest api/tests/ -x -q` |
| Full suite command | `cd backend && python -m pytest api/tests/ -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-01 | HuggingFaceProvider.generate() returns text | unit (mock) | `cd backend && python -m pytest api/tests/test_llm_providers.py -x` | No -- Wave 0 |
| EVAL-01 | HuggingFaceProvider registered in get_provider() | unit | Same file | No -- Wave 0 |
| EVAL-02 | llm_grounding.py --provider huggingface runs | integration (mock) | `cd backend && python -m pytest api/tests/test_eval_grounding.py -x` | No -- Wave 0 |
| EVAL-02 | llm_grounding.py --provider all iterates providers | unit | Same file | No -- Wave 0 |
| EVAL-03 | OpenAIVisionProvider.generate() with images | unit (mock) | `cd backend && python -m pytest api/tests/test_vlm_providers.py -x` | No -- Wave 0 |
| EVAL-03 | vlm_comparison.py --provider all runs matrix | integration (mock) | Same file | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `cd backend && python -m pytest api/tests/ -x -q`
- **Per wave merge:** `cd backend && python -m pytest api/tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `backend/api/tests/test_llm_providers.py` -- unit tests for HuggingFaceProvider (mock AsyncInferenceClient)
- [ ] `backend/api/tests/test_vlm_providers.py` -- unit tests for OpenAIVisionProvider (mock AsyncOpenAI with images)
- [ ] No test fixture changes needed -- existing `conftest.py` has async support configured

## Sources

### Primary (HIGH confidence)
- Existing codebase: `backend/api/services/llm_providers.py` -- provider ABC pattern, get_provider() registry
- Existing codebase: `backend/evaluation/llm_grounding.py` -- 3 formats x 4 analysis types evaluation loop
- Existing codebase: `backend/evaluation/vlm_comparison.py` -- GeminiVisionProvider, 3 conditions
- Existing codebase: `backend/evaluation/tracking_quality.py` -- proxy metrics, no LLM needed
- Existing codebase: `backend/evaluation/_common.py` -- EvalConfig, save helpers
- pip index: huggingface-hub 1.9.0 (verified 2026-04-03)

### Secondary (MEDIUM confidence)
- [HuggingFace Hub inference guide](https://github.com/huggingface/huggingface_hub/blob/main/docs/source/en/guides/inference.md) -- AsyncInferenceClient chat.completions.create() interface
- [HuggingFace Hub InferenceClient reference](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client) -- API reference
- [OpenAI Vision API docs](https://platform.openai.com/docs/guides/images-vision) -- base64 image content parts format
- [HF chat_completion deprecated task issue #3416](https://github.com/huggingface/huggingface_hub/issues/3416) -- conversational task deprecation

### Tertiary (LOW confidence)
- OpenAI community reports on base64 image compatibility with gpt-4o-mini -- some users report issues; needs validation during implementation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- existing codebase has clear patterns; HF SDK is well-documented
- Architecture: HIGH -- extending existing ABC with one new class; refactoring vlm_comparison.py to accept providers
- Pitfalls: MEDIUM -- HF rate limits and deprecated task routing need runtime validation

**Research date:** 2026-04-03
**Valid until:** 2026-05-03 (stable libraries, 30-day window)
