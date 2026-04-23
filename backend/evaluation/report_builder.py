"""HTML report builder for unified evaluation results.

Produces a self-contained report.html from the dict returned by unified_runner.run_all().
Uses only stdlib (string formatting + base64) — no Jinja2 or external template deps.

Called by unified_runner.run_all() automatically, or standalone:
    from backend.evaluation.report_builder import build_report
    build_report(results, "eval_output/unified/report.html")
"""

import base64
import json
from pathlib import Path
from typing import Any


# ── Colour map for verdicts ───────────────────────────────────────────────────

_VERDICT_COLORS = {
    "verified":    ("#d4edda", "#155724", "Verified"),
    "refuted":     ("#f8d7da", "#721c24", "Refuted"),
    "unverifiable":("#fff3cd", "#856404", "Unverifiable"),
    "plausible":   ("#cce5ff", "#004085", "Plausible"),
}

_FORMAT_LABELS = {
    "markdown": "Structured Markdown (grounded)",
    "json":     "Raw JSON (control)",
    "prose":    "Natural Language Prose",
}

_ANALYSIS_LABELS = {
    "match_overview":     "Match Overview",
    "tactical_deep_dive": "Tactical Deep Dive",
    "event_analysis":     "Event Analysis",
    "player_spotlight":   "Player Spotlight",
}


# ── Image embedding ───────────────────────────────────────────────────────────


def _embed_png(png_path: str) -> str:
    """Return an <img> tag with base64-encoded PNG, or empty string if not found."""
    p = Path(png_path)
    if not p.exists():
        return f'<p class="muted">Chart not available: {p.name}</p>'
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" style="max-width:100%;border-radius:6px;">'


# ── Section builders ──────────────────────────────────────────────────────────


def _build_commentary_section(grounding: dict) -> str:
    """Section 1: LLM commentary output per format/analysis_type."""
    if not grounding or grounding.get("_status") == "error":
        return "<p>Grounding evaluation did not run or failed.</p>"

    provider_scores = grounding.get("provider_scores", {})
    if not provider_scores:
        return "<p>No grounding results available.</p>"

    # Look for artifacts in any of the standard locations
    output_dir = grounding.get("_output_dir", "")
    html_parts: list[str] = []

    for provider, fmt_scores in provider_scores.items():
        html_parts.append(f'<h3>Provider: <code>{provider}</code></h3>')
        for fmt_name, type_scores in fmt_scores.items():
            label = _FORMAT_LABELS.get(fmt_name, fmt_name)
            html_parts.append(f'<details><summary><strong>{label}</strong></summary>')
            for atype, score in type_scores.items():
                if not isinstance(score, dict):
                    continue
                atype_label = _ANALYSIS_LABELS.get(atype, atype)
                gr = score.get("grounding_rate", 0)
                n = score.get("total_claims", 0)
                color_class = "success" if gr >= 0.7 else ("warning" if gr >= 0.4 else "danger")
                html_parts.append(f"""
                <div class="card mb-2">
                  <div class="card-header">
                    {atype_label}
                    <span class="badge bg-{color_class} float-end">
                      Grounding: {gr:.1%} ({score.get('verified',0)}/{n} claims)
                    </span>
                  </div>
                </div>""")
            html_parts.append("</details>")

    return "\n".join(html_parts)


def _build_claims_table(artifacts_dir: str, provider: str = "openai") -> str:
    """Section 2: Per-claim verdict table from artifact JSON files."""
    art_dir = Path(artifacts_dir)
    if not art_dir.exists():
        return "<p>No artifact files found. Run the grounding evaluator first.</p>"

    rows_html: list[str] = []
    files = sorted(art_dir.glob(f"{provider}_markdown_*.json"))
    if not files:
        files = sorted(art_dir.glob("*.json"))[:4]

    for fpath in files:
        try:
            data = json.loads(fpath.read_text())
        except Exception:
            continue
        atype = data.get("analysis_type", fpath.stem)
        label = _ANALYSIS_LABELS.get(atype, atype)
        commentary = data.get("commentary", "")
        verification_results = data.get("verification_results", [])

        rows_html.append(f'<tr><td colspan="6" class="table-secondary fw-bold">{label}</td></tr>')

        for vr in verification_results:
            verdict = vr.get("verdict", "unverifiable")
            bg, fg, badge = _VERDICT_COLORS.get(verdict, ("#fff", "#000", verdict))
            claim_text = (vr.get("text") or "")[:120]
            ctype = vr.get("claim_type") or ""
            metric = (vr.get("referenced_metric") or "")[:60]
            actual = str(vr.get("actual_value") or "")[:40]
            explanation = (vr.get("explanation") or "")[:100]
            rows_html.append(f"""
            <tr style="background:{bg};color:{fg};">
              <td><span class="badge" style="background:{fg};color:{bg};">{badge}</span></td>
              <td><small>{claim_text}</small></td>
              <td><code><small>{ctype}</small></code></td>
              <td><code><small>{metric}</small></code></td>
              <td><small>{actual}</small></td>
              <td><small>{explanation}</small></td>
            </tr>""")

    if not rows_html:
        return "<p>No claim data found in artifacts.</p>"

    return f"""
    <div class="table-responsive">
      <table class="table table-sm table-bordered small">
        <thead class="table-dark">
          <tr>
            <th>Verdict</th>
            <th>Claim</th>
            <th>Type</th>
            <th>Metric</th>
            <th>Actual Value</th>
            <th>Explanation</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>"""


def _build_grounding_charts(grounding_dir: str, provider: str = "openai") -> str:
    """Section 3: Embedded grounding rate comparison charts."""
    gdir = Path(grounding_dir)
    chart_names = [
        f"grounding_format_comparison_{provider}_match_overview",
        f"grounding_by_type_{provider}_markdown",
    ]
    html_parts: list[str] = ['<div class="row">']
    found_any = False
    for name in chart_names:
        png_path = str(gdir / f"{name}.png")
        img = _embed_png(png_path)
        if "not available" not in img:
            found_any = True
        title = name.replace("_", " ").replace(provider, "").strip()
        html_parts.append(f"""
        <div class="col-md-6 mb-3">
          <div class="card">
            <div class="card-body p-2 text-center">
              <p class="text-muted small mb-1">{title}</p>
              {img}
            </div>
          </div>
        </div>""")
    html_parts.append("</div>")

    if not found_any:
        html_parts.append("""
        <div class="alert alert-warning">
          Charts not found. Run the grounding evaluator to generate them:<br>
          <code>python -m backend.evaluation.llm_grounding --analytics ... --provider openai --output ...</code>
        </div>""")

    return "\n".join(html_parts)


def _build_side_by_side(artifacts_dir: str, provider: str = "openai") -> str:
    """Section 4: Grounded (markdown) vs ungrounded (JSON) side-by-side comparison."""
    art_dir = Path(artifacts_dir)
    atype = "match_overview"

    markdown_file = art_dir / f"{provider}_markdown_{atype}.json"
    json_file = art_dir / f"{provider}_json_{atype}.json"

    def _load_condition(fpath: Path) -> dict:
        if fpath.exists():
            try:
                return json.loads(fpath.read_text())
            except Exception:
                pass
        return {}

    md_data = _load_condition(markdown_file)
    json_data = _load_condition(json_file)

    def _condition_card(data: dict, title: str, color: str) -> str:
        if not data:
            return f'<div class="col-md-6"><div class="card"><div class="card-body"><p class="text-muted">{title}: data not available</p></div></div></div>'
        score = data.get("score", {})
        gr = score.get("grounding_rate", 0)
        hr = score.get("hallucination_rate", 0)
        n = score.get("total_claims", 0)
        verified = score.get("verified", 0)
        refuted = score.get("refuted", 0)
        unverifiable = score.get("unverifiable", 0)
        fs = score.get("factscore", {})
        intrinsic = fs.get("intrinsic_hallucinations", 0)
        extrinsic = fs.get("extrinsic_hallucinations", 0)
        commentary_preview = (data.get("commentary", "")[:600] + "...") if data.get("commentary") else "N/A"
        commentary_preview = commentary_preview.replace("<", "&lt;").replace(">", "&gt;")

        return f"""
        <div class="col-md-6">
          <div class="card border-{color}">
            <div class="card-header bg-{color} text-white">
              <strong>{title}</strong>
              <span class="badge bg-light text-dark float-end">Grounding: {gr:.1%}</span>
            </div>
            <div class="card-body p-3">
              <div class="row text-center mb-3">
                <div class="col-4">
                  <div class="fw-bold text-success">{verified}</div>
                  <div class="text-muted small">Verified</div>
                </div>
                <div class="col-4">
                  <div class="fw-bold text-danger">{refuted}</div>
                  <div class="text-muted small">Refuted</div>
                </div>
                <div class="col-4">
                  <div class="fw-bold text-warning">{unverifiable}</div>
                  <div class="text-muted small">Unverifiable</div>
                </div>
              </div>
              <hr>
              <p class="small mb-1"><strong>FActScore breakdown:</strong></p>
              <ul class="list-unstyled small mb-3">
                <li>Intrinsic hallucinations: <strong class="text-danger">{intrinsic}</strong></li>
                <li>Extrinsic hallucinations: <strong class="text-warning">{extrinsic}</strong></li>
                <li>Grounded: <strong class="text-success">{score.get('factscore', {}).get('grounded', 0)}</strong></li>
                <li>Hallucination rate: <strong class="text-danger">{hr:.1%}</strong></li>
              </ul>
              <p class="small mb-1"><strong>Commentary (preview):</strong></p>
              <pre class="small p-2 bg-light rounded" style="max-height:200px;overflow-y:auto;white-space:pre-wrap;">{commentary_preview}</pre>
            </div>
          </div>
        </div>"""

    left = _condition_card(md_data, "Structured Markdown (Grounded)", "success")
    right = _condition_card(json_data, "Raw JSON (Ungrounded)", "danger")

    return f'<div class="row">{left}{right}</div>'


def _build_qa_section(qa: dict) -> str:
    """Build QA benchmark results section."""
    if not qa or qa.get("_status") in ("error", "skipped"):
        reason = qa.get("_error", qa.get("_reason", "not run"))
        return f'<div class="alert alert-warning">QA benchmark: {reason}</div>'

    overall = qa.get("overall_accuracy", 0)
    unans = qa.get("unanswerable_detection", {})
    by_cat = qa.get("by_category", {})

    cat_rows = ""
    for cat, stats in by_cat.items():
        pct = stats.get("accuracy", 0)
        c = stats.get("correct", 0)
        n = stats.get("n", 0)
        color = "success" if pct >= 0.7 else ("warning" if pct >= 0.4 else "danger")
        cat_rows += f"""
        <tr>
          <td>{cat.capitalize()}</td>
          <td>{n}</td>
          <td>{c}</td>
          <td><span class="badge bg-{color}">{pct:.1%}</span></td>
        </tr>"""

    return f"""
    <div class="row">
      <div class="col-md-6">
        <h5>Accuracy by Category</h5>
        <table class="table table-sm table-bordered">
          <thead class="table-dark">
            <tr><th>Category</th><th>N</th><th>Correct</th><th>Accuracy</th></tr>
          </thead>
          <tbody>
            {cat_rows}
            <tr class="table-secondary fw-bold">
              <td>Overall</td>
              <td>—</td>
              <td>—</td>
              <td>{overall:.1%}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="col-md-6">
        <h5>Unanswerable Detection (SQuAD 2.0)</h5>
        <table class="table table-sm table-bordered">
          <thead class="table-dark">
            <tr><th>Metric</th><th>Value</th></tr>
          </thead>
          <tbody>
            <tr><td>Precision</td><td>{unans.get('precision', 0):.1%}</td></tr>
            <tr><td>Recall</td>   <td>{unans.get('recall', 0):.1%}</td></tr>
            <tr><td>F1</td>       <td><strong>{unans.get('f1', 0):.1%}</strong></td></tr>
          </tbody>
        </table>
        <p class="text-muted small">F1 measures how well the model refuses to answer questions
        about data not present in the analytics (player names, xG, cards, attendance etc.).</p>
      </div>
    </div>"""


# ── Main report builder ───────────────────────────────────────────────────────


def build_report(results: dict, output_path: str) -> None:
    """Build a self-contained HTML report from unified_runner results.

    Args:
        results: dict returned by unified_runner.run_all()
        output_path: absolute or relative path to write report.html
    """
    meta = results.get("_meta", {})
    grounding = results.get("grounding", {})
    qa = results.get("qa", {})

    # Derive artifact paths from output_dir
    output_dir = meta.get("output_dir", str(Path(output_path).parent))
    provider = meta.get("provider", "openai")
    if provider == "all":
        provider = "openai"

    grounding_dir = str(Path(output_dir) / "grounding")
    artifacts_dir = str(Path(grounding_dir) / "artifacts")

    timestamp = meta.get("timestamp", "")
    analytics_path = meta.get("analytics_path", "unknown")

    # Build sections
    commentary_html = _build_commentary_section(grounding)
    claims_html = _build_claims_table(artifacts_dir, provider)
    charts_html = _build_grounding_charts(grounding_dir, provider)
    side_by_side_html = _build_side_by_side(artifacts_dir, provider)
    qa_html = _build_qa_section(qa)

    # Grounding summary badge
    provider_scores = grounding.get("provider_scores", {})
    summary_badges = ""
    for prov, fmt_scores in provider_scores.items():
        md_scores = fmt_scores.get("markdown", {})
        rates = [s.get("grounding_rate", 0) for s in md_scores.values() if isinstance(s, dict)]
        avg = sum(rates) / len(rates) if rates else 0.0
        color = "success" if avg >= 0.7 else ("warning" if avg >= 0.4 else "danger")
        summary_badges += f'<span class="badge bg-{color} me-2">{prov}: {avg:.1%} avg grounding (markdown)</span>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM/VLM Tactical Analysis Evaluation Report</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #f8f9fa; }}
  .section-card {{ background: white; border-radius: 8px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  .nav-pills .nav-link.active {{ background-color: #343a40; }}
  pre {{ font-size: 0.82rem; }}
  .muted {{ color: #6c757d; font-style: italic; }}
  .factscore-bar {{ height: 12px; border-radius: 6px; }}
</style>
</head>
<body>
<nav class="navbar navbar-dark bg-dark px-4 mb-4">
  <span class="navbar-brand fw-bold">Football Analytics — LLM/VLM Evaluation Report</span>
  <span class="text-light small">{timestamp[:19]} UTC</span>
</nav>

<div class="container-xl pb-5">

  <!-- Header -->
  <div class="section-card">
    <h4>Evaluation Summary</h4>
    <dl class="row mb-2">
      <dt class="col-sm-3">Analytics file</dt>
      <dd class="col-sm-9"><code>{analytics_path}</code></dd>
      <dt class="col-sm-3">Provider(s)</dt>
      <dd class="col-sm-9"><code>{meta.get('provider', '—')}</code></dd>
      <dt class="col-sm-3">Modules run</dt>
      <dd class="col-sm-9">{', '.join(meta.get('evals_run', []))}</dd>
    </dl>
    <div>{summary_badges}</div>
  </div>

  <!-- Nav tabs -->
  <ul class="nav nav-pills mb-4" id="evalTabs">
    <li class="nav-item"><a class="nav-link active" data-bs-toggle="pill" href="#sec-commentary">1. Commentary Output</a></li>
    <li class="nav-item"><a class="nav-link" data-bs-toggle="pill" href="#sec-claims">2. Claim Verdicts</a></li>
    <li class="nav-item"><a class="nav-link" data-bs-toggle="pill" href="#sec-charts">3. Grounding Charts</a></li>
    <li class="nav-item"><a class="nav-link" data-bs-toggle="pill" href="#sec-sidebyside">4. Grounded vs Ungrounded</a></li>
    <li class="nav-item"><a class="nav-link" data-bs-toggle="pill" href="#sec-qa">5. QA Benchmark</a></li>
  </ul>

  <div class="tab-content">

    <!-- Section 1: Commentary Output -->
    <div class="tab-pane fade show active" id="sec-commentary">
      <div class="section-card">
        <h4>1. LLM Commentary Output</h4>
        <p class="text-muted">Grounding rate summary per format and analysis type.
          Full commentary text is in <code>eval_output/unified/grounding/artifacts/*.json</code>.</p>
        {commentary_html}
      </div>
    </div>

    <!-- Section 2: Claim Verdicts -->
    <div class="tab-pane fade" id="sec-claims">
      <div class="section-card">
        <h4>2. Claim Verdicts</h4>
        <p class="text-muted">
          Each factual claim extracted from the LLM commentary is verified against the source analytics data.
          <span class="badge" style="background:#155724;color:#d4edda;">Verified</span>: claim matches analytics within 5% tolerance.
          <span class="badge" style="background:#721c24;color:#f8d7da;">Refuted</span>: claim contradicts analytics (intrinsic hallucination).
          <span class="badge" style="background:#856404;color:#fff3cd;">Unverifiable</span>: cannot be confirmed from available data (extrinsic hallucination).
          <span class="badge" style="background:#004085;color:#cce5ff;">Plausible</span>: qualitative claim consistent with tactical metrics.
        </p>
        {claims_html}
      </div>
    </div>

    <!-- Section 3: Charts -->
    <div class="tab-pane fade" id="sec-charts">
      <div class="section-card">
        <h4>3. Grounding Rate Comparison</h4>
        <p class="text-muted">Grounding rates by input format (structured markdown vs raw JSON vs prose)
          and by analysis type. Structured markdown consistently achieves higher grounding rates
          because the LLM has clearer access to verified data fields.</p>
        {charts_html}
      </div>
    </div>

    <!-- Section 4: Grounded vs Ungrounded Side-by-Side -->
    <div class="tab-pane fade" id="sec-sidebyside">
      <div class="section-card">
        <h4>4. Grounded vs Ungrounded Side-by-Side</h4>
        <p class="text-muted">
          Comparing the production grounding format (structured markdown) against the control condition
          (raw JSON). Lower grounding rate in the JSON condition demonstrates that input format
          significantly affects LLM hallucination frequency.
          FActScore methodology (Min et al. 2023): intrinsic = claim contradicts source;
          extrinsic = claim not verifiable from source.
        </p>
        {side_by_side_html}
      </div>
    </div>

    <!-- Section 5: QA Benchmark -->
    <div class="tab-pane fade" id="sec-qa">
      <div class="section-card">
        <h4>5. Chat QA Benchmark</h4>
        <p class="text-muted">RAGAS-style faithfulness evaluation with 45 auto-generated QA pairs
          across numeric, comparative, tactical, and unanswerable categories.
          Unanswerable detection (SQuAD 2.0) measures whether the model correctly refuses
          to answer questions not in the analytics data.</p>
        {qa_html}
      </div>
    </div>

  </div><!-- /tab-content -->
</div><!-- /container -->

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
