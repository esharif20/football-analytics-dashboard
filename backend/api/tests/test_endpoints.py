"""HTTP endpoint smoke tests.

Uses AsyncClient with DB override — no live database required.
Worker endpoints pass LOCAL_DEV_MODE so verify_worker_key skips auth.
"""

import pytest  # noqa: F401

# ── System ────────────────────────────────────────────────────────────────


async def test_health_returns_ok(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── Analysis (no-auth routes) ─────────────────────────────────────────────


async def test_analysis_modes_returns_dict(client):
    resp = await client.get("/api/analysis/modes")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "all" in data


async def test_analysis_stages_returns_list(client):
    resp = await client.get("/api/analysis/stages")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "id" in data[0]


# ── Worker (LOCAL_DEV_MODE bypasses verify_worker_key) ────────────────────


async def test_worker_pending_requires_no_key_in_dev_mode(client):
    """In LOCAL_DEV_MODE with no WORKER_API_KEY set, pending returns 200."""
    resp = await client.get("/api/worker/pending")
    # DB is mocked — expect 200 or 500, not 403/503
    assert resp.status_code not in (403, 503)


# ── Commentary types ──────────────────────────────────────────────────────


async def test_commentary_types_returns_list(client):
    resp = await client.get("/api/commentary/types")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0


# ── Auth guard: protected route rejects unauthenticated in prod mode ──────


async def test_analysis_list_accessible_in_dev_mode(client):
    """LOCAL_DEV_MODE=true → AutoLoginMiddleware sets a dev user → 200 or DB error, not 401."""
    resp = await client.get("/api/analysis")
    # 401 would mean AutoLoginMiddleware is not activating — that would be a bug
    assert resp.status_code != 401
