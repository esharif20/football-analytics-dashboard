-- Phase 8: Schema Redesign -- FK constraints, indexes, RLS
-- Applies to existing tables created by Alembic baseline migration
-- Safe to run on fresh DB (IF NOT EXISTS guards) or existing DB (DO $$ blocks skip existing constraints)

BEGIN;

-- ============================================================
-- 1. ENUM TYPES (create if not exists)
-- ============================================================
DO $$ BEGIN
  CREATE TYPE userrole AS ENUM ('user', 'admin');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
  CREATE TYPE pipelinemode AS ENUM ('all', 'radar', 'team', 'track', 'players', 'ball', 'pitch');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
  CREATE TYPE processingstatus AS ENUM ('pending', 'uploading', 'processing', 'completed', 'failed');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ============================================================
-- 2. FOREIGN KEY CONSTRAINTS (add if not exists)
-- ============================================================

-- videos.userId -> users.id
DO $$ BEGIN
  ALTER TABLE videos
    ADD CONSTRAINT fk_videos_user FOREIGN KEY ("userId") REFERENCES users(id) ON DELETE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- analyses.userId -> users.id
DO $$ BEGIN
  ALTER TABLE analyses
    ADD CONSTRAINT fk_analyses_user FOREIGN KEY ("userId") REFERENCES users(id) ON DELETE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- analyses.videoId -> videos.id
DO $$ BEGIN
  ALTER TABLE analyses
    ADD CONSTRAINT fk_analyses_video FOREIGN KEY ("videoId") REFERENCES videos(id) ON DELETE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- events.analysisId -> analyses.id
DO $$ BEGIN
  ALTER TABLE events
    ADD CONSTRAINT fk_events_analysis FOREIGN KEY ("analysisId") REFERENCES analyses(id) ON DELETE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- tracks.analysisId -> analyses.id
DO $$ BEGIN
  ALTER TABLE tracks
    ADD CONSTRAINT fk_tracks_analysis FOREIGN KEY ("analysisId") REFERENCES analyses(id) ON DELETE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- statistics.analysisId -> analyses.id
DO $$ BEGIN
  ALTER TABLE statistics
    ADD CONSTRAINT fk_statistics_analysis FOREIGN KEY ("analysisId") REFERENCES analyses(id) ON DELETE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- commentary.analysisId -> analyses.id
DO $$ BEGIN
  ALTER TABLE commentary
    ADD CONSTRAINT fk_commentary_analysis FOREIGN KEY ("analysisId") REFERENCES analyses(id) ON DELETE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- commentary.eventId -> events.id (nullable FK -- ON DELETE SET NULL so commentary survives event deletion)
DO $$ BEGIN
  ALTER TABLE commentary
    ADD CONSTRAINT fk_commentary_event FOREIGN KEY ("eventId") REFERENCES events(id) ON DELETE SET NULL;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ============================================================
-- 3. PERFORMANCE INDEXES
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_videos_user_id ON videos ("userId");
CREATE INDEX IF NOT EXISTS idx_analyses_user_id ON analyses ("userId");
CREATE INDEX IF NOT EXISTS idx_analyses_video_id ON analyses ("videoId");
CREATE INDEX IF NOT EXISTS idx_events_analysis_id ON events ("analysisId");
CREATE INDEX IF NOT EXISTS idx_tracks_analysis_id ON tracks ("analysisId");
CREATE INDEX IF NOT EXISTS idx_tracks_analysis_frame ON tracks ("analysisId", "frameNumber");
CREATE INDEX IF NOT EXISTS idx_statistics_analysis_id ON statistics ("analysisId");
CREATE INDEX IF NOT EXISTS idx_commentary_analysis_id ON commentary ("analysisId");

-- ============================================================
-- 4. ROW LEVEL SECURITY
-- Permissive now (USING (true)) -- strict RLS requires Supabase Auth JWT integration (future work).
-- RLS is ENABLED so the policies exist; the permissive policy prevents breaking current auto-login flow.
-- ============================================================

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE videos ENABLE ROW LEVEL SECURITY;
ALTER TABLE analyses ENABLE ROW LEVEL SECURITY;

-- Drop and recreate policies to be idempotent
DROP POLICY IF EXISTS users_open_access ON users;
DROP POLICY IF EXISTS videos_open_access ON videos;
DROP POLICY IF EXISTS analyses_open_access ON analyses;

CREATE POLICY users_open_access ON users FOR ALL USING (true);
CREATE POLICY videos_open_access ON videos FOR ALL USING (true);
CREATE POLICY analyses_open_access ON analyses FOR ALL USING (true);

-- Comment documents the intent for future strict RLS:
-- To enable strict RLS: replace USING (true) with USING (auth.uid()::text = "openId")
-- for users table and USING ("userId" = (SELECT id FROM users WHERE "openId" = auth.uid()::text))
-- for videos and analyses tables. Requires Supabase Auth JWT to be passed in requests.

COMMIT;
