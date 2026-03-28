import { useMemo } from "react";

const toNumber = (v: any, fallback = 0) => { const n = Number(v); return Number.isFinite(n) ? n : fallback; };

/** Derive match-level statistics from DB statistics row OR raw analytics JSON. */
export function useDemoStats(statistics: any, analyticsData: any, demoEvents: any[]) {
  return useMemo(() => {
    if (statistics) {
      return {
        possessionTeam1: toNumber(statistics.possessionTeam1, 0), possessionTeam2: toNumber(statistics.possessionTeam2, 0),
        passesTeam1: toNumber(statistics.passesTeam1, 0), passesTeam2: toNumber(statistics.passesTeam2, 0),
        shotsTeam1: toNumber(statistics.shotsTeam1, 0), shotsTeam2: toNumber(statistics.shotsTeam2, 0),
        distanceCoveredTeam1: toNumber(statistics.distanceCoveredTeam1, 0), distanceCoveredTeam2: toNumber(statistics.distanceCoveredTeam2, 0),
        avgSpeedTeam1: toNumber(statistics.avgSpeedTeam1, 0), avgSpeedTeam2: toNumber(statistics.avgSpeedTeam2, 0),
        maxSpeedTeam1: toNumber(statistics.maxSpeedTeam1, 0), maxSpeedTeam2: toNumber(statistics.maxSpeedTeam2, 0),
        possessionChanges: toNumber(statistics.possessionChanges, 0), ballDistance: toNumber(statistics.ballDistance, 0),
        ballAvgSpeed: toNumber(statistics.ballAvgSpeed, 0), ballMaxSpeed: toNumber(statistics.ballMaxSpeed, 0),
        directionChanges: toNumber(statistics.directionChanges, 0),
      };
    }
    const poss = analyticsData?.possession ?? {};
    const playerKinematics = analyticsData?.player_kinematics ?? {};
    const ballKinematics = analyticsData?.ball_kinematics ?? {};
    const ballPath = analyticsData?.ball_path ?? {};
    let team1DistanceM = 0, team2DistanceM = 0, team1TrackedTimeS = 0, team2TrackedTimeS = 0;
    let team1MaxSpeedMps = 0, team2MaxSpeedMps = 0;
    for (const pdata of Object.values(playerKinematics) as any[]) {
      if (!pdata || typeof pdata !== "object") continue;
      const rawTeamId = pdata.team_id;
      if (rawTeamId !== 0 && rawTeamId !== 1) continue;
      const d = toNumber(pdata.total_distance_m, 0), avg = toNumber(pdata.avg_speed_m_per_sec, 0), mx = toNumber(pdata.max_speed_m_per_sec, 0);
      if (rawTeamId === 0) { team1DistanceM += d; if (d > 0 && avg > 0) team1TrackedTimeS += d / avg; team1MaxSpeedMps = Math.max(team1MaxSpeedMps, mx); }
      else { team2DistanceM += d; if (d > 0 && avg > 0) team2TrackedTimeS += d / avg; team2MaxSpeedMps = Math.max(team2MaxSpeedMps, mx); }
    }
    const passesTeam1 = demoEvents.filter((e: any) => e.type === "pass" && e.teamId === 1).length;
    const passesTeam2 = demoEvents.filter((e: any) => e.type === "pass" && e.teamId === 2).length;
    const shotsTeam1 = demoEvents.filter((e: any) => e.type === "shot" && e.teamId === 1).length;
    const shotsTeam2 = demoEvents.filter((e: any) => e.type === "shot" && e.teamId === 2).length;
    const possTeam1 = toNumber(poss.team_1_percentage, 0);
    const possTeam2 = toNumber(poss.team_2_percentage, Math.max(0, 100 - possTeam1));
    const ballDistanceM = toNumber(ballPath.total_distance_m ?? ballKinematics.total_distance_m, 0);
    return {
      possessionTeam1: possTeam1, possessionTeam2: possTeam2, passesTeam1, passesTeam2, shotsTeam1, shotsTeam2,
      distanceCoveredTeam1: +(team1DistanceM / 1000).toFixed(2), distanceCoveredTeam2: +(team2DistanceM / 1000).toFixed(2),
      avgSpeedTeam1: team1TrackedTimeS > 0 ? +((team1DistanceM / team1TrackedTimeS) * 3.6).toFixed(1) : 0,
      avgSpeedTeam2: team2TrackedTimeS > 0 ? +((team2DistanceM / team2TrackedTimeS) * 3.6).toFixed(1) : 0,
      maxSpeedTeam1: +(team1MaxSpeedMps * 3.6).toFixed(1), maxSpeedTeam2: +(team2MaxSpeedMps * 3.6).toFixed(1),
      possessionChanges: toNumber(poss.possession_changes, 0), ballDistance: +(ballDistanceM / 1000).toFixed(2),
      ballAvgSpeed: +(toNumber(ballKinematics.avg_speed_m_per_sec, 0) * 3.6).toFixed(1),
      ballMaxSpeed: +(toNumber(ballKinematics.max_speed_m_per_sec, 0) * 3.6).toFixed(1),
      directionChanges: toNumber(ballPath.direction_changes, 0),
    };
  }, [statistics, analyticsData, demoEvents]);
}

/** Derive per-player stats from analytics kinematics + events. */
export function useDemoPlayerStats(analyticsData: any, demoEvents: any[]) {
  return useMemo(() => {
    const playerKinematics = analyticsData?.player_kinematics;
    if (!playerKinematics || typeof playerKinematics !== "object") return [];
    const passAttemptsByPlayer = new Map<number, number>();
    const passSuccessByPlayer = new Map<number, number>();
    for (const ev of demoEvents) {
      if (ev.type !== "pass") continue;
      const pid = Number(ev.playerId);
      if (!Number.isFinite(pid)) continue;
      passAttemptsByPlayer.set(pid, (passAttemptsByPlayer.get(pid) ?? 0) + 1);
      if (ev.success !== false) passSuccessByPlayer.set(pid, (passSuccessByPlayer.get(pid) ?? 0) + 1);
    }
    const rows = Object.entries(playerKinematics)
      .map(([trackIdRaw, pdata]: [string, any]) => {
        if (!pdata || typeof pdata !== "object") return null;
        const trackId = Number(trackIdRaw);
        if (!Number.isFinite(trackId)) return null;
        const rawTeamId = pdata.team_id;
        if (rawTeamId !== 0 && rawTeamId !== 1) return null;
        const teamId = rawTeamId + 1;
        const distanceM = Number(pdata.total_distance_m), avgSpeedMps = Number(pdata.avg_speed_m_per_sec), maxSpeedMps = Number(pdata.max_speed_m_per_sec);
        const passes = passAttemptsByPlayer.get(trackId) ?? 0, passSuccess = passSuccessByPlayer.get(trackId) ?? 0;
        return {
          trackId, teamId,
          distance: Number.isFinite(distanceM) ? +(distanceM / 1000).toFixed(2) : 0,
          avgSpeed: Number.isFinite(avgSpeedMps) ? +(avgSpeedMps * 3.6).toFixed(1) : 0,
          maxSpeed: Number.isFinite(maxSpeedMps) ? +(maxSpeedMps * 3.6).toFixed(1) : 0,
          passes, passAcc: passes > 0 ? Math.round((passSuccess / passes) * 100) : null,
          sprints: null as number | null,
        };
      })
      .filter((row): row is NonNullable<typeof row> => row !== null);
    rows.sort((a, b) => b.distance - a.distance);
    return rows;
  }, [analyticsData?.player_kinematics, demoEvents]);
}
