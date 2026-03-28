import { CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Hash, Flame } from "lucide-react";
import { useTeamColors } from "../context";

export function PlayerStatsTable({
  players, filterTeam, selectedPlayer, setSelectedPlayer,
}: {
  players: any[];
  filterTeam: "all" | "team1" | "team2";
  selectedPlayer: number | null;
  setSelectedPlayer: (id: number | null) => void;
}) {
  const { TEAM1_HEX, TEAM2_HEX } = useTeamColors();
  const filtered = filterTeam === "all" ? players : players.filter((p) => (filterTeam === "team1" ? p.teamId === 1 : p.teamId === 2));

  return (
    <div className="glass-card overflow-hidden hover-lift group">
      <div className="absolute -top-10 -left-10 w-32 h-32 rounded-full bg-violet-500/5 blur-3xl group-hover:bg-violet-500/10 transition-colors duration-700" />
      <CardHeader className="pb-4 pt-5 px-5 border-b border-border/30 relative">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="section-icon icon-accent">
              <Hash className="w-4 h-4 text-sky-400" />
            </div>
            <div>
              <CardTitle className="text-sm font-semibold">Per-Player Statistics</CardTitle>
              <CardDescription className="text-xs mt-0.5">Individual metrics by track ID &middot; Click to highlight on pitch</CardDescription>
            </div>
          </div>
          <Badge variant="outline" className="text-[10px] border-violet-400/20 text-violet-400/70 bg-violet-500/5">Placeholder</Badge>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border/20">
                <th className="text-left py-3 px-4 text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">Track ID</th>
                <th className="text-left py-3 px-4 text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">Team</th>
                <th className="text-right py-3 px-4 text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">Distance (km)</th>
                <th className="text-right py-3 px-4 text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">Avg Speed</th>
                <th className="text-right py-3 px-4 text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">Max Speed</th>
                <th className="text-right py-3 px-4 text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">Passes</th>
                <th className="text-right py-3 px-4 text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">Pass Acc%</th>
                <th className="text-right py-3 px-4 text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">Sprints</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((p) => {
                const isSelected = selectedPlayer === p.trackId;
                const teamColor = p.teamId === 1 ? TEAM1_HEX : TEAM2_HEX;
                return (
                  <tr
                    key={p.trackId}
                    onClick={() => setSelectedPlayer(isSelected ? null : p.trackId)}
                    className={`border-b border-border/10 cursor-pointer transition-all duration-300 ${
                      isSelected
                        ? "bg-white/[0.06] shadow-[inset_3px_0_0_0_var(--row-color)]"
                        : "hover:bg-white/[0.03]"
                    }`}
                    style={{ "--row-color": teamColor } as React.CSSProperties}
                  >
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <div className="w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold text-white border border-white/20"
                          style={{ backgroundColor: teamColor }}>
                          {p.trackId}
                        </div>
                        <span className="font-mono font-semibold">#{p.trackId}</span>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <span className="px-2 py-0.5 rounded-full text-[10px] font-medium" style={{ backgroundColor: `${teamColor}20`, color: teamColor }}>
                        Team {p.teamId}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right font-mono">{p.distance}</td>
                    <td className="py-3 px-4 text-right font-mono">{p.avgSpeed} km/h</td>
                    <td className="py-3 px-4 text-right font-mono font-semibold" style={{ color: teamColor }}>{p.maxSpeed} km/h</td>
                    <td className="py-3 px-4 text-right font-mono">{p.passes}</td>
                    <td className="py-3 px-4 text-right font-mono">{p.passAcc}%</td>
                    <td className="py-3 px-4 text-right">
                      <span className="inline-flex items-center gap-1">
                        <Flame className="w-3 h-3 text-orange-400" />
                        <span className="font-mono">{p.sprints}</span>
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </CardContent>
    </div>
  );
}
