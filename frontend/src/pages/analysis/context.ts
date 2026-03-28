import { createContext, useContext, useState, useEffect, useRef } from "react";

// ==================== Constants ====================
export const PITCH_WIDTH = 105;
export const PITCH_HEIGHT = 68;
export const TEAM1_DEFAULT = "#e05252";
export const TEAM2_DEFAULT = "#4a9ede";

// Team color context — lets all sub-components read pipeline-detected jersey colors
export const TeamColorsCtx = createContext({ TEAM1_HEX: TEAM1_DEFAULT, TEAM2_HEX: TEAM2_DEFAULT });
export function useTeamColors() { return useContext(TeamColorsCtx); }

// ==================== Custom Tooltip ====================
export function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <p className="text-[10px] text-muted-foreground mb-1 uppercase tracking-wider">{label}</p>
      {payload.map((p: any, i: number) => (
        <p key={i} className="text-xs font-semibold font-mono" style={{ color: p.color }}>
          {p.name}: {typeof p.value === "number" ? p.value.toFixed(1) : p.value}
        </p>
      ))}
    </div>
  );
}

// ==================== Section Wrapper (animation helper) ====================
export function AnimatedSection({ children, className = "", delay = 0 }: { children: React.ReactNode; className?: string; delay?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(([e]) => { if (e.isIntersecting) { setVisible(true); obs.disconnect(); } }, { threshold: 0.1 });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);
  return (
    <div
      ref={ref}
      className={`transition-all duration-700 ease-out ${visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"} ${className}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
}

// ==================== Shared Type Re-exports ====================
export type { PipelineMode } from "@/shared/types";
export { PIPELINE_MODES, PROCESSING_STAGES, EVENT_TYPES } from "@/shared/types";
