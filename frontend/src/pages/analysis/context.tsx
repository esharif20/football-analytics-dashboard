import { createContext, useContext } from 'react'
import { motion, type Variants } from 'framer-motion'
import { ScrollReveal } from '@/components/ScrollReveal'
import { SplitText } from '@/components/SplitText'

// ==================== Constants ====================
export const PITCH_WIDTH = 105
export const PITCH_HEIGHT = 68
export const TEAM1_DEFAULT = '#e05252'
export const TEAM2_DEFAULT = '#4a9ede'

// Team color context — lets all sub-components read pipeline-detected jersey colors
export const TeamColorsCtx = createContext({ TEAM1_HEX: TEAM1_DEFAULT, TEAM2_HEX: TEAM2_DEFAULT })
export function useTeamColors() {
  return useContext(TeamColorsCtx)
}

// ==================== Custom Tooltip ====================
export function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <p className="text-[10px] text-muted-foreground mb-1 uppercase tracking-wider">{label}</p>
      {payload.map((p: any, i: number) => (
        <p key={i} className="text-xs font-semibold font-mono" style={{ color: p.color }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(1) : p.value}
        </p>
      ))}
    </div>
  )
}

// ==================== Framer Motion Presets ====================
const EASE = [0.16, 1, 0.3, 1] as const

const revealVariants: Variants = {
  hidden: { opacity: 0, y: 40, filter: 'blur(6px)' },
  visible: { opacity: 1, y: 0, filter: 'blur(0px)' },
}

// Stagger container for grids — wrap children in AnimatedSection
export const staggerGridVariants: Variants = {
  hidden: {},
  visible: {
    transition: { staggerChildren: 0.1, delayChildren: 0.05 },
  },
}

export function StaggerGrid({
  children,
  className = '',
}: {
  children: React.ReactNode
  className?: string
}) {
  return (
    <motion.div
      className={className}
      variants={staggerGridVariants}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, amount: 0.05 }}
    >
      {children}
    </motion.div>
  )
}

// ==================== Section Wrapper (framer-motion scroll reveal) ====================
export function AnimatedSection({
  children,
  className = '',
  delay = 0,
}: {
  children: React.ReactNode
  className?: string
  delay?: number
}) {
  return (
    <motion.div
      className={className}
      variants={revealVariants}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, amount: 0.08 }}
      transition={{
        duration: 0.7,
        delay: delay / 1000,
        ease: EASE,
      }}
    >
      {children}
    </motion.div>
  )
}

// ==================== Section Heading ====================
export function SectionHeading({ number, title }: { number: string; title: string }) {
  return (
    <ScrollReveal variant="fadeUp" className="flex items-center gap-4 pt-8 pb-1">
      <span className="text-[10px] font-mono text-primary/50 tracking-[0.25em] uppercase shrink-0">
        {number}
      </span>
      <h2
        className="text-[clamp(1.4rem,2.8vw,2rem)] font-black uppercase tracking-tight text-foreground/90 leading-none shrink-0"
        style={{ fontFamily: 'Barlow Condensed, Inter, sans-serif' }}
      >
        <SplitText text={title} />
      </h2>
      <div className="flex-1 h-px bg-gradient-to-r from-primary/25 via-primary/10 to-transparent" />
    </ScrollReveal>
  )
}

// ==================== Shared Type Re-exports ====================
export type { PipelineMode } from '@/types'
export { PIPELINE_MODES, PROCESSING_STAGES, EVENT_TYPES } from '@/types'
