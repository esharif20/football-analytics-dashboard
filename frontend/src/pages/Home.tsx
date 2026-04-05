import { useState, useEffect, useRef } from 'react'
import { motion, useMotionValue, useSpring, useScroll, useTransform } from 'framer-motion'
import { PitchCanvas } from '@/components/PitchCanvas'
import { useAuth } from '@/hooks/useAuth'
import { Button } from '@/components/ui/button'
import { getLoginUrl } from '@/const'
import { Link } from 'wouter'
import { useLenis } from '@/hooks/useLenis'
import { ScrollReveal } from '@/components/ScrollReveal'
import { SplitText } from '@/components/SplitText'
import { ScrollStagger, StaggerItem } from '@/components/ScrollStagger'
import { FeatureCard } from '@/components/ui/grid-feature-cards'
import DotCard from '@/components/ui/moving-dot-card'
import { BauhausCard } from '@/components/ui/bauhaus-card'
import { ParallaxSection } from '@/components/ParallaxSection'
import {
  Activity,
  BarChart3,
  Flame,
  Play,
  Radar,
  Target,
  Upload,
  Users,
  Zap,
  ArrowRight,
  CheckCircle2,
  Cpu,
  Layers,
  Eye,
  Map,
  Circle,
  GitBranch,
  Sparkles,
  ArrowUp,
} from 'lucide-react'

// ─── Custom Cursor ────────────────────────────────────────────────────────────
function CustomCursor() {
  const [hovering, setHovering] = useState(false)
  const [visible, setVisible] = useState(false)

  const mx = useMotionValue(-100)
  const my = useMotionValue(-100)

  // Dot tracks instantly (stiff spring ≈ direct follow)
  const dotX = useSpring(mx, { stiffness: 2000, damping: 60, mass: 0.2 })
  const dotY = useSpring(my, { stiffness: 2000, damping: 60, mass: 0.2 })

  // Ring lags behind with softer spring
  const ringX = useSpring(mx, { stiffness: 300, damping: 28, mass: 0.6 })
  const ringY = useSpring(my, { stiffness: 300, damping: 28, mass: 0.6 })

  useEffect(() => {
    // Skip on touch devices
    if (window.matchMedia('(pointer: coarse)').matches) return

    const onMove = (e: MouseEvent) => {
      mx.set(e.clientX)
      my.set(e.clientY)
      if (!visible) setVisible(true)
    }
    const onOver = (e: MouseEvent) => {
      if ((e.target as HTMLElement).closest('a, button, [role="button"]')) setHovering(true)
    }
    const onOut = (e: MouseEvent) => {
      if ((e.target as HTMLElement).closest('a, button, [role="button"]')) setHovering(false)
    }
    const onLeave = () => setVisible(false)
    const onEnter = () => setVisible(true)

    window.addEventListener('mousemove', onMove)
    document.addEventListener('mouseover', onOver)
    document.addEventListener('mouseout', onOut)
    document.addEventListener('mouseleave', onLeave)
    document.addEventListener('mouseenter', onEnter)
    return () => {
      window.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseover', onOver)
      document.removeEventListener('mouseout', onOut)
      document.removeEventListener('mouseleave', onLeave)
      document.removeEventListener('mouseenter', onEnter)
    }
  }, [mx, my, visible])

  return (
    <>
      {/* Dot — snappy, precise */}
      <motion.div
        className="custom-cursor-dot"
        style={{ x: dotX, y: dotY, translateX: '-50%', translateY: '-50%' }}
        animate={{
          opacity: visible ? 1 : 0,
          scale: hovering ? 1.8 : 1,
          backgroundColor: hovering ? '#ffffff' : 'oklch(0.65 0.2 145)',
        }}
        transition={{
          opacity: { duration: 0.2 },
          scale: { type: 'spring', stiffness: 500, damping: 30 },
          backgroundColor: { duration: 0.2 },
        }}
      />
    </>
  )
}

// ─── Parallax Divider ──────────────────────────────────────────────────────────
function ParallaxDivider() {
  const ref = useRef<HTMLDivElement>(null)
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start end', 'end start'] })
  const backX = useTransform(scrollYProgress, [0, 1], [-60, 60])
  const midX = useTransform(scrollYProgress, [0, 1], [-120, 120])
  const frontX = useTransform(scrollYProgress, [0, 1], [-30, 30])
  const frontOpacity = useTransform(scrollYProgress, [0, 0.5, 1], [0.4, 0.9, 0.4])

  return (
    <div
      ref={ref}
      className="relative overflow-hidden h-[100px]"
      style={{
        borderTop: '1px solid rgba(34,197,94,0.15)',
        borderBottom: '1px solid rgba(34,197,94,0.15)',
      }}
    >
      {/* Back: moving grid */}
      <motion.div
        className="absolute inset-y-0 pointer-events-none"
        style={{
          x: backX,
          left: '-50%',
          width: '200%',
          backgroundImage:
            'linear-gradient(rgba(34,197,94,0.07) 1px, transparent 1px), linear-gradient(90deg, rgba(34,197,94,0.07) 1px, transparent 1px)',
          backgroundSize: '40px 40px',
        }}
      />

      {/* Middle: pitch SVG strip */}
      <motion.div
        className="absolute inset-y-0 flex items-center"
        style={{ x: midX, left: '-50%', width: '200%' }}
      >
        <svg
          viewBox="0 0 210 68"
          preserveAspectRatio="xMidYMid meet"
          className="w-full h-full opacity-[0.13]"
        >
          <rect x="0" y="0" width="210" height="68" fill="none" stroke="white" strokeWidth="0.5" />
          <line x1="105" y1="0" x2="105" y2="68" stroke="white" strokeWidth="0.5" />
          <circle cx="105" cy="34" r="9.15" fill="none" stroke="white" strokeWidth="0.5" />
          <circle cx="105" cy="34" r="0.8" fill="white" />
          <rect
            x="0"
            y="24.84"
            width="16.5"
            height="18.32"
            fill="none"
            stroke="white"
            strokeWidth="0.5"
          />
          <rect
            x="193.5"
            y="24.84"
            width="16.5"
            height="18.32"
            fill="none"
            stroke="white"
            strokeWidth="0.5"
          />
          <rect
            x="0"
            y="29.84"
            width="5.5"
            height="8.32"
            fill="none"
            stroke="white"
            strokeWidth="0.5"
          />
          <rect
            x="204.5"
            y="29.84"
            width="5.5"
            height="8.32"
            fill="none"
            stroke="white"
            strokeWidth="0.5"
          />
        </svg>
      </motion.div>

      {/* Front: gradient glow sweep */}
      <motion.div
        className="absolute inset-y-0 pointer-events-none"
        style={{
          x: frontX,
          left: '-50%',
          width: '200%',
          opacity: frontOpacity,
          background:
            'linear-gradient(90deg, transparent 15%, rgba(34,197,94,0.1) 40%, rgba(34,197,94,0.2) 50%, rgba(34,197,94,0.1) 60%, transparent 85%)',
        }}
      />
    </div>
  )
}

// ─── Section Header ────────────────────────────────────────────────────────────
function SectionHeader({
  number,
  label,
  title,
  description,
}: {
  number: string
  label: string
  title: string
  description?: string
}) {
  return (
    <div className="mb-12">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-4">
          <span className="text-xs font-mono text-primary tracking-widest uppercase">{number}</span>
          <span className="text-xs font-mono text-muted-foreground tracking-widest uppercase">
            {label}
          </span>
        </div>
        <div className="w-6 h-6 rounded-full border border-primary/40 flex items-center justify-center">
          <div className="w-2 h-2 rounded-full bg-primary" />
        </div>
      </div>
      <div className="w-full h-px bg-border/50 mb-6" />
      <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-4">
        <h2
          className="text-[2.5rem] md:text-[3.5rem] lg:text-[4.5vw] font-black leading-none tracking-tight uppercase"
          style={{ fontFamily: 'Barlow Condensed, Inter, sans-serif' }}
        >
          <SplitText text={title} />
        </h2>
        {description && (
          <p className="text-muted-foreground max-w-sm text-sm leading-relaxed lg:text-right">
            {description}
          </p>
        )}
      </div>
    </div>
  )
}

// ─── Floating Stat Card ────────────────────────────────────────────────────────
function FloatingStatCard({
  value,
  label,
  icon,
  className = '',
}: {
  value: string
  label: string
  icon: React.ReactNode
  className?: string
}) {
  return (
    <div
      className={`absolute bg-card/90 backdrop-blur-md border border-border/60 rounded-2xl p-4 shadow-2xl shadow-black/40 ${className}`}
    >
      <div className="flex items-center gap-3 mb-1">
        <span className="text-primary">{icon}</span>
        <span
          className="text-2xl font-black tracking-tight"
          style={{ fontFamily: 'Barlow Condensed, Inter, sans-serif' }}
        >
          {value}
        </span>
      </div>
      <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider">{label}</p>
    </div>
  )
}

// ─── Pipeline Feature Card ─────────────────────────────────────────────────────
function PipelineCard({
  topLeft,
  topRight,
  icon,
  title,
  description,
  number,
  href,
  disabled,
}: {
  topLeft: string
  topRight: string
  icon: React.ReactNode
  title: string
  description: string
  number: string
  href?: string
  disabled?: boolean
}) {
  const [isHovered, setIsHovered] = useState(false)

  const inner = (
    <motion.div
      className={`relative h-full py-4 px-5 flex flex-col justify-between rounded-xl overflow-hidden
        ${disabled ? 'bg-secondary/20 cursor-default' : 'bg-card cursor-pointer'}
      `}
      style={{
        border: '1px solid rgba(255,255,255,0.07)',
        boxShadow: '0 2px 12px rgba(0,0,0,0.25)',
      }}
      onHoverStart={() => !disabled && setIsHovered(true)}
      onHoverEnd={() => setIsHovered(false)}
      whileHover={
        disabled
          ? {}
          : {
              y: -16,
              scale: 1.03,
              border: '1px solid rgba(34,197,94,0.75)',
              boxShadow:
                '0 0 0 1px rgba(34,197,94,0.3), 0 0 60px rgba(34,197,94,0.2), 0 24px 64px rgba(0,0,0,0.65)',
            }
      }
      transition={{ type: 'spring', stiffness: 380, damping: 28 }}
    >
      {/* Shimmer sweep */}
      <motion.div
        className="absolute inset-y-0 left-0 w-1/2 pointer-events-none z-10"
        style={{
          background: 'linear-gradient(90deg, transparent, rgba(34,197,94,0.13), transparent)',
        }}
        animate={isHovered ? { x: '300%', opacity: [0, 1, 1, 0] } : { x: '-100%', opacity: 0 }}
        transition={{ duration: 0.75, ease: 'easeInOut' }}
      />

      {/* Corner accents */}
      <motion.div
        className="absolute top-0 left-0 w-5 h-5 pointer-events-none"
        style={{
          borderTop: '2px solid rgba(34,197,94,0.9)',
          borderLeft: '2px solid rgba(34,197,94,0.9)',
        }}
        animate={isHovered ? { opacity: 1 } : { opacity: 0 }}
        transition={{ duration: 0.2 }}
      />
      <motion.div
        className="absolute bottom-0 right-0 w-5 h-5 pointer-events-none"
        style={{
          borderBottom: '2px solid rgba(34,197,94,0.9)',
          borderRight: '2px solid rgba(34,197,94,0.9)',
        }}
        animate={isHovered ? { opacity: 1 } : { opacity: 0 }}
        transition={{ duration: 0.2, delay: 0.07 }}
      />

      {/* Top row */}
      <div className="flex items-center justify-between mb-8">
        <span className="text-xs font-mono text-muted-foreground uppercase tracking-widest">
          {topLeft}
        </span>
        <span className="text-xs font-mono text-muted-foreground uppercase tracking-widest">
          {topRight}
        </span>
      </div>

      {/* Center icon + title */}
      <div className="flex flex-col items-center text-center gap-3 py-8">
        <motion.div
          className="w-12 h-12 rounded-full border flex items-center justify-center"
          animate={
            isHovered
              ? {
                  scale: 1.35,
                  borderColor: 'oklch(0.65 0.2 145)',
                  backgroundColor: 'oklch(0.65 0.2 145 / 0.15)',
                  color: 'oklch(0.65 0.2 145)',
                }
              : {
                  scale: 1,
                  borderColor: 'rgba(255,255,255,0.15)',
                  backgroundColor: 'transparent',
                  color: 'oklch(0.65 0.2 145)',
                }
          }
          transition={{ type: 'spring', stiffness: 450, damping: 28 }}
        >
          {icon}
        </motion.div>
        <motion.h3
          className="text-xl font-black uppercase tracking-tight"
          style={{ fontFamily: 'Barlow Condensed, Inter, sans-serif' }}
          animate={
            isHovered ? { color: 'oklch(0.65 0.2 145)' } : { color: 'rgba(255,255,255,0.95)' }
          }
          transition={{ duration: 0.22 }}
        >
          {title}
        </motion.h3>
        <motion.div
          className="h-px bg-primary"
          animate={isHovered ? { width: 80, opacity: 0.8 } : { width: 48, opacity: 0.3 }}
          transition={{ type: 'spring', stiffness: 300, damping: 25 }}
        />
        <p className="text-xs text-muted-foreground leading-relaxed max-w-[200px]">{description}</p>
      </div>

      {/* Bottom row */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-mono text-muted-foreground uppercase tracking-widest">
          {number}.
        </span>
        <span className="text-xs font-mono text-muted-foreground uppercase tracking-widest">
          AI
        </span>
      </div>

      {disabled && (
        <div className="absolute inset-x-0 bottom-3 flex justify-center">
          <span className="text-xs font-mono text-muted-foreground/60 uppercase tracking-widest">
            Coming soon
          </span>
        </div>
      )}
    </motion.div>
  )

  if (href && !disabled) return <Link href={href}>{inner}</Link>
  return <div>{inner}</div>
}

// ─── Mode Badge ────────────────────────────────────────────────────────────────
function ModeBadge({ icon, name, desc }: { icon: React.ReactNode; name: string; desc: string }) {
  return (
    <div className="group flex items-start gap-4 py-5 border-b border-border/30 hover:pl-2 transition-all duration-200">
      <div className="w-9 h-9 rounded-lg bg-secondary flex items-center justify-center text-muted-foreground group-hover:bg-primary/10 group-hover:text-primary transition-colors flex-shrink-0 mt-0.5">
        {icon}
      </div>
      <div>
        <p className="font-semibold text-sm mb-0.5">{name}</p>
        <p className="text-xs text-muted-foreground">{desc}</p>
      </div>
    </div>
  )
}

// ─── Main Component ────────────────────────────────────────────────────────────
export default function Home() {
  const { loading, isAuthenticated } = useAuth()
  useLenis()

  const heroRef = useRef<HTMLElement>(null)
  const { scrollYProgress: heroProgress } = useScroll({
    target: heroRef,
    offset: ['start start', 'end start'],
  })

  const glowTopY = useTransform(heroProgress, [0, 1], [0, -60])
  const glowBottomY = useTransform(heroProgress, [0, 1], [0, -40])
  const gridY = useTransform(heroProgress, [0, 1], [0, -20])
  const statCardsY = useTransform(heroProgress, [0, 1], [0, -80])

  // Heading parallax refs
  const missionRef = useRef<HTMLDivElement>(null)
  const { scrollYProgress: missionProgress } = useScroll({
    target: missionRef,
    offset: ['start end', 'end start'],
  })
  const missionHeadingX = useTransform(missionProgress, [0, 1], [15, -15])

  const statsRef = useRef<HTMLDivElement>(null)
  const { scrollYProgress: statsProgress } = useScroll({
    target: statsRef,
    offset: ['start end', 'end start'],
  })
  const statsHeadingX = useTransform(statsProgress, [0, 1], [20, -20])

  const ctaRef = useRef<HTMLDivElement>(null)
  const { scrollYProgress: ctaProgress } = useScroll({
    target: ctaRef,
    offset: ['start end', 'end start'],
  })
  const ctaHeadingX = useTransform(ctaProgress, [0, 1], [15, -15])

  const scrollToTop = () => window.scrollTo({ top: 0, behavior: 'smooth' })

  return (
    <div
      className="min-h-screen bg-background custom-cursor-page"
      style={{ fontFamily: 'Barlow, Inter, sans-serif' }}
    >
      <CustomCursor />

      {/* ── Header ──────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-50 bg-background/95 backdrop-blur-xl border-b border-border/40">
        <div className="container flex items-center justify-between h-16 px-6">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center shadow-md shadow-primary/30">
              <Activity className="w-4 h-4 text-black" />
            </div>
            <span
              className="font-black text-base tracking-tight uppercase"
              style={{ fontFamily: 'Barlow Condensed, Inter, sans-serif' }}
            >
              Football<span className="text-primary">AI</span>
            </span>
          </div>

          <nav className="flex items-center gap-1">
            {loading ? (
              <div className="w-24 h-9 bg-secondary animate-pulse rounded-lg" />
            ) : isAuthenticated ? (
              <>
                <Link href="/dashboard">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-muted-foreground hover:text-foreground"
                  >
                    Dashboard
                  </Button>
                </Link>
                <Link href="/upload">
                  <Button size="sm" className="ml-2 shadow-md shadow-primary/25 font-semibold">
                    <Upload className="w-3.5 h-3.5 mr-1.5" />
                    Upload
                  </Button>
                </Link>
              </>
            ) : (
              <a href={getLoginUrl()}>
                <Button size="sm" className="shadow-md shadow-primary/25 font-semibold">
                  Sign In
                </Button>
              </a>
            )}
          </nav>
        </div>
      </header>

      {/* ── Hero ────────────────────────────────────────────────────── */}
      <section
        ref={heroRef}
        className="relative min-h-[92vh] flex flex-col justify-center overflow-hidden pt-8 pb-0"
      >
        {/* Background layers */}
        <div className="absolute inset-0">
          {/* Pitch SVG watermark — fallback when canvas is not available */}
          <svg
            className="pitch-svg-fallback absolute inset-0 w-full h-full opacity-[0.13]"
            viewBox="0 0 105 68"
            preserveAspectRatio="xMidYMid meet"
          >
            <rect
              x="0"
              y="0"
              width="105"
              height="68"
              fill="none"
              stroke="white"
              strokeWidth="0.6"
            />
            <line x1="52.5" y1="0" x2="52.5" y2="68" stroke="white" strokeWidth="0.6" />
            <circle cx="52.5" cy="34" r="9.15" fill="none" stroke="white" strokeWidth="0.6" />
            <rect
              x="0"
              y="24.84"
              width="16.5"
              height="18.32"
              fill="none"
              stroke="white"
              strokeWidth="0.6"
            />
            <rect
              x="88.5"
              y="24.84"
              width="16.5"
              height="18.32"
              fill="none"
              stroke="white"
              strokeWidth="0.6"
            />
            <rect
              x="0"
              y="13.84"
              width="5.5"
              height="40.32"
              fill="none"
              stroke="white"
              strokeWidth="0.6"
            />
            <rect
              x="99.5"
              y="13.84"
              width="5.5"
              height="40.32"
              fill="none"
              stroke="white"
              strokeWidth="0.6"
            />
          </svg>
          {/* Interactive canvas pitch — hides the SVG fallback once initialized */}
          <PitchCanvas />
          {/* Green glow top left */}
          <motion.div
            style={{ y: glowTopY }}
            className="absolute -top-32 -left-32 w-[600px] h-[600px] bg-primary/20 rounded-full blur-[120px]"
          />
          {/* Subtle green glow bottom right */}
          <motion.div
            style={{ y: glowBottomY }}
            className="absolute bottom-0 right-0 w-[400px] h-[400px] bg-emerald-500/14 rounded-full blur-[100px]"
          />
          {/* Grid */}
          <motion.div
            style={{ y: gridY }}
            className="absolute inset-0 bg-[linear-gradient(rgba(34,197,94,0.07)_1px,transparent_1px),linear-gradient(90deg,rgba(34,197,94,0.07)_1px,transparent_1px)] bg-[size:80px_80px]"
          />
        </div>

        <div className="container relative z-10 px-6">
          {/* Label row */}
          <div className="animate-fade-up flex items-center gap-3 mb-8">
            <span className="text-xs font-mono text-primary tracking-widest uppercase">ST/00</span>
            <div className="w-px h-4 bg-border/50" />
            <span className="text-xs font-mono text-muted-foreground tracking-widest uppercase">
              AI-Powered Match Analysis
            </span>
          </div>

          {/* Main headline */}
          <div className="mb-8 overflow-hidden">
            <h1
              className="animate-fade-up delay-100 text-[clamp(3.5rem,10vw,9rem)] font-black leading-[0.92] tracking-tighter uppercase"
              style={{ fontFamily: 'Barlow Condensed, Inter, sans-serif' }}
            >
              Transform
            </h1>
          </div>
          <div className="overflow-hidden">
            <h1
              className="animate-fade-up delay-200 text-[clamp(3.5rem,10vw,9rem)] font-black leading-[0.92] tracking-tighter uppercase gradient-text-green"
              style={{ fontFamily: 'Barlow Condensed, Inter, sans-serif' }}
            >
              Football
            </h1>
          </div>
          <div className="overflow-hidden mb-10">
            <h1
              className="animate-fade-up delay-300 text-[clamp(3.5rem,10vw,9rem)] font-black leading-[0.92] tracking-tighter uppercase text-foreground/80"
              style={{ fontFamily: 'Barlow Condensed, Inter, sans-serif' }}
            >
              Into Intelligence.
            </h1>
          </div>

          {/* Subtitle + CTAs */}
          <div className="animate-fade-up delay-400 flex flex-col sm:flex-row items-start sm:items-center gap-6 mb-16">
            <p className="text-base md:text-lg text-muted-foreground max-w-md leading-relaxed">
              Upload match videos and get real-time player tracking, tactical heatmaps, event
              detection, and AI commentary.
            </p>
            <div className="flex items-center gap-3 flex-shrink-0">
              {isAuthenticated ? (
                <Link href="/upload">
                  <Button
                    size="lg"
                    className="gap-2 h-12 px-6 font-semibold shadow-lg shadow-primary/30"
                  >
                    Get Started <ArrowRight className="w-4 h-4" />
                  </Button>
                </Link>
              ) : (
                <a href={getLoginUrl()}>
                  <Button
                    size="lg"
                    className="gap-2 h-12 px-6 font-semibold shadow-lg shadow-primary/30"
                  >
                    Get Started <ArrowRight className="w-4 h-4" />
                  </Button>
                </a>
              )}
              <Link href="/dashboard">
                <Button
                  size="lg"
                  variant="outline"
                  className="gap-2 h-12 px-6 font-semibold border-border/50"
                >
                  <Play className="w-4 h-4" /> Demo
                </Button>
              </Link>
            </div>
          </div>

          {/* Bottom label row with divider — DecideAI style */}
          <div className="animate-fade-up delay-500">
            <div className="w-full h-px bg-border/40 mb-4" />
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-muted-foreground tracking-widest">
                ©2025:V.1
              </span>
              <div className="hidden md:flex items-center gap-8">
                {['01. DETECT', '02. TRACK', '03. ANALYZE'].map((label) => (
                  <span
                    key={label}
                    className="text-xs font-mono text-muted-foreground tracking-widest"
                  >
                    {label}
                  </span>
                ))}
              </div>
              <span className="text-xs font-mono text-muted-foreground tracking-widest flex items-center gap-1.5">
                Discover <ArrowRight className="w-3 h-3" />
              </span>
            </div>
          </div>
        </div>

        {/* Floating stat cards — right side on large screens */}
        <motion.div style={{ y: statCardsY }} className="hidden lg:block">
          <FloatingStatCard
            value="22+"
            label="Players Tracked"
            icon={<Users className="w-4 h-4" />}
            className="right-[8%] top-[22%] animate-scale-in delay-600"
          />
          <FloatingStatCard
            value="98.5%"
            label="Detection Accuracy"
            icon={<Target className="w-4 h-4" />}
            className="right-[14%] top-[42%] animate-scale-in delay-700"
          />
          <FloatingStatCard
            value="Real-time"
            label="AI Commentary"
            icon={<Sparkles className="w-4 h-4" />}
            className="right-[6%] top-[60%] animate-scale-in delay-800"
          />
        </motion.div>
      </section>

      {/* ── Parallax Divider ────────────────────────────────────────── */}
      <ParallaxDivider />

      {/* ── Our Mission ─────────────────────────────────────────────── */}
      <ParallaxSection
        className="py-20 px-6"
        bgRange={[-25, 25]}
        backgroundElement={
          <div className="absolute inset-0">
            <div className="absolute top-0 left-1/4 w-[500px] h-[500px] bg-primary/6 rounded-full blur-[140px]" />
            <div className="absolute inset-0 bg-[linear-gradient(rgba(34,197,94,0.04)_1px,transparent_1px),linear-gradient(90deg,rgba(34,197,94,0.04)_1px,transparent_1px)] bg-[size:60px_60px]" />
          </div>
        }
      >
        <div className="container" ref={missionRef}>
          <ScrollReveal variant="fadeUp">
            <motion.div style={{ x: missionHeadingX }}>
              <SectionHeader
                number="ST/01"
                label="Our Mission"
                title="Our Mission"
                description="Closed systems and generic models hold analysts back. We built a pipeline that is reliable, open, and affordable for every club."
              />
            </motion.div>
          </ScrollReveal>

          {/* Three product cards — DecideAI style */}
          <ScrollStagger className="grid md:grid-cols-3 gap-4" stagger={0.1}>
            <StaggerItem>
              <PipelineCard
                topLeft="Player"
                topRight="Detection"
                icon={<Target className="w-6 h-6" />}
                title="Player Detection"
                description="YOLOv8-powered detection with custom-trained models. Identify every player on the pitch with precision."
                number="001"
                href="/upload"
              />
            </StaggerItem>
            <StaggerItem>
              <PipelineCard
                topLeft="Tactical"
                topRight="Analysis"
                icon={<BarChart3 className="w-6 h-6" />}
                title="Tactical Analysis"
                description="Heatmaps, pass networks, Voronoi diagrams, and team formations — all generated automatically."
                number="002"
                href="/upload"
              />
            </StaggerItem>
            <StaggerItem>
              <PipelineCard
                topLeft="AI"
                topRight="Commentary"
                icon={<Sparkles className="w-6 h-6" />}
                title="AI Commentary"
                description="Grounded tactical analysis from tracking data. Let the AI narrate what the numbers mean."
                number="003"
                href="/upload"
              />
            </StaggerItem>
          </ScrollStagger>
        </div>
      </ParallaxSection>

      {/* ── Best-in-class / Stats ────────────────────────────────────── */}
      <ParallaxSection
        className="py-20 px-6 bg-card/20 border-y border-border/40"
        bgRange={[-30, 30]}
        backgroundElement={
          <div className="absolute inset-0">
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[400px] bg-emerald-500/5 rounded-full blur-[160px]" />
          </div>
        }
      >
        <div className="container" ref={statsRef}>
          <ScrollReveal variant="fadeUp">
            <motion.div style={{ x: statsHeadingX }}>
              <SectionHeader
                number="ST/02"
                label="By The Numbers"
                title="A Best-In-Class Solution"
                description="The optimal mix of computer vision, deep learning, and tactical intelligence."
              />
            </motion.div>
          </ScrollReveal>

          <ScrollStagger
            className="grid grid-cols-2 lg:grid-cols-4 gap-0 border border-border/40 rounded-xl overflow-hidden"
            stagger={0.08}
          >
            {[
              {
                value: '22+',
                label: 'Players Tracked Per Frame',
                icon: <Users className="w-5 h-5" />,
              },
              { value: '98.5%', label: 'Detection Accuracy', icon: <Target className="w-5 h-5" /> },
              {
                value: '7',
                label: 'Analysis Pipeline Modes',
                icon: <Layers className="w-5 h-5" />,
              },
              { value: '<5min', label: 'Processing Time', icon: <Zap className="w-5 h-5" /> },
            ].map((stat, i) => (
              <StaggerItem
                key={i}
                className={`p-8 flex flex-col justify-between ${i < 3 ? 'border-r border-border/40' : ''} ${i < 2 ? 'border-b border-border/40 lg:border-b-0' : ''}`}
              >
                <span className="text-primary mb-4">{stat.icon}</span>
                <div>
                  <div
                    className="text-[2.8rem] md:text-[3.5rem] font-black leading-none mb-2 gradient-text-green"
                    style={{ fontFamily: 'Barlow Condensed, Inter, sans-serif' }}
                  >
                    {stat.value}
                  </div>
                  <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">
                    {stat.label}
                  </p>
                </div>
              </StaggerItem>
            ))}
          </ScrollStagger>

          {/* Feature list — numbered like DecideAI's advantage list */}
          <ScrollStagger className="mt-12 grid md:grid-cols-3 gap-8" stagger={0.1}>
            {[
              {
                num: '001.',
                title: 'Cost-effective',
                desc: '10x cheaper than cloud APIs by using your own GPU-hosted models. No per-frame API charges.',
              },
              {
                num: '002.',
                title: 'Accurate',
                desc: 'Custom-trained YOLOv8 models combined with ByteTrack persistence for smooth, reliable tracking.',
              },
              {
                num: '003.',
                title: 'Open & Flexible',
                desc: 'Bring your own models or use the Roboflow API fallback. Plug-and-play architecture.',
              },
            ].map((item) => (
              <StaggerItem key={item.num} className="flex flex-col gap-3">
                <div className="flex items-end gap-3">
                  <span className="text-xs font-mono text-muted-foreground/60">{item.num}</span>
                  <h3
                    className="text-2xl font-black uppercase tracking-tight"
                    style={{ fontFamily: 'Barlow Condensed, Inter, sans-serif' }}
                  >
                    {item.title}
                  </h3>
                </div>
                <div className="w-full h-px bg-border/40" />
                <p className="text-sm text-muted-foreground leading-relaxed">{item.desc}</p>
              </StaggerItem>
            ))}
          </ScrollStagger>

          {/* Animated stat counters */}
          <ScrollReveal variant="fadeUp" delay={0.15}>
            <div className="mt-16 flex flex-wrap justify-center gap-8">
              <DotCard target={22} duration={1500} label="Players Per Frame" />
              <DotCard target={985} duration={1800} label="Detection Accuracy ‰" />
              <DotCard target={5} duration={1200} label="Pipeline Modes" />
            </div>
          </ScrollReveal>
        </div>
      </ParallaxSection>

      {/* ── Capabilities Grid ───────────────────────────────────────── */}
      <ParallaxSection
        className="py-20 px-6"
        bgRange={[-20, 20]}
        backgroundElement={
          <div className="absolute inset-0 bg-[linear-gradient(rgba(34,197,94,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(34,197,94,0.03)_1px,transparent_1px)] bg-[size:40px_40px]" />
        }
      >
        <div className="container">
          <ScrollReveal variant="fadeUp">
            <SectionHeader
              number="ST/02b"
              label="Capabilities"
              title="What We Track"
              description="Six core intelligence modules — each powered by deep learning models trained on professional football footage."
            />
          </ScrollReveal>

          <ScrollStagger
            className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 divide-x divide-y divide-dashed border border-dashed border-border/60"
            stagger={0.06}
          >
            {[
              {
                title: 'Player Detection',
                icon: Target,
                description:
                  'YOLOv8-powered bounding boxes with confidence scores on every player in frame.',
              },
              {
                title: 'Tactical Heatmaps',
                icon: Flame,
                description:
                  'Spatial density maps showing player influence zones across the full match duration.',
              },
              {
                title: 'Pass Networks',
                icon: GitBranch,
                description:
                  'Graph-based passing connections and team formations derived from tracking data.',
              },
              {
                title: 'Team Classification',
                icon: Users,
                description:
                  'SigLIP + UMAP clustering to assign players to teams without manual labelling.',
              },
              {
                title: 'Ball Tracking',
                icon: Circle,
                description:
                  'SAHI-enhanced detection with interpolation for sub-frame precision on high-speed play.',
              },
              {
                title: 'AI Commentary',
                icon: Sparkles,
                description:
                  'Grounded natural language analysis narrating what the tactical numbers actually mean.',
              },
            ].map((feature, i) => (
              <StaggerItem key={i}>
                <FeatureCard
                  feature={feature}
                  className="h-full min-h-[180px] hover:bg-primary/5 transition-colors duration-300"
                />
              </StaggerItem>
            ))}
          </ScrollStagger>
        </div>
      </ParallaxSection>

      {/* ── Editorial Quote ──────────────────────────────────────────── */}
      <ParallaxSection
        className="py-24 px-6 border-b border-border/40"
        bgRange={[-35, 35]}
        backgroundElement={
          <div className="absolute inset-0">
            <div className="absolute bottom-0 right-1/4 w-[600px] h-[400px] bg-primary/7 rounded-full blur-[160px]" />
          </div>
        }
      >
        <div className="container">
          <ScrollReveal variant="fadeUp">
            <div className="max-w-4xl">
              <p className="text-xs font-mono text-primary tracking-widest uppercase mb-6">
                ST/03 — Intelligence
              </p>
              <blockquote
                className="text-[clamp(2rem,5vw,4rem)] font-black leading-tight tracking-tight uppercase mb-8"
                style={{ fontFamily: 'Barlow Condensed, Inter, sans-serif' }}
              >
                Data is more than numbers;{' '}
                <span className="gradient-text-green">it is intelligence</span> in action.
              </blockquote>
              <p className="text-muted-foreground text-base max-w-lg leading-relaxed">
                At FootballAI, we combine intelligent analytics and motion tracking to help coaches
                and analysts reach their true tactical potential.
              </p>
            </div>
          </ScrollReveal>
        </div>
      </ParallaxSection>

      {/* ── Pipeline Modes ──────────────────────────────────────────── */}
      <ParallaxSection
        className="py-20 px-6"
        bgRange={[-20, 20]}
        backgroundElement={
          <div className="absolute inset-0 bg-[linear-gradient(rgba(34,197,94,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(34,197,94,0.03)_1px,transparent_1px)] bg-[size:50px_50px]" />
        }
      >
        <div className="container">
          <ScrollReveal variant="fadeUp">
            <SectionHeader
              number="ST/04"
              label="Pipeline Modes"
              title="Flexible Analysis Options"
              description="Run the full pipeline or select specific components based on your needs and available hardware."
            />
          </ScrollReveal>

          <ScrollStagger className="flex flex-wrap justify-center gap-8 mt-4" stagger={0.1}>
            {[
              {
                id: 'full',
                topInscription: 'Mode 01',
                mainText: 'Full Analysis',
                subMainText: 'Complete pipeline with all features enabled',
                progressBarInscription: 'Pipeline coverage:',
                progress: 100,
                progressValue: '100%',
                accentColor: '#22c55e',
                filledButtonInscription: 'Start',
                outlinedButtonInscription: 'Details',
              },
              {
                id: 'radar',
                topInscription: 'Mode 02',
                mainText: 'Radar View',
                subMainText: "Bird's-eye 2D pitch visualization of player positions",
                progressBarInscription: 'Processing load:',
                progress: 45,
                progressValue: 'Light',
                accentColor: '#22c55e',
                filledButtonInscription: 'Start',
                outlinedButtonInscription: 'Details',
              },
              {
                id: 'team',
                topInscription: 'Mode 03',
                mainText: 'Team Analysis',
                subMainText: 'SigLIP + UMAP clustering for automatic team classification',
                progressBarInscription: 'ML intensity:',
                progress: 75,
                progressValue: 'High',
                accentColor: '#22c55e',
                filledButtonInscription: 'Start',
                outlinedButtonInscription: 'Details',
              },
            ].map((mode) => (
              <StaggerItem key={mode.id} className="w-[20rem] flex-none">
                <BauhausCard
                  {...mode}
                  onFilledButtonClick={() => {}}
                  onOutlinedButtonClick={() => {}}
                  onMoreOptionsClick={() => {}}
                  backgroundColor="var(--bauhaus-card-bg)"
                  separatorColor="var(--bauhaus-card-separator)"
                  borderRadius="0.75rem"
                  borderWidth="1.5px"
                  chronicleButtonBg="var(--bauhaus-chronicle-bg)"
                  chronicleButtonFg="var(--bauhaus-chronicle-fg)"
                  chronicleButtonHoverFg="#000"
                />
              </StaggerItem>
            ))}
          </ScrollStagger>
        </div>
      </ParallaxSection>

      {/* ── Tech Stack Comparison ───────────────────────────────────── */}
      <ParallaxSection
        className="py-20 px-6 bg-card/20 border-y border-border/40"
        bgRange={[-25, 25]}
        backgroundElement={
          <div className="absolute inset-0">
            <div className="absolute bottom-0 right-0 w-[400px] h-[400px] bg-primary/5 rounded-full blur-[130px]" />
          </div>
        }
      >
        <div className="container">
          <ScrollReveal variant="fadeUp">
            <SectionHeader
              number="ST/05"
              label="Technology"
              title="Powered By"
              description="Best-in-class open-source computer vision and deep learning components."
            />
          </ScrollReveal>

          <ScrollReveal variant="fadeUp" delay={0.1}>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border/40">
                    <th className="text-left py-4 pr-8 text-xs font-mono text-muted-foreground uppercase tracking-widest w-[200px]" />
                    {['Open Source', 'GPU Optimized', 'FootballAI'].map((h) => (
                      <th
                        key={h}
                        className="text-center py-4 px-4 text-xs font-mono text-muted-foreground uppercase tracking-widest"
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[
                    { name: 'High-quality', cols: [false, false, true] },
                    { name: 'Real-time tracking', cols: [false, true, true] },
                    { name: 'Custom models', cols: [false, false, true] },
                    { name: 'No API costs', cols: [true, false, true] },
                    { name: 'Tactical AI', cols: [false, false, true] },
                  ].map((row) => (
                    <tr
                      key={row.name}
                      className="border-b border-border/20 hover:bg-card/30 transition-colors"
                    >
                      <td className="py-4 pr-8 text-sm font-medium uppercase tracking-wide text-muted-foreground">
                        {row.name}
                      </td>
                      {row.cols.map((val, ci) => (
                        <td key={ci} className="text-center py-4 px-4">
                          {val ? (
                            <CheckCircle2 className="w-5 h-5 text-primary mx-auto" />
                          ) : (
                            <div className="w-5 h-5 rounded-full border border-border/40 mx-auto" />
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </ScrollReveal>

          {/* Tech logos row */}
          <ScrollStagger
            className="flex flex-wrap items-center gap-6 mt-12 pt-8 border-t border-border/40"
            stagger={0.08}
          >
            <span className="text-xs font-mono text-muted-foreground/60 uppercase tracking-widest">
              Built with
            </span>
            {[
              { icon: <Cpu className="w-4 h-4" />, name: 'YOLOv8' },
              { icon: <GitBranch className="w-4 h-4" />, name: 'ByteTrack' },
              { icon: <Eye className="w-4 h-4" />, name: 'SigLIP' },
              { icon: <Sparkles className="w-4 h-4" />, name: 'Custom Models' },
            ].map((tech) => (
              <StaggerItem
                key={tech.name}
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-secondary/60 text-sm font-medium"
              >
                <span className="text-muted-foreground">{tech.icon}</span>
                {tech.name}
              </StaggerItem>
            ))}
          </ScrollStagger>
        </div>
      </ParallaxSection>

      {/* ── CTA Section ─────────────────────────────────────────────── */}
      <ParallaxSection
        className="py-28 px-6"
        bgRange={[-30, 30]}
        backgroundElement={
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_60%_60%_at_50%_50%,rgba(34,197,94,0.1),transparent)]" />
        }
      >
        <div className="container text-center" ref={ctaRef}>
          <ScrollReveal variant="scaleIn">
            <p className="text-xs font-mono text-primary tracking-widest uppercase mb-6">
              ST/06 — Ready
            </p>
            <motion.h2
              className="text-[clamp(2.5rem,8vw,7rem)] font-black leading-none tracking-tighter uppercase mb-8"
              style={{ fontFamily: 'Barlow Condensed, Inter, sans-serif', x: ctaHeadingX }}
            >
              Ready to Analyze
              <br />
              <span className="gradient-text-green">Your Matches?</span>
            </motion.h2>
            <p className="text-muted-foreground text-lg mb-10 max-w-md mx-auto leading-relaxed">
              Upload your first video and see AI-powered tactical analysis in action.
            </p>
            {isAuthenticated ? (
              <Link href="/upload">
                <Button
                  size="lg"
                  className="gap-2 h-14 px-10 text-lg font-semibold shadow-xl shadow-primary/30"
                >
                  <Upload className="w-5 h-5" /> Upload Video
                </Button>
              </Link>
            ) : (
              <a href={getLoginUrl()}>
                <Button
                  size="lg"
                  className="gap-2 h-14 px-10 text-lg font-semibold shadow-xl shadow-primary/30"
                >
                  Sign In to Get Started <ArrowRight className="w-5 h-5" />
                </Button>
              </a>
            )}
          </ScrollReveal>
        </div>
      </ParallaxSection>

      {/* ── Footer ──────────────────────────────────────────────────── */}
      <footer className="border-t border-border/40 bg-card/20">
        <div className="container px-6 py-12">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-8">
            {/* Brand */}
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2.5">
                <div className="w-9 h-9 rounded-xl bg-primary flex items-center justify-center shadow-md shadow-primary/30">
                  <Activity className="w-5 h-5 text-black" />
                </div>
                <span
                  className="font-black text-lg tracking-tight uppercase"
                  style={{ fontFamily: 'Barlow Condensed, Inter, sans-serif' }}
                >
                  Football<span className="text-primary">AI</span>
                </span>
              </div>
              <p className="text-xs text-muted-foreground max-w-xs leading-relaxed mt-1">
                AI-powered football match analysis. Real-time player tracking, tactical heatmaps,
                and event detection.
              </p>
            </div>

            {/* Links */}
            <div className="flex items-start gap-16">
              <ul className="flex flex-col gap-3">
                {[
                  { label: 'Dashboard', href: '/dashboard' },
                  { label: 'Upload', href: '/upload' },
                ].map((link) => (
                  <li key={link.label}>
                    <Link
                      href={link.href}
                      className="text-sm text-muted-foreground hover:text-foreground transition-colors"
                    >
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
              <ul className="flex flex-col gap-3 text-right">
                {[
                  { label: 'YOLOv8', href: '#' },
                  { label: 'ByteTrack', href: '#' },
                  { label: 'SigLIP', href: '#' },
                ].map((link) => (
                  <li key={link.label}>
                    <a
                      href={link.href}
                      className="text-sm text-muted-foreground hover:text-foreground transition-colors"
                    >
                      {link.label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="flex items-center justify-between mt-10 pt-6 border-t border-border/30">
            <span className="text-xs text-muted-foreground/60">©2025 FootballAI</span>
            <button
              onClick={scrollToTop}
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              Back to top <ArrowUp className="w-3 h-3" />
            </button>
          </div>
        </div>
      </footer>
    </div>
  )
}
