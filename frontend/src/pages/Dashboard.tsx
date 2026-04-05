import { useState, useEffect } from 'react'
import { useAuth } from '@/hooks/useAuth'
import { videosApi, analysisApi } from '@/lib/api-local'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { toast } from 'sonner'
import { Link, useLocation } from 'wouter'
import { getLoginUrl } from '@/const'
import { motion, useReducedMotion } from 'framer-motion'
import { cn } from '@/lib/utils'
import {
  Activity,
  Upload,
  FileVideo,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  Trash2,
  BarChart3,
  ArrowRight,
  Zap,
} from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'
import { PIPELINE_MODES, PipelineMode } from '@/types'

// ── Animated counter hook ──────────────────────────────────────────────
function useAnimatedCount(target: number, duration = 1200) {
  const [count, setCount] = useState(0)
  const shouldReduceMotion = useReducedMotion()

  useEffect(() => {
    if (shouldReduceMotion || target === 0) {
      setCount(target)
      return
    }
    const start = performance.now()
    let raf: number
    const step = (now: number) => {
      const progress = Math.min((now - start) / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3)
      setCount(Math.round(eased * target))
      if (progress < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [target, duration, shouldReduceMotion])

  return count
}

// ── Motion presets ─────────────────────────────────────────────────────
const EASE = [0.16, 1, 0.3, 1] as const

const staggerContainer = {
  hidden: {},
  visible: {
    transition: { staggerChildren: 0.08, delayChildren: 0.1 },
  },
}

const fadeUp = {
  hidden: { opacity: 0, y: 24, filter: 'blur(4px)' },
  visible: {
    opacity: 1,
    y: 0,
    filter: 'blur(0px)',
    transition: { duration: 0.6, ease: EASE },
  },
}

const scaleIn = {
  hidden: { opacity: 0, scale: 0.95 },
  visible: {
    opacity: 1,
    scale: 1,
    transition: { duration: 0.5, ease: EASE },
  },
}

// ── Dashboard ──────────────────────────────────────────────────────────
export default function Dashboard() {
  const { user, loading: authLoading, isAuthenticated } = useAuth()
  const [, navigate] = useLocation()
  const queryClient = useQueryClient()

  const { data: videos, isLoading: videosLoading } = useQuery({
    queryKey: ['videos'],
    queryFn: () => videosApi.list(),
    enabled: isAuthenticated,
  })

  const { data: analyses, isLoading: analysesLoading } = useQuery({
    queryKey: ['analyses'],
    queryFn: () => analysisApi.list(),
    enabled: isAuthenticated,
  })

  const deleteVideoMutation = useMutation({
    mutationFn: (videoId: number) => videosApi.delete(videoId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['videos'] })
      queryClient.invalidateQueries({ queryKey: ['analyses'] })
      toast.success('Video deleted successfully')
    },
    onError: () => {
      toast.error('Failed to delete video')
    },
  })

  const handleDeleteVideo = async (videoId: number) => {
    if (!confirm('Are you sure you want to delete this video?')) return
    deleteVideoMutation.mutate(videoId)
  }

  if (authLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    )
  }

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardHeader className="text-center">
            <CardTitle>Sign In Required</CardTitle>
            <CardDescription>Please sign in to view your dashboard</CardDescription>
          </CardHeader>
          <CardContent>
            <a href={getLoginUrl()}>
              <Button className="w-full">Sign In</Button>
            </a>
          </CardContent>
        </Card>
      </div>
    )
  }

  const recentAnalyses = analyses?.slice(0, 5) || []
  const processingAnalyses = analyses?.filter((a: any) => a.status === 'processing') || []
  const completedCount = analyses?.filter((a: any) => a.status === 'completed').length || 0

  return (
    <div className="min-h-screen bg-background">
      {/* ── Header ── */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container flex items-center justify-between h-14">
          <div className="flex items-center gap-2">
            <Link href="/">
              <div className="flex items-center gap-2 cursor-pointer">
                <div className="w-7 h-7 rounded-md bg-primary flex items-center justify-center">
                  <Activity className="w-4 h-4 text-primary-foreground" />
                </div>
                <span className="font-mono font-semibold text-sm tracking-widest">
                  FOOTBALL ANALYTICS
                </span>
              </div>
            </Link>
          </div>
          <nav className="flex items-center gap-4">
            <span className="text-xs font-mono text-muted-foreground hidden sm:block">
              {user?.name || 'User'}
            </span>
            <Link href="/upload">
              <Button size="sm" className="gap-2 font-mono tracking-wide">
                <Upload className="w-3.5 h-3.5" />
                UPLOAD
              </Button>
            </Link>
          </nav>
        </div>
      </header>

      <main className="container py-8 space-y-8">
        {/* ── Stats Grid ── */}
        <motion.div
          className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4"
          variants={staggerContainer}
          initial="hidden"
          animate="visible"
        >
          <StatCard
            title="TOTAL VIDEOS"
            value={videos?.length || 0}
            icon={<FileVideo className="w-5 h-5" />}
            color="text-primary"
          />
          <StatCard
            title="TOTAL ANALYSES"
            value={analyses?.length || 0}
            icon={<BarChart3 className="w-5 h-5" />}
            color="text-blue-400"
          />
          <StatCard
            title="PROCESSING"
            value={processingAnalyses.length}
            icon={
              processingAnalyses.length > 0 ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Zap className="w-5 h-5" />
              )
            }
            color="text-amber-400"
          />
          <StatCard
            title="COMPLETED"
            value={completedCount}
            icon={<CheckCircle2 className="w-5 h-5" />}
            color="text-emerald-400"
          />
        </motion.div>

        {/* ── Main Grid ── */}
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Recent Analyses */}
          <motion.div
            className="lg:col-span-2"
            variants={scaleIn}
            initial="hidden"
            animate="visible"
          >
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="font-mono text-xs tracking-widest text-muted-foreground">
                    RECENT ANALYSES
                  </h2>
                  <p className="text-sm text-muted-foreground mt-0.5">
                    Your latest video analysis jobs
                  </p>
                </div>
              </div>

              {analysesLoading ? (
                <div className="space-y-3">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-20 w-full rounded-xl" />
                  ))}
                </div>
              ) : recentAnalyses.length === 0 ? (
                <motion.div
                  className="text-center py-16 border border-dashed border-border rounded-xl"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.3 }}
                >
                  <BarChart3 className="w-10 h-10 mx-auto text-muted-foreground mb-3" />
                  <p className="text-muted-foreground mb-4 font-mono text-sm">No analyses yet</p>
                  <Link href="/upload">
                    <Button size="sm" className="gap-2 font-mono tracking-wide">
                      <Upload className="w-3.5 h-3.5" /> UPLOAD FIRST VIDEO
                    </Button>
                  </Link>
                </motion.div>
              ) : (
                <motion.div
                  className="space-y-2"
                  variants={staggerContainer}
                  initial="hidden"
                  animate="visible"
                >
                  {recentAnalyses.map((analysis: any) => (
                    <AnalysisCard key={analysis.id} analysis={analysis} />
                  ))}
                </motion.div>
              )}
            </div>
          </motion.div>

          {/* Videos */}
          <motion.div variants={scaleIn} initial="hidden" animate="visible">
            <div className="space-y-3">
              <div>
                <h2 className="font-mono text-xs tracking-widest text-muted-foreground">
                  YOUR VIDEOS
                </h2>
                <p className="text-sm text-muted-foreground mt-0.5">Uploaded match footage</p>
              </div>

              {videosLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-16 w-full rounded-xl" />
                  ))}
                </div>
              ) : videos?.length === 0 ? (
                <motion.div
                  className="text-center py-12 border border-dashed border-border rounded-xl"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.4 }}
                >
                  <FileVideo className="w-8 h-8 mx-auto text-muted-foreground mb-2" />
                  <p className="text-sm text-muted-foreground font-mono">No videos uploaded</p>
                </motion.div>
              ) : (
                <motion.div
                  className="space-y-2"
                  variants={staggerContainer}
                  initial="hidden"
                  animate="visible"
                >
                  {videos?.slice(0, 6).map((video: any) => (
                    <VideoCard
                      key={video.id}
                      video={video}
                      onDelete={() => handleDeleteVideo(video.id)}
                    />
                  ))}
                </motion.div>
              )}
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  )
}

// ── StatCard ───────────────────────────────────────────────────────────
function StatCard({
  title,
  value,
  icon,
  color,
}: {
  title: string
  value: number
  icon: React.ReactNode
  color: string
}) {
  const animatedValue = useAnimatedCount(value)

  return (
    <motion.div variants={fadeUp}>
      <motion.div
        whileHover={{ y: -2, transition: { duration: 0.2 } }}
        className="group relative overflow-hidden rounded-xl border border-border bg-card p-5 transition-colors hover:border-primary/40"
      >
        {/* Subtle gradient glow on hover */}
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

        <div className="relative flex items-center justify-between">
          <div>
            <p className="font-mono text-[10px] tracking-widest text-muted-foreground">{title}</p>
            <p className="text-4xl font-bold font-mono mt-1">{animatedValue}</p>
          </div>
          <div
            className={cn(
              'w-10 h-10 rounded-lg flex items-center justify-center bg-card border border-border transition-colors group-hover:border-primary/30',
              color
            )}
          >
            {icon}
          </div>
        </div>

        {/* Bottom accent line */}
        <div className="absolute bottom-0 left-0 h-px w-0 bg-primary group-hover:w-full transition-all duration-500" />
      </motion.div>
    </motion.div>
  )
}

// ── AnalysisCard ───────────────────────────────────────────────────────
function AnalysisCard({ analysis }: { analysis: any }) {
  const statusConfig = {
    pending: {
      icon: <Clock className="w-4 h-4" />,
      color: 'text-amber-400',
      bg: 'bg-amber-400/10',
      border: 'border-l-amber-400/50',
    },
    uploading: {
      icon: <Loader2 className="w-4 h-4 animate-spin" />,
      color: 'text-blue-400',
      bg: 'bg-blue-400/10',
      border: 'border-l-blue-400/50',
    },
    processing: {
      icon: <Loader2 className="w-4 h-4 animate-spin" />,
      color: 'text-blue-400',
      bg: 'bg-blue-400/10',
      border: 'border-l-blue-400/50',
    },
    completed: {
      icon: <CheckCircle2 className="w-4 h-4" />,
      color: 'text-emerald-400',
      bg: 'bg-emerald-400/10',
      border: 'border-l-emerald-400/50',
    },
    failed: {
      icon: <XCircle className="w-4 h-4" />,
      color: 'text-red-400',
      bg: 'bg-red-400/10',
      border: 'border-l-red-400/50',
    },
  }

  const status = statusConfig[analysis.status as keyof typeof statusConfig] || statusConfig.pending
  const mode = PIPELINE_MODES[analysis.mode as PipelineMode]

  return (
    <motion.div variants={fadeUp}>
      <Link href={`/analysis/${analysis.id}`}>
        <motion.div
          whileHover={{ x: 4, transition: { duration: 0.15 } }}
          whileTap={{ scale: 0.995 }}
          className={cn(
            'group flex items-center justify-between p-4 rounded-xl border border-border bg-card cursor-pointer transition-colors hover:border-primary/40',
            'border-l-2',
            status.border
          )}
        >
          <div className="flex items-center gap-3 min-w-0">
            <div
              className={cn(
                'w-9 h-9 rounded-lg flex items-center justify-center shrink-0',
                status.bg,
                status.color
              )}
            >
              {status.icon}
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-mono text-sm font-semibold">#{analysis.id}</span>
                <Badge variant="outline" className="text-[10px] font-mono h-4 px-1.5 tracking-wide">
                  {mode?.name || analysis.mode}
                </Badge>
                {analysis.status === 'processing' && (
                  <span className="text-[10px] font-mono text-blue-400">{analysis.progress}%</span>
                )}
              </div>
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground font-mono mt-0.5">
                <span className={cn('capitalize', status.color)}>{analysis.status}</span>
                <span className="text-border">|</span>
                <span>
                  {formatDistanceToNow(new Date(analysis.createdAt), { addSuffix: true })}
                </span>
              </div>
            </div>
          </div>

          <ArrowRight className="w-4 h-4 text-muted-foreground opacity-0 group-hover:opacity-100 group-hover:translate-x-0.5 transition-all duration-200 shrink-0" />
        </motion.div>
      </Link>
    </motion.div>
  )
}

// ── VideoCard ──────────────────────────────────────────────────────────
function VideoCard({ video, onDelete }: { video: any; onDelete: () => void }) {
  return (
    <motion.div variants={fadeUp}>
      <motion.div
        whileHover={{ x: 2, transition: { duration: 0.15 } }}
        className="group flex items-center justify-between p-3 rounded-xl border border-border bg-card transition-colors hover:border-primary/40"
      >
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
            <FileVideo className="w-4 h-4 text-primary" />
          </div>
          <div className="min-w-0">
            <p className="font-mono text-sm font-medium truncate">{video.title}</p>
            <p className="text-[10px] text-muted-foreground font-mono">
              {formatDistanceToNow(new Date(video.createdAt), { addSuffix: true })}
            </p>
          </div>
        </div>
        <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}>
          <Button
            variant="ghost"
            size="icon"
            onClick={(e) => {
              e.preventDefault()
              onDelete()
            }}
            className="shrink-0 opacity-0 group-hover:opacity-100 transition-opacity h-8 w-8"
          >
            <Trash2 className="w-3.5 h-3.5 text-muted-foreground hover:text-destructive transition-colors" />
          </Button>
        </motion.div>
      </motion.div>
    </motion.div>
  )
}
