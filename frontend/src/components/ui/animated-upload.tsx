import { motion, useReducedMotion, type Easing, type Variants } from 'framer-motion'
import { ChevronUp } from 'lucide-react'
import { cn } from '@/lib/utils'
import { HyperText } from '@/components/ui/hyper-text'

interface AnimatedUploadProps {
  className?: string
  progress: number
  stage: 'uploading' | 'processing' | 'done'
  speed: number
  timeRemaining: number
  bytesTransferred: number
  totalBytes: number
}

const STAGE_LABELS = {
  uploading: 'UPLOADING',
  processing: 'PROCESSING',
  done: 'COMPLETE',
} as const

const STAGES = ['uploading', 'processing', 'done'] as const
const STAGE_DISPLAY = ['UPLOAD', 'PROCESS', 'DONE'] as const

export function AnimatedUpload({
  className,
  progress,
  stage,
  speed,
  timeRemaining,
  bytesTransferred,
  totalBytes,
}: AnimatedUploadProps) {
  const shouldReduceMotion = useReducedMotion()
  const easing: Easing = shouldReduceMotion ? 'linear' : 'easeOut'
  const isMoving = stage === 'uploading' || stage === 'processing'
  const currentStageIdx = STAGES.indexOf(stage)

  const formatSpeed = (bps: number) => {
    if (bps > 1024 * 1024) return `${(bps / (1024 * 1024)).toFixed(1)} MB/s`
    if (bps > 0) return `${(bps / 1024).toFixed(0)} KB/s`
    return '—'
  }

  const formatTime = (seconds: number) => {
    if (seconds <= 0) return '—'
    const m = Math.floor(seconds / 60)
    const s = seconds % 60
    return m > 0 ? `${m}m ${s.toString().padStart(2, '0')}s` : `${s}s`
  }

  const formatBytes = (bytes: number) => `${(bytes / (1024 * 1024)).toFixed(1)}`

  const containerVariants: Variants = {
    hidden: { opacity: 0, y: shouldReduceMotion ? 0 : 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.3, ease: easing },
    },
  }

  const chevronVariants: Variants = {
    idle: { y: 0, opacity: 0.7 },
    animating: {
      y: shouldReduceMotion ? 0 : [0, -8, 0],
      opacity: [0.7, 1, 0.7],
      transition: {
        duration: 1.5,
        ease: 'easeInOut' as Easing,
        repeat: Infinity,
        repeatType: 'loop',
      },
    },
  }

  const chevron2Variants: Variants = {
    idle: { y: -14, opacity: 0.5 },
    animating: {
      y: shouldReduceMotion ? -8 : [-14, -18, -14],
      opacity: [0.5, 1, 0.5],
      transition: {
        duration: 1.5,
        ease: 'easeInOut' as Easing,
        repeat: Infinity,
        repeatType: 'loop',
        delay: 0.3,
      },
    },
  }

  return (
    <motion.div
      className={cn('w-full max-w-lg', className)}
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* Header row */}
      <div className="flex items-center mb-2">
        {/* Animated chevrons (pointing up for upload) */}
        <div className="flex -mt-3 flex-col items-center justify-center w-8 h-16 overflow-hidden relative">
          <motion.div
            className="absolute"
            variants={chevronVariants}
            animate={isMoving ? 'animating' : 'idle'}
          >
            <ChevronUp size={24} className="text-primary" />
          </motion.div>
          <motion.div
            className="absolute"
            variants={chevron2Variants}
            animate={isMoving ? 'animating' : 'idle'}
          >
            <ChevronUp size={24} className="text-primary" />
          </motion.div>
        </div>

        {/* Status banner with geometric SVG background */}
        <div className="relative ml-2 flex-1 max-w-xs">
          <svg
            height="32"
            viewBox="0 0 107 15"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            className="absolute top-1/2 left-0 -translate-y-1/2 fill-foreground"
            style={{ width: '65%' }}
            preserveAspectRatio="none"
          >
            <path d="M0.445312 0.5H106.103V8.017L99.2813 14.838H0.445312V0.5Z" />
          </svg>
          <div className="relative px-4 py-1.5">
            <HyperText
              text={STAGE_LABELS[stage]}
              className="text-sm font-bold text-white dark:text-black"
            />
          </div>
        </div>
      </div>

      {/* Separator */}
      <div className="w-full h-px bg-foreground mb-3" />

      {/* Labels row */}
      <div className="flex items-center mb-1.5">
        <div className="w-32 text-[10px] font-mono text-muted-foreground">PROGRESS</div>
        <div className="w-24 text-[10px] font-mono text-muted-foreground ml-6">EST. TIME</div>
        <div className="text-[10px] font-mono text-muted-foreground">TRANSFERRED</div>
      </div>

      {/* Values row */}
      <div className="flex items-center">
        {/* Progress bar */}
        <div className="w-32">
          <div className="w-full h-2.5 border border-foreground bg-transparent rounded-full flex items-center px-0.5">
            <motion.div
              className="h-1 bg-primary rounded-full"
              initial={{ width: '0%' }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: shouldReduceMotion ? 0.1 : 0.4, ease: easing }}
            />
          </div>
        </div>

        {/* Est. time */}
        <div className="w-24 ml-6 text-sm font-mono">
          {stage === 'uploading' ? formatTime(timeRemaining) : '—'}
        </div>

        {/* Transferred */}
        <div className="text-sm font-mono">
          {totalBytes > 0 && bytesTransferred > 0
            ? `${formatBytes(bytesTransferred)} / ${formatBytes(totalBytes)} MB`
            : stage === 'uploading' && speed > 0
              ? formatSpeed(speed)
              : '—'}
        </div>
      </div>

      {/* Stage dots */}
      <div className="flex items-center mt-5">
        {STAGES.map((s, i) => (
          <div key={s} className="flex items-center">
            <div
              className={cn(
                'w-2.5 h-2.5 rounded-full transition-colors duration-300',
                currentStageIdx >= i ? 'bg-primary' : 'bg-muted'
              )}
            />
            {i < STAGES.length - 1 && (
              <div
                className={cn(
                  'w-20 h-px mx-1 transition-colors duration-300',
                  currentStageIdx > i ? 'bg-primary' : 'bg-border'
                )}
              />
            )}
          </div>
        ))}
      </div>

      {/* Stage labels */}
      <div className="flex items-center mt-1 gap-[72px]">
        {STAGE_DISPLAY.map((label) => (
          <span key={label} className="text-[10px] font-mono text-muted-foreground">
            {label}
          </span>
        ))}
      </div>

      {/* Bottom accent */}
      <div className="w-3/4 h-0.5 bg-primary mt-4 rounded-full" />
    </motion.div>
  )
}
