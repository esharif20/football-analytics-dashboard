import { useEffect, useRef, useState, useCallback } from 'react'
import { Slider } from '@/components/ui/slider'
import { useTeamColors } from '../context'
import { Play, Pause, SkipBack, SkipForward } from 'lucide-react'

interface FrameScrubberProps {
  currentFrame: number
  minFrame: number
  maxFrame: number
  fps?: number
  onFrameChange: (frame: number) => void
  events?: Array<{ frameNumber: number; type: string; teamId?: number | null }>
}

const SPEEDS = [0.5, 1, 2, 4]

export function FrameScrubber({
  currentFrame,
  minFrame,
  maxFrame,
  fps = 25,
  onFrameChange,
  events = [],
}: FrameScrubberProps) {
  const { TEAM1_HEX, TEAM2_HEX } = useTeamColors()
  const [isPlaying, setIsPlaying] = useState(false)
  const [speedIdx, setSpeedIdx] = useState(1) // 1x default
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const frameRef = useRef(currentFrame)
  frameRef.current = currentFrame

  const range = maxFrame - minFrame || 1
  const currentTime = currentFrame / fps
  const duration = maxFrame / fps
  const speed = SPEEDS[speedIdx]

  const stop = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    setIsPlaying(false)
  }, [])

  const play = useCallback(() => {
    stop()
    setIsPlaying(true)
    intervalRef.current = setInterval(
      () => {
        const next = frameRef.current + 1
        if (next > maxFrame) {
          onFrameChange(minFrame)
          clearInterval(intervalRef.current!)
          setIsPlaying(false)
        } else {
          onFrameChange(next)
        }
      },
      1000 / (fps * speed)
    )
  }, [stop, maxFrame, minFrame, fps, speed, onFrameChange])

  // Restart interval when speed changes while playing
  useEffect(() => {
    if (isPlaying) play()
  }, [speedIdx]) // eslint-disable-line react-hooks/exhaustive-deps

  // Stop on unmount
  useEffect(() => () => stop(), [stop])

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Only if no input is focused
      if (
        document.activeElement?.tagName === 'INPUT' ||
        document.activeElement?.tagName === 'TEXTAREA'
      )
        return
      if (e.code === 'Space') {
        e.preventDefault()
        if (isPlaying) stop()
        else play()
      } else if (e.code === 'ArrowRight') {
        e.preventDefault()
        stop()
        const step = e.shiftKey ? 10 : 1
        onFrameChange(Math.min(maxFrame, frameRef.current + step))
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault()
        stop()
        const step = e.shiftKey ? 10 : 1
        onFrameChange(Math.max(minFrame, frameRef.current - step))
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [isPlaying, play, stop, onFrameChange, minFrame, maxFrame])

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60)
    const sec = (s % 60).toFixed(1)
    return `${m}:${sec.padStart(4, '0')}`
  }

  return (
    <div className="glass-card p-4 space-y-3">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          Frame Scrubber
        </span>
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span>
            Frame <span className="text-foreground font-mono">{currentFrame}</span>
            <span className="text-muted-foreground/50"> / {maxFrame}</span>
          </span>
          <span className="font-mono text-foreground">{formatTime(currentTime)}</span>
          <span className="text-muted-foreground/50">/ {formatTime(duration)}</span>
        </div>
      </div>

      {/* Possession color band */}
      <div className="relative h-1.5 rounded-full overflow-hidden bg-white/5">
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-all duration-100"
          style={{
            width: `${((currentFrame - minFrame) / range) * 100}%`,
            background: `linear-gradient(to right, ${TEAM1_HEX}88, ${TEAM1_HEX})`,
          }}
        />
        <div
          className="absolute inset-y-0 right-0 rounded-full"
          style={{
            width: `${((maxFrame - currentFrame) / range) * 100}%`,
            background: `linear-gradient(to left, ${TEAM2_HEX}88, ${TEAM2_HEX})`,
          }}
        />
      </div>

      {/* Event markers */}
      {events.length > 0 && (
        <div className="relative h-3">
          {events.slice(0, 150).map((event, i) => {
            const pct = ((event.frameNumber - minFrame) / range) * 100
            const color =
              event.teamId === 0 ? TEAM1_HEX : event.teamId === 1 ? TEAM2_HEX : '#6b7280'
            return (
              <div
                key={i}
                className="absolute w-0.5 h-2.5 rounded-full opacity-70 hover:opacity-100 transition-opacity"
                style={{ left: `${pct}%`, backgroundColor: color, top: 0 }}
                title={`${event.type} — frame ${event.frameNumber}`}
              />
            )
          })}
        </div>
      )}

      {/* Slider */}
      <Slider
        value={[currentFrame]}
        min={minFrame}
        max={maxFrame}
        step={1}
        onValueChange={([v]) => {
          stop()
          onFrameChange(v)
        }}
        className="w-full"
      />

      {/* Controls row */}
      <div className="flex items-center justify-between">
        {/* Playback controls */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => {
              stop()
              onFrameChange(minFrame)
            }}
            className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-white/5 transition-colors"
            title="Go to start"
          >
            <SkipBack className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => (isPlaying ? stop() : play())}
            className="p-1.5 rounded-lg bg-white/8 hover:bg-white/12 text-foreground transition-colors border border-white/10"
            title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
          >
            {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
          </button>
          <button
            onClick={() => {
              stop()
              onFrameChange(maxFrame)
            }}
            className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-white/5 transition-colors"
            title="Go to end"
          >
            <SkipForward className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Speed selector */}
        <div className="flex items-center gap-1">
          {SPEEDS.map((s, i) => (
            <button
              key={s}
              onClick={() => setSpeedIdx(i)}
              className={`px-2 py-0.5 rounded text-[10px] font-mono transition-colors ${
                speedIdx === i
                  ? 'bg-white/10 text-foreground border border-white/15'
                  : 'text-muted-foreground hover:text-foreground hover:bg-white/5'
              }`}
            >
              {s}x
            </button>
          ))}
        </div>

        {/* Keyboard hint */}
        <div className="text-[10px] text-muted-foreground/40 hidden sm:block">
          ←→ step · Shift+←→ ×10 · Space play
        </div>
      </div>
    </div>
  )
}
