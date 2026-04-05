import { cn } from '@/lib/utils'
import React from 'react'
import { motion } from 'framer-motion'

export type FeatureType = {
  title: string
  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>
  description: string
}

type FeatureCardProps = React.ComponentPropsWithoutRef<typeof motion.div> & {
  feature: FeatureType
}

export function FeatureCard({ feature, className, ...props }: FeatureCardProps) {
  const p = genRandomPattern()

  return (
    <motion.div
      className={cn('relative overflow-hidden p-6', className)}
      {...props}
      initial="idle"
      whileHover="hovered"
      variants={{ idle: { y: 0 }, hovered: { y: -8 } }}
      transition={{ type: 'spring', stiffness: 380, damping: 26 }}
    >
      {/* Green gradient glow on hover */}
      <motion.div
        className="absolute inset-0 pointer-events-none z-0"
        variants={{ idle: { opacity: 0 }, hovered: { opacity: 1 } }}
        transition={{ duration: 0.35 }}
        style={{
          background: 'linear-gradient(135deg, rgba(34,197,94,0.1) 0%, transparent 60%)',
          boxShadow: 'inset 0 1px 0 rgba(34,197,94,0.4), inset 1px 0 0 rgba(34,197,94,0.2)',
        }}
      />

      {/* Shimmer sweep left→right */}
      <motion.div
        className="absolute inset-y-0 left-0 w-2/5 pointer-events-none z-10"
        style={{
          background: 'linear-gradient(90deg, transparent, rgba(34,197,94,0.15), transparent)',
        }}
        variants={{
          idle: { x: '-100%', opacity: 0 },
          hovered: { x: '320%', opacity: [0, 1, 1, 0] },
        }}
        transition={{ duration: 0.65, ease: 'easeOut' }}
      />

      {/* Grid pattern */}
      <div className="pointer-events-none absolute top-0 left-1/2 -mt-2 -ml-20 h-full w-full [mask-image:linear-gradient(white,transparent)]">
        <div className="from-foreground/5 to-foreground/1 absolute inset-0 bg-gradient-to-r [mask-image:radial-gradient(farthest-side_at_top,white,transparent)] opacity-100">
          <GridPattern
            width={20}
            height={20}
            x="-12"
            y="4"
            squares={p}
            className="fill-foreground/5 stroke-foreground/25 absolute inset-0 h-full w-full mix-blend-overlay"
          />
        </div>
      </div>

      {/* Icon */}
      <motion.div
        style={{ display: 'inline-block' }}
        variants={{
          idle: { scale: 1, y: 0, color: 'rgba(255,255,255,0.75)' },
          hovered: { scale: 1.45, y: -5, color: 'oklch(0.65 0.2 145)' },
        }}
        transition={{ type: 'spring', stiffness: 500, damping: 30 }}
      >
        <feature.icon className="size-6" strokeWidth={1} aria-hidden />
      </motion.div>

      <motion.h3
        className="mt-10 text-sm md:text-base"
        variants={{
          idle: { color: 'rgba(255,255,255,0.9)' },
          hovered: { color: 'oklch(0.65 0.2 145)' },
        }}
        transition={{ duration: 0.22 }}
      >
        {feature.title}
      </motion.h3>
      <p className="text-muted-foreground relative z-20 mt-2 text-xs font-light">
        {feature.description}
      </p>
    </motion.div>
  )
}

export function GridPattern({
  width,
  height,
  x,
  y,
  squares,
  ...props
}: React.ComponentProps<'svg'> & {
  width: number
  height: number
  x: string
  y: string
  squares?: number[][]
}) {
  const patternId = React.useId()

  return (
    <svg aria-hidden="true" {...props}>
      <defs>
        <pattern
          id={patternId}
          width={width}
          height={height}
          patternUnits="userSpaceOnUse"
          x={x}
          y={y}
        >
          <path d={`M.5 ${height}V.5H${width}`} fill="none" />
        </pattern>
      </defs>
      <rect width="100%" height="100%" strokeWidth={0} fill={`url(#${patternId})`} />
      {squares && (
        <svg x={x} y={y} className="overflow-visible">
          {squares.map(([sx, sy], index) => (
            <rect
              strokeWidth="0"
              key={index}
              width={width + 1}
              height={height + 1}
              x={sx * width}
              y={sy * height}
            />
          ))}
        </svg>
      )}
    </svg>
  )
}

export function genRandomPattern(length = 5): number[][] {
  return Array.from({ length }, () => [
    Math.floor(Math.random() * 4) + 7,
    Math.floor(Math.random() * 6) + 1,
  ])
}
