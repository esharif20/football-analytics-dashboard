import { useRef } from 'react'
import { motion, useScroll, useTransform } from 'framer-motion'
import type { ReactNode } from 'react'

interface ParallaxSectionProps {
  children: ReactNode
  className?: string
  backgroundElement?: ReactNode
  bgRange?: [number, number]
}

export function ParallaxSection({
  children,
  className = '',
  backgroundElement,
  bgRange = [-30, 30],
}: ParallaxSectionProps) {
  const ref = useRef<HTMLDivElement>(null)
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ['start end', 'end start'],
  })
  const bgY = useTransform(scrollYProgress, [0, 1], bgRange)

  return (
    <div ref={ref} className={`relative overflow-hidden ${className}`}>
      {backgroundElement && (
        <motion.div
          className="absolute inset-x-0 pointer-events-none"
          style={{ y: bgY, top: '-40px', bottom: '-40px' }}
        >
          {backgroundElement}
        </motion.div>
      )}
      <div className="relative z-10">{children}</div>
    </div>
  )
}
