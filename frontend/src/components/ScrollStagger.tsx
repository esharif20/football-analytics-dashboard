import { motion, type Variants } from 'framer-motion'
import type { ReactNode } from 'react'

const containerVariants: Variants = {
  hidden: {},
  visible: (stagger: number) => ({
    transition: {
      staggerChildren: stagger,
      delayChildren: 0,
    },
  }),
}

export const staggerItemVariants: Variants = {
  hidden: { opacity: 0, y: 8 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.18,
      ease: [0.16, 1, 0.3, 1],
    },
  },
}

interface ScrollStaggerProps {
  children: ReactNode
  stagger?: number
  className?: string
  amount?: number
  delay?: number
}

export function ScrollStagger({
  children,
  stagger = 0.03,
  className,
  amount = 0.1,
  delay = 0,
}: ScrollStaggerProps) {
  return (
    <motion.div
      className={className}
      variants={containerVariants}
      custom={stagger}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, amount }}
      transition={{ delayChildren: delay }}
    >
      {children}
    </motion.div>
  )
}

export function StaggerItem({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <motion.div className={className} variants={staggerItemVariants}>
      {children}
    </motion.div>
  )
}
