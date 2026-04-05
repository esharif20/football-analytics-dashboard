import { motion, type Variants } from 'framer-motion'

const containerVariants: Variants = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.04,
    },
  },
}

const wordVariants: Variants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5,
      ease: [0.16, 1, 0.3, 1],
    },
  },
}

interface SplitTextProps {
  text: string
  className?: string
  delay?: number
  amount?: number
}

export function SplitText({ text, className, delay = 0, amount = 0.2 }: SplitTextProps) {
  const words = text.split(' ')

  return (
    <motion.span
      className={className}
      variants={containerVariants}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, amount }}
      transition={{ delayChildren: delay }}
      style={{ display: 'inline' }}
    >
      {words.map((word, i) => (
        <span
          key={i}
          style={{ display: 'inline-block', overflow: 'hidden', verticalAlign: 'bottom' }}
        >
          <motion.span variants={wordVariants} style={{ display: 'inline-block' }}>
            {word}
            {i < words.length - 1 ? '\u00a0' : ''}
          </motion.span>
        </span>
      ))}
    </motion.span>
  )
}
