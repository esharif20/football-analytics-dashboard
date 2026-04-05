import { useState, useEffect, useRef } from 'react'
import { useReducedMotion } from 'framer-motion'
import { cn } from '@/lib/utils'

const ALPHABETS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')
const randomChar = () => ALPHABETS[Math.floor(Math.random() * ALPHABETS.length)]

interface HyperTextProps {
  text: string
  className?: string
  duration?: number
}

export function HyperText({ text, className, duration = 800 }: HyperTextProps) {
  const shouldReduceMotion = useReducedMotion()
  const [displayText, setDisplayText] = useState(text.split(''))
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (intervalRef.current) clearInterval(intervalRef.current)

    if (shouldReduceMotion) {
      setDisplayText(text.split(''))
      return
    }

    let iteration = 0

    intervalRef.current = setInterval(
      () => {
        if (iteration < text.length) {
          setDisplayText(
            text
              .split('')
              .map((char, i) => (char === ' ' ? char : i <= iteration ? text[i] : randomChar()))
          )
          iteration += 0.1
        } else {
          setDisplayText(text.split(''))
          if (intervalRef.current) {
            clearInterval(intervalRef.current)
            intervalRef.current = null
          }
        }
      },
      duration / (text.length * 10)
    )

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }
  }, [text, shouldReduceMotion, duration])

  return (
    <span className={cn('font-mono', className)}>
      {displayText.map((letter, i) => (
        <span key={i} className={letter === ' ' ? 'inline-block w-2' : ''}>
          {letter}
        </span>
      ))}
    </span>
  )
}
