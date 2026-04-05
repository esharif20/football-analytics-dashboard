import { useEffect } from 'react'
import { useAnimationFrame } from 'framer-motion'
import Lenis from 'lenis'

let lenisInstance: Lenis | null = null

export function useLenis() {
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return

    lenisInstance = new Lenis({
      duration: 1.2,
      easing: (t: number) => 1 - Math.pow(2, -10 * t),
      wheelMultiplier: 1,
      touchMultiplier: 2,
    })

    return () => {
      lenisInstance?.destroy()
      lenisInstance = null
    }
  }, [])

  useAnimationFrame((time) => {
    lenisInstance?.raf(time)
  })
}

export function getLenis() {
  return lenisInstance
}
