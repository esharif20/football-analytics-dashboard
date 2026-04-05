import { useEffect, useRef } from 'react'
import { createHeatRenderer } from './pitchHeatGL'

interface PitchCanvasProps {
  className?: string
}

/**
 * WebGL GPGPU cursor heat overlay.
 * Ping-pong framebuffer simulation — GPU diffusion, decay, Gaussian injection.
 * Renders pitch line SDF + 80px grid as in-shader geometry that subtly reveals
 * on cursor proximity. Hides the DOM SVG + CSS grid overlays when active.
 * Skipped on touch devices and prefers-reduced-motion.
 * Pauses rAF when scrolled off-screen.
 */
export function PitchCanvas({ className }: PitchCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (window.matchMedia('(pointer: coarse)').matches) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return

    const canvas = canvasRef.current
    if (!canvas) return

    const parent = canvas.parentElement ?? canvas
    let w = parent.clientWidth
    let h = parent.clientHeight
    canvas.width = w
    canvas.height = h
    canvas.style.width = `${w}px`
    canvas.style.height = `${h}px`

    const renderer = createHeatRenderer(canvas)

    // Hide DOM fallbacks — pitch lines + grid are now rendered in-shader
    const hero = canvas.closest('section')
    let svgFallback: Element | null = null
    let gridDiv: HTMLElement | null = null
    if (hero) {
      svgFallback = hero.querySelector('.pitch-svg-fallback')
      svgFallback?.classList.add('hidden')
      // Grid div: bg-[linear-gradient(...)] inside the same background container
      gridDiv = parent.querySelector<HTMLElement>('[class*="bg-[linear-gradient"]')
      if (gridDiv) gridDiv.style.display = 'none'
    }

    let rafId = 0
    let lastTime = 0
    let isVisible = true
    let mouseX = -1,
      mouseY = -1
    let prevMouseX = -1,
      prevMouseY = -1
    let mouseInHero = false

    // ── Resize ───────────────────────────────────────────────────
    const ro = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (!entry) return
      w = entry.contentRect.width
      h = entry.contentRect.height
      canvas.width = w
      canvas.height = h
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      renderer.resize(w, h)
    })
    ro.observe(parent)

    // ── Pause when off-screen ────────────────────────────────────
    const io = new IntersectionObserver(
      ([entry]) => {
        isVisible = entry.isIntersecting
      },
      { threshold: 0 }
    )
    io.observe(canvas)

    // ── Mouse tracking ───────────────────────────────────────────
    function onMouseMove(e: MouseEvent) {
      const bounds = canvas!.getBoundingClientRect()
      mouseInHero =
        e.clientX >= bounds.left &&
        e.clientX <= bounds.right &&
        e.clientY >= bounds.top &&
        e.clientY <= bounds.bottom
      mouseX = e.clientX - bounds.left
      mouseY = e.clientY - bounds.top
    }

    function onMouseLeave() {
      mouseInHero = false
    }

    window.addEventListener('mousemove', onMouseMove, { passive: true })
    document.addEventListener('mouseleave', onMouseLeave)

    // ── Animation loop ───────────────────────────────────────────
    function frame(now: number) {
      rafId = requestAnimationFrame(frame)
      if (!isVisible) return

      const dt = lastTime === 0 ? 1 / 60 : Math.min((now - lastTime) / 1000, 0.05)
      lastTime = now

      if (mouseInHero && mouseX >= 0) {
        const px = prevMouseX >= 0 ? prevMouseX : mouseX
        const py = prevMouseY >= 0 ? prevMouseY : mouseY
        renderer.inject(mouseX / w, 1.0 - mouseY / h, px / w, 1.0 - py / h, 0.08, 0.78)
      }

      renderer.step(dt)
      renderer.render()

      prevMouseX = mouseX
      prevMouseY = mouseY
    }

    rafId = requestAnimationFrame(frame)
    canvas.classList.add('canvas-ready')

    return () => {
      cancelAnimationFrame(rafId)
      ro.disconnect()
      io.disconnect()
      window.removeEventListener('mousemove', onMouseMove)
      document.removeEventListener('mouseleave', onMouseLeave)
      renderer.destroy()

      // Restore DOM fallbacks
      svgFallback?.classList.remove('hidden')
      if (gridDiv) gridDiv.style.display = ''
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 pointer-events-none ${className ?? ''}`}
      aria-hidden="true"
    />
  )
}
