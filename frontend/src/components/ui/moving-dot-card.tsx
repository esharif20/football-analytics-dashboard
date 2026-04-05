import React, { useState, useEffect } from 'react'

interface DotCardProps {
  target?: number
  duration?: number
  label?: string
}

export default function DotCard({
  target = 777000,
  duration = 2000,
  label = 'Views',
}: DotCardProps) {
  const [count, setCount] = useState(0)

  useEffect(() => {
    let start = 0
    const end = target
    const range = end - start
    if (range <= 0) return
    const increment = Math.ceil(end / (duration / 50))
    const timer = setInterval(() => {
      start += increment
      if (start >= end) {
        start = end
        clearInterval(timer)
      }
      setCount(start)
    }, 50)
    return () => clearInterval(timer)
  }, [target, duration])

  const display = count < 1000 ? count : `${Math.floor(count / 1000)}k`

  return (
    <div className="moving-dot-outer">
      <div className="moving-dot-card">
        <div className="moving-dot-ray" />
        <div className="moving-dot-text">{display}</div>
        <div className="moving-dot-label">{label}</div>
        <div className="moving-dot-line moving-dot-topl" />
        <div className="moving-dot-line moving-dot-leftl" />
        <div className="moving-dot-line moving-dot-bottoml" />
        <div className="moving-dot-line moving-dot-rightl" />
      </div>
    </div>
  )
}
