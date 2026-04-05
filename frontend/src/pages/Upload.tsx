import { useState, useCallback, useRef, useEffect } from 'react'
import { useAuth } from '@/hooks/useAuth'
import { analysisApi } from '@/lib/api-local'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { AnimatedUpload } from '@/components/ui/animated-upload'
import { toast } from 'sonner'
import { useLocation } from 'wouter'
import { getLoginUrl } from '@/const'
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion'
import { cn } from '@/lib/utils'
import {
  Activity,
  Upload as UploadIcon,
  FileVideo,
  X,
  Loader2,
  Layers,
  Radar,
  Users,
  Target,
  User,
  Circle,
  Map,
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  Cpu,
  Cloud,
  Video,
  Camera,
  RotateCcw,
} from 'lucide-react'
import { Link } from 'wouter'
import { PIPELINE_MODES, PipelineMode } from '@/types'

const MODE_ICONS: Record<PipelineMode, React.ReactNode> = {
  all: <Layers className="w-4 h-4" />,
  radar: <Radar className="w-4 h-4" />,
  team: <Users className="w-4 h-4" />,
  track: <Target className="w-4 h-4" />,
  players: <User className="w-4 h-4" />,
  ball: <Circle className="w-4 h-4" />,
  pitch: <Map className="w-4 h-4" />,
}

const STEPS = ['SELECT FILE', 'CONFIGURE', 'UPLOAD'] as const
const SLIDE_EASE = [0.16, 1, 0.3, 1] as const

export default function Upload() {
  const { loading: authLoading, isAuthenticated } = useAuth()
  const [, navigate] = useLocation()
  const shouldReduceMotion = useReducedMotion()

  // Wizard navigation
  const [step, setStep] = useState(0)
  const [direction, setDirection] = useState(1)
  const uploadTriggered = useRef(false)

  // Form state (all preserved from original)
  const titleRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [selectedMode, setSelectedMode] = useState<PipelineMode>('all')
  const [isDragging, setIsDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadStage, setUploadStage] = useState<'uploading' | 'processing' | 'done'>('uploading')
  const [uploadSpeed, setUploadSpeed] = useState(0)
  const [timeRemaining, setTimeRemaining] = useState(0)
  const [bytesTransferred, setBytesTransferred] = useState(0)
  const [useCustomModels, setUseCustomModels] = useState(true)
  const [freshRun, setFreshRun] = useState(false)
  const [cameraType, setCameraType] = useState<'tactical' | 'broadcast'>('tactical')

  const goTo = (nextStep: number) => {
    setDirection(nextStep > step ? 1 : -1)
    if (nextStep < 2) uploadTriggered.current = false
    setStep(nextStep)
  }

  // Drag & drop
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      const droppedFile = e.dataTransfer.files[0]
      if (droppedFile && droppedFile.type.startsWith('video/')) {
        setFile(droppedFile)
        if (!title) setTitle(droppedFile.name.replace(/\.[^/.]+$/, ''))
      } else {
        toast.error('Please upload a video file')
      }
    },
    [title]
  )

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selectedFile = e.target.files?.[0]
      if (selectedFile) {
        setFile(selectedFile)
        if (!title) setTitle(selectedFile.name.replace(/\.[^/.]+$/, ''))
      }
    },
    [title]
  )

  const handleSubmit = async () => {
    if (!file) {
      toast.error('Please select a video file')
      return
    }
    const domTitle = titleRef.current?.value ?? title
    if (!domTitle.trim()) {
      toast.error('Please enter a title')
      return
    }

    setUploading(true)
    setUploadProgress(0)
    setUploadStage('uploading')
    setUploadSpeed(0)
    setTimeRemaining(0)
    setBytesTransferred(0)

    try {
      const formData = new FormData()
      formData.append('video', file)
      formData.append('title', domTitle.trim())
      formData.append('description', description.trim() || '')

      let lastLoaded = 0
      let lastTime = Date.now()

      const uploadResult = await new Promise<{ id: number }>((resolve, reject) => {
        const xhr = new XMLHttpRequest()

        xhr.upload.addEventListener('progress', (e) => {
          if (e.lengthComputable) {
            setUploadProgress(Math.round((e.loaded / e.total) * 80))
            setBytesTransferred(e.loaded)

            const now = Date.now()
            const timeDiff = (now - lastTime) / 1000
            if (timeDiff > 0.5) {
              const speed = (e.loaded - lastLoaded) / timeDiff
              setUploadSpeed(speed)
              setTimeRemaining(Math.round(speed > 0 ? (e.total - e.loaded) / speed : 0))
              lastLoaded = e.loaded
              lastTime = now
            }
          }
        })

        xhr.addEventListener('load', () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              const data = JSON.parse(xhr.responseText)
              if (!data.id) {
                reject(new Error('No video ID in response'))
                return
              }
              resolve(data)
            } catch {
              reject(new Error('Invalid response'))
            }
          } else {
            reject(new Error(`Upload failed: ${xhr.status}`))
          }
        })

        xhr.addEventListener('error', () => reject(new Error('Network error')))
        xhr.addEventListener('abort', () => reject(new Error('Upload cancelled')))
        xhr.open('POST', '/api/upload/video')
        xhr.send(formData)
      })

      setUploadProgress(85)
      setUploadStage('processing')
      setUploadSpeed(0)
      setTimeRemaining(0)

      const { id: analysisId } = await analysisApi.create({
        videoId: uploadResult.id,
        mode: selectedMode,
        fresh: freshRun,
        cameraType,
        useCustomModels,
      })

      setUploadProgress(100)
      setUploadStage('done')
      toast.success('Video uploaded successfully! Starting analysis...')
      navigate(`/analysis/${analysisId}`)
    } catch (error) {
      console.error('Upload error:', error)
      toast.error('Failed to upload video. Please try again.')
      goTo(1)
    } finally {
      setUploading(false)
    }
  }

  // Auto-trigger upload when entering step 2
  useEffect(() => {
    if (step === 2 && !uploadTriggered.current) {
      uploadTriggered.current = true
      handleSubmit()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step])

  if (authLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    )
  }

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardHeader className="text-center">
            <CardTitle>Sign In Required</CardTitle>
            <CardDescription>Please sign in to upload videos</CardDescription>
          </CardHeader>
          <CardContent>
            <a href={getLoginUrl()}>
              <Button className="w-full">Sign In</Button>
            </a>
          </CardContent>
        </Card>
      </div>
    )
  }

  const slideVariants = {
    enter: (dir: number) => ({
      x: shouldReduceMotion ? 0 : dir * 80,
      opacity: 0,
    }),
    center: { x: 0, opacity: 1 },
    exit: (dir: number) => ({
      x: shouldReduceMotion ? 0 : dir * -80,
      opacity: 0,
    }),
  }

  const slideTx = { duration: shouldReduceMotion ? 0.1 : 0.3, ease: SLIDE_EASE }

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* ── Header ── */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container flex items-center justify-between h-14">
          <div className="flex items-center gap-3">
            <Link href="/dashboard">
              <Button variant="ghost" size="icon">
                <ArrowLeft className="w-4 h-4" />
              </Button>
            </Link>
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-md bg-primary flex items-center justify-center">
                <Activity className="w-4 h-4 text-primary-foreground" />
              </div>
              <span className="font-mono font-semibold text-sm tracking-widest">UPLOAD VIDEO</span>
            </div>
          </div>
        </div>

        {/* Step indicator */}
        <div className="container pb-3">
          <div className="flex items-center">
            {STEPS.map((label, i) => (
              <div key={label} className="flex items-center">
                <div className="flex items-center gap-1.5">
                  <motion.div
                    className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-mono font-bold shrink-0"
                    animate={{
                      backgroundColor:
                        step > i
                          ? 'var(--color-primary)'
                          : step === i
                            ? 'var(--color-primary)'
                            : 'var(--color-muted)',
                      color:
                        step >= i
                          ? 'var(--color-primary-foreground)'
                          : 'var(--color-muted-foreground)',
                    }}
                    transition={{ duration: 0.2 }}
                  >
                    {step > i ? <CheckCircle2 className="w-3 h-3" /> : i + 1}
                  </motion.div>
                  <span
                    className={cn(
                      'text-[10px] font-mono hidden sm:block tracking-wider',
                      step === i ? 'text-foreground' : 'text-muted-foreground'
                    )}
                  >
                    {label}
                  </span>
                </div>
                {i < STEPS.length - 1 && (
                  <motion.div
                    className="w-10 h-px mx-2"
                    animate={{
                      backgroundColor: step > i ? 'var(--color-primary)' : 'var(--color-border)',
                    }}
                    transition={{ duration: 0.3 }}
                  />
                )}
              </div>
            ))}
          </div>
        </div>
      </header>

      {/* ── Main ── */}
      <main className="flex-1 container py-10">
        <div className="max-w-2xl mx-auto">
          <AnimatePresence mode="wait" custom={direction}>
            {/* ──────────────────── STEP 0: SELECT FILE ──────────────────── */}
            {step === 0 && (
              <motion.div
                key="step-0"
                custom={direction}
                variants={slideVariants}
                initial="enter"
                animate="center"
                exit="exit"
                transition={slideTx}
                className="space-y-6"
              >
                <div>
                  <h2 className="text-2xl font-bold font-mono tracking-wide">SELECT FILE</h2>
                  <p className="text-muted-foreground text-sm mt-1">
                    Upload football match footage (MP4, MOV, AVI)
                  </p>
                </div>

                {file ? (
                  <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="border-2 border-primary bg-primary/5 rounded-xl p-5 flex items-center justify-between"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-11 h-11 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                        <FileVideo className="w-5 h-5 text-primary" />
                      </div>
                      <div>
                        <p className="font-mono text-sm font-medium truncate max-w-[280px]">
                          {file.name}
                        </p>
                        <p className="text-xs text-muted-foreground font-mono mt-0.5">
                          {(file.size / (1024 * 1024)).toFixed(2)} MB
                        </p>
                      </div>
                    </div>
                    <Button type="button" variant="ghost" size="icon" onClick={() => setFile(null)}>
                      <X className="w-4 h-4" />
                    </Button>
                  </motion.div>
                ) : (
                  <motion.label
                    htmlFor="video-upload"
                    className={cn(
                      'relative flex flex-col items-center justify-center p-16 rounded-xl border-2 border-dashed cursor-pointer transition-colors',
                      isDragging
                        ? 'border-primary bg-primary/5'
                        : 'border-border hover:border-muted-foreground'
                    )}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    animate={isDragging ? { scale: 1.01 } : { scale: 1 }}
                    transition={{ duration: 0.15 }}
                  >
                    <motion.div
                      animate={isDragging ? { y: -6 } : { y: 0 }}
                      transition={{ duration: 0.2 }}
                    >
                      <UploadIcon className="w-12 h-12 text-muted-foreground mb-4" />
                    </motion.div>
                    <p className="text-lg font-semibold mb-1 pointer-events-none">
                      Drag and drop your video here
                    </p>
                    <p className="text-sm text-muted-foreground mb-5 pointer-events-none">
                      or click to browse files
                    </p>
                    <span className="inline-flex h-9 items-center justify-center rounded-md border border-input bg-background px-4 text-sm font-medium hover:bg-accent transition-colors pointer-events-none">
                      Browse Files
                    </span>
                    <input
                      type="file"
                      accept="video/*"
                      onChange={handleFileSelect}
                      className="hidden"
                      id="video-upload"
                    />
                  </motion.label>
                )}

                <div className="flex justify-end pt-2">
                  <Button
                    onClick={() => goTo(1)}
                    disabled={!file}
                    size="lg"
                    className="gap-2 font-mono tracking-wide"
                  >
                    CONFIGURE <ArrowRight className="w-4 h-4" />
                  </Button>
                </div>
              </motion.div>
            )}

            {/* ──────────────────── STEP 1: CONFIGURE ──────────────────── */}
            {step === 1 && (
              <motion.div
                key="step-1"
                custom={direction}
                variants={slideVariants}
                initial="enter"
                animate="center"
                exit="exit"
                transition={slideTx}
                className="space-y-6"
              >
                <div>
                  <h2 className="text-2xl font-bold font-mono tracking-wide">CONFIGURE</h2>
                  <p className="text-muted-foreground text-sm mt-1">
                    Set analysis options for your footage
                  </p>
                </div>

                {/* Title + Description */}
                <div className="space-y-4">
                  <div className="space-y-1.5">
                    <Label
                      htmlFor="title"
                      className="font-mono text-[10px] tracking-widest text-muted-foreground"
                    >
                      TITLE *
                    </Label>
                    <Input
                      ref={titleRef}
                      id="title"
                      name="title"
                      value={title}
                      onChange={(e) => setTitle(e.target.value)}
                      placeholder="e.g., Arsenal vs Chelsea - Premier League"
                      className="font-mono"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <Label
                      htmlFor="description"
                      className="font-mono text-[10px] tracking-widest text-muted-foreground"
                    >
                      DESCRIPTION
                    </Label>
                    <Textarea
                      id="description"
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      placeholder="Add notes about the match, teams, or specific moments..."
                      rows={2}
                      className="font-mono resize-none"
                    />
                  </div>
                </div>

                {/* Camera angle */}
                <div className="space-y-2">
                  <p className="font-mono text-[10px] tracking-widest text-muted-foreground">
                    CAMERA ANGLE
                  </p>
                  <div className="grid grid-cols-2 gap-3">
                    <motion.button
                      type="button"
                      onClick={() => setCameraType('tactical')}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      className={cn(
                        'relative p-3 rounded-xl border-2 text-left transition-colors',
                        cameraType === 'tactical'
                          ? 'border-primary bg-primary/5'
                          : 'border-border hover:border-primary/30'
                      )}
                    >
                      {cameraType === 'tactical' && (
                        <Badge className="absolute top-2 right-2 text-[10px] h-4 px-1">
                          Selected
                        </Badge>
                      )}
                      <div className="flex items-center gap-2 mb-1">
                        <Video className="w-4 h-4 text-primary" />
                        <span className="font-semibold text-sm">Tactical View</span>
                      </div>
                      <p className="text-xs text-muted-foreground">Wide-angle, full pitch</p>
                    </motion.button>

                    <button
                      type="button"
                      disabled
                      className="relative p-3 rounded-xl border-2 border-border opacity-50 cursor-not-allowed text-left"
                    >
                      <Badge className="absolute top-2 right-2 text-[10px] h-4 px-1 bg-amber-500/80 text-white hover:bg-amber-500/80">
                        Soon
                      </Badge>
                      <div className="flex items-center gap-2 mb-1">
                        <Camera className="w-4 h-4" />
                        <span className="font-semibold text-sm">Broadcast View</span>
                      </div>
                      <p className="text-xs text-muted-foreground">TV camera, dynamic</p>
                    </button>
                  </div>
                </div>

                {/* Detection models */}
                <div className="space-y-2">
                  <p className="font-mono text-[10px] tracking-widest text-muted-foreground">
                    DETECTION MODELS
                  </p>
                  <div className="grid grid-cols-2 gap-3">
                    <motion.button
                      type="button"
                      onClick={() => setUseCustomModels(true)}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      className={cn(
                        'relative p-3 rounded-xl border-2 text-left transition-colors',
                        useCustomModels
                          ? 'border-primary bg-primary/5'
                          : 'border-border hover:border-primary/30'
                      )}
                    >
                      {useCustomModels && (
                        <Badge className="absolute top-2 right-2 text-[10px] h-4 px-1">
                          Selected
                        </Badge>
                      )}
                      <div className="flex items-center gap-2 mb-1">
                        <Cpu className="w-4 h-4 text-primary" />
                        <span className="font-semibold text-sm">Custom YOLO</span>
                      </div>
                      <p className="text-xs text-muted-foreground">Local GPU, recommended</p>
                    </motion.button>

                    <motion.button
                      type="button"
                      onClick={() => setUseCustomModels(false)}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      className={cn(
                        'relative p-3 rounded-xl border-2 text-left transition-colors',
                        !useCustomModels
                          ? 'border-primary bg-primary/5'
                          : 'border-border hover:border-primary/30'
                      )}
                    >
                      {!useCustomModels && (
                        <Badge className="absolute top-2 right-2 text-[10px] h-4 px-1">
                          Selected
                        </Badge>
                      )}
                      <div className="flex items-center gap-2 mb-1">
                        <Cloud className="w-4 h-4" />
                        <span className="font-semibold text-sm">Roboflow API</span>
                      </div>
                      <p className="text-xs text-muted-foreground">Cloud-based, fallback</p>
                    </motion.button>
                  </div>
                </div>

                {/* Analysis mode */}
                <div className="space-y-2">
                  <p className="font-mono text-[10px] tracking-widest text-muted-foreground">
                    ANALYSIS MODE
                  </p>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                    {(
                      Object.entries(PIPELINE_MODES) as [
                        PipelineMode,
                        (typeof PIPELINE_MODES)[PipelineMode],
                      ][]
                    ).map(([mode, config]) => (
                      <motion.button
                        key={mode}
                        type="button"
                        onClick={() => setSelectedMode(mode)}
                        whileHover={{ scale: 1.03 }}
                        whileTap={{ scale: 0.97 }}
                        className={cn(
                          'p-3 rounded-xl border text-left transition-colors',
                          selectedMode === mode
                            ? 'border-primary bg-primary/10'
                            : 'border-border hover:border-primary/40 bg-card'
                        )}
                      >
                        <div
                          className={cn(
                            'w-7 h-7 rounded-lg flex items-center justify-center mb-2',
                            selectedMode === mode
                              ? 'bg-primary text-primary-foreground'
                              : 'bg-secondary'
                          )}
                        >
                          {MODE_ICONS[mode]}
                        </div>
                        <p className="text-xs font-semibold leading-tight">{config.name}</p>
                      </motion.button>
                    ))}
                  </div>
                </div>

                {/* Fresh run */}
                <div className="flex items-center justify-between p-4 rounded-xl border border-border">
                  <div className="flex items-center gap-3">
                    <RotateCcw className="w-4 h-4 text-muted-foreground shrink-0" />
                    <div>
                      <p className="text-sm font-semibold">Fresh Run</p>
                      <p className="text-xs text-muted-foreground">
                        Skip cached data, re-run full pipeline
                      </p>
                    </div>
                  </div>
                  <Switch checked={freshRun} onCheckedChange={setFreshRun} />
                </div>

                {/* Navigation */}
                <div className="flex items-center justify-between pt-2">
                  <Button
                    type="button"
                    variant="ghost"
                    onClick={() => goTo(0)}
                    className="gap-2 font-mono tracking-wide"
                  >
                    <ArrowLeft className="w-4 h-4" /> BACK
                  </Button>
                  <Button
                    onClick={() => goTo(2)}
                    disabled={!title.trim()}
                    size="lg"
                    className="gap-2 font-mono tracking-wide"
                  >
                    START ANALYSIS <UploadIcon className="w-4 h-4" />
                  </Button>
                </div>
              </motion.div>
            )}

            {/* ──────────────────── STEP 2: UPLOAD ──────────────────── */}
            {step === 2 && (
              <motion.div
                key="step-2"
                custom={direction}
                variants={slideVariants}
                initial="enter"
                animate="center"
                exit="exit"
                transition={slideTx}
                className="space-y-8"
              >
                <div>
                  <h2 className="text-2xl font-bold font-mono tracking-wide">UPLOAD</h2>
                  <p className="text-muted-foreground text-sm mt-1">
                    Transferring and preparing your analysis
                  </p>
                </div>

                {/* File info strip */}
                {file && (
                  <div className="flex items-center gap-3 p-3 rounded-lg bg-card border border-border">
                    <div className="w-8 h-8 rounded-md bg-primary/10 flex items-center justify-center shrink-0">
                      <FileVideo className="w-4 h-4 text-primary" />
                    </div>
                    <div className="min-w-0">
                      <p className="text-sm font-mono font-medium truncate">{file.name}</p>
                      <p className="text-xs text-muted-foreground font-mono">
                        {(file.size / (1024 * 1024)).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                )}

                {/* Animated upload indicator */}
                <div className="flex justify-center py-4">
                  <AnimatedUpload
                    progress={uploadProgress}
                    stage={uploadStage}
                    speed={uploadSpeed}
                    timeRemaining={timeRemaining}
                    bytesTransferred={bytesTransferred}
                    totalBytes={file?.size ?? 0}
                  />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>
    </div>
  )
}
