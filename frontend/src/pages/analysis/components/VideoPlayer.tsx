import { useRef } from 'react'
import { CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Video } from 'lucide-react'
import { AnimatedSection } from '../context'

export function VideoPlayer({ analysis, videoUrl }: { analysis: any; videoUrl: string | null }) {
  const videoRef = useRef<HTMLVideoElement>(null)

  return (
    <AnimatedSection className="lg:col-span-3" delay={0}>
      {videoUrl ? (
        <div className="glass-card overflow-hidden hover-lift group">
          <div className="absolute -top-10 -left-10 w-32 h-32 rounded-full bg-sky-500/5 blur-3xl group-hover:bg-sky-500/10 transition-colors duration-700" />
          <CardHeader className="pb-4 pt-5 px-5 border-b border-border/30 relative">
            <div className="flex items-center gap-3">
              <div className="section-icon icon-accent">
                <Video className="w-4 h-4 text-sky-400" />
              </div>
              <div>
                <CardTitle className="text-sm font-semibold">Annotated Video</CardTitle>
                <CardDescription className="text-xs mt-0.5">
                  AI-processed output with bounding boxes &amp; track IDs
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="p-4">
            <div className="video-player-container rounded-xl overflow-hidden ring-1 ring-white/5">
              <video
                ref={videoRef}
                src={videoUrl}
                controls
                className="w-full h-full object-contain"
              />
            </div>
          </CardContent>
        </div>
      ) : (
        <div className="glass-card p-8 flex flex-col items-center justify-center min-h-[300px] hover-lift">
          <div className="w-16 h-16 rounded-2xl bg-muted/50 flex items-center justify-center mb-4">
            <Video className="w-8 h-8 text-muted-foreground" />
          </div>
          <p className="text-muted-foreground text-sm">
            No annotated video available for this mode
          </p>
        </div>
      )}
    </AnimatedSection>
  )
}
