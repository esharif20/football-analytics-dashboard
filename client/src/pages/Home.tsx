import { useAuth } from "@/_core/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { getLoginUrl } from "@/const";
import { Link } from "wouter";
import { 
  Activity, 
  BarChart3, 
  Play, 
  Radar, 
  Target, 
  Upload, 
  Users,
  Zap,
  ArrowRight,
  CheckCircle2
} from "lucide-react";

export default function Home() {
  const { user, loading, isAuthenticated } = useAuth();

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container flex items-center justify-between h-16">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
              <Activity className="w-5 h-5 text-primary-foreground" />
            </div>
            <span className="font-semibold text-lg">Football Analytics</span>
          </div>
          <nav className="flex items-center gap-4">
            {loading ? (
              <div className="w-24 h-9 bg-secondary animate-pulse rounded-md" />
            ) : isAuthenticated ? (
              <>
                <Link href="/dashboard">
                  <Button variant="ghost">Dashboard</Button>
                </Link>
                <Link href="/upload">
                  <Button>
                    <Upload className="w-4 h-4 mr-2" />
                    Upload Video
                  </Button>
                </Link>
              </>
            ) : (
              <a href={getLoginUrl()}>
                <Button>Sign In</Button>
              </a>
            )}
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-20 lg:py-32">
        <div className="container">
          <div className="max-w-4xl mx-auto text-center">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-sm mb-6">
              <Zap className="w-4 h-4" />
              <span>AI-Powered Match Analysis</span>
            </div>
            <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-6">
              Transform Football Footage into{" "}
              <span className="gradient-text">Tactical Insights</span>
            </h1>
            <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              Upload match videos and get real-time player tracking, event detection, 
              heatmaps, pass networks, and AI-generated tactical commentary.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              {isAuthenticated ? (
                <Link href="/upload">
                  <Button size="lg" className="gap-2">
                    Get Started <ArrowRight className="w-4 h-4" />
                  </Button>
                </Link>
              ) : (
                <a href={getLoginUrl()}>
                  <Button size="lg" className="gap-2">
                    Get Started <ArrowRight className="w-4 h-4" />
                  </Button>
                </a>
              )}
              <Link href="/dashboard">
                <Button size="lg" variant="outline" className="gap-2">
                  <Play className="w-4 h-4" /> View Demo
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-20 bg-card/30">
        <div className="container">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Comprehensive Analysis Pipeline</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Choose from multiple analysis modes to get exactly the insights you need
            </p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <FeatureCard
              icon={<Target className="w-6 h-6" />}
              title="Player Detection"
              description="YOLOv8-powered detection of players, referees, and goalkeepers with high accuracy"
            />
            <FeatureCard
              icon={<Users className="w-6 h-6" />}
              title="Team Classification"
              description="SigLIP embeddings with UMAP and KMeans for automatic team identification"
            />
            <FeatureCard
              icon={<Radar className="w-6 h-6" />}
              title="2D Pitch Radar"
              description="Real-time bird's eye view with player positions and ball trajectory"
            />
            <FeatureCard
              icon={<BarChart3 className="w-6 h-6" />}
              title="Match Statistics"
              description="Possession, passes, shots, distance covered, and speed metrics"
            />
            <FeatureCard
              icon={<Activity className="w-6 h-6" />}
              title="Event Detection"
              description="Automatic detection of passes, shots, challenges, and interceptions"
            />
            <FeatureCard
              icon={<Zap className="w-6 h-6" />}
              title="AI Commentary"
              description="Grounded tactical analysis generated from tracking data"
            />
          </div>
        </div>
      </section>

      {/* Pipeline Modes */}
      <section className="py-20">
        <div className="container">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Flexible Pipeline Modes</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Run the full analysis or select specific components based on your needs
            </p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 max-w-5xl mx-auto">
            {[
              { name: "Full Analysis", desc: "Complete pipeline" },
              { name: "Radar View", desc: "2D pitch visualization" },
              { name: "Team Analysis", desc: "Team classification" },
              { name: "Object Tracking", desc: "ByteTrack persistence" },
              { name: "Player Detection", desc: "Bounding boxes" },
              { name: "Ball Tracking", desc: "SAHI + interpolation" },
              { name: "Pitch Mapping", desc: "Homography transform" },
            ].map((mode, i) => (
              <div key={i} className="bg-card border border-border rounded-lg p-4 hover:border-primary/50 transition-colors">
                <div className="flex items-center gap-2 mb-1">
                  <CheckCircle2 className="w-4 h-4 text-primary" />
                  <span className="font-medium">{mode.name}</span>
                </div>
                <p className="text-sm text-muted-foreground">{mode.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-b from-card/50 to-background">
        <div className="container">
          <div className="max-w-2xl mx-auto text-center">
            <h2 className="text-3xl font-bold mb-4">Ready to Analyze Your Matches?</h2>
            <p className="text-muted-foreground mb-8">
              Upload your first video and see the AI-powered analysis in action.
            </p>
            {isAuthenticated ? (
              <Link href="/upload">
                <Button size="lg" className="gap-2">
                  <Upload className="w-4 h-4" /> Upload Video
                </Button>
              </Link>
            ) : (
              <a href={getLoginUrl()}>
                <Button size="lg" className="gap-2">
                  Sign In to Get Started
                </Button>
              </a>
            )}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-8">
        <div className="container">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded bg-primary flex items-center justify-center">
                <Activity className="w-4 h-4 text-primary-foreground" />
              </div>
              <span className="text-sm text-muted-foreground">Football Analytics Platform</span>
            </div>
            <p className="text-sm text-muted-foreground">
              Powered by YOLOv8, ByteTrack, SigLIP, and advanced computer vision
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <div className="bg-card border border-border rounded-xl p-6 hover:border-primary/30 transition-colors">
      <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center text-primary mb-4">
        {icon}
      </div>
      <h3 className="font-semibold text-lg mb-2">{title}</h3>
      <p className="text-muted-foreground">{description}</p>
    </div>
  );
}
