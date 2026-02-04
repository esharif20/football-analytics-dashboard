import { useAuth } from "@/_core/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
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
  CheckCircle2,
  Cpu,
  Layers,
  Eye,
  TrendingUp,
  Map,
  Circle,
  GitBranch,
  Sparkles
} from "lucide-react";

// CDN URLs for images
const IMAGES = {
  stadium: "https://files.manuscdn.com/user_upload_by_module/session_file/310519663334363677/xjEbKFcxSRGvJxJg.jpg",
  heatmap: "https://files.manuscdn.com/user_upload_by_module/session_file/310519663334363677/yErWHRzqZbziIiNx.png",
  aiSports: "https://files.manuscdn.com/user_upload_by_module/session_file/310519663334363677/DZhnIKilPcyeZjEv.jpg",
};

export default function Home() {
  const { user, loading, isAuthenticated } = useAuth();

  return (
    <div className="min-h-screen bg-background overflow-hidden">
      {/* Header */}
      <header className="border-b border-border/50 bg-background/80 backdrop-blur-xl sticky top-0 z-50">
        <div className="container flex items-center justify-between h-16">
          <div className="flex items-center gap-2">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary to-emerald-400 flex items-center justify-center shadow-lg shadow-primary/25">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-lg tracking-tight">Football Analytics</span>
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
                  <Button className="shadow-lg shadow-primary/25">
                    <Upload className="w-4 h-4 mr-2" />
                    Upload Video
                  </Button>
                </Link>
              </>
            ) : (
              <a href={getLoginUrl()}>
                <Button className="shadow-lg shadow-primary/25">Sign In</Button>
              </a>
            )}
          </nav>
        </div>
      </header>

      {/* Hero Section with Background */}
      <section className="relative py-24 lg:py-36 overflow-hidden">
        {/* Animated Background Elements */}
        <div className="absolute inset-0 overflow-hidden">
          {/* Gradient Orbs */}
          <div className="absolute top-20 left-10 w-72 h-72 bg-primary/20 rounded-full blur-3xl animate-pulse" />
          <div className="absolute bottom-20 right-10 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: "1s" }} />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-primary/5 rounded-full blur-3xl" />
          
          {/* Grid Pattern */}
          <div className="absolute inset-0 bg-[linear-gradient(rgba(34,197,94,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(34,197,94,0.03)_1px,transparent_1px)] bg-[size:60px_60px]" />
          
          {/* Floating Pitch Lines */}
          <svg className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[500px] opacity-[0.03]" viewBox="0 0 105 68">
            <rect x="0" y="0" width="105" height="68" fill="none" stroke="currentColor" strokeWidth="0.5" />
            <line x1="52.5" y1="0" x2="52.5" y2="68" stroke="currentColor" strokeWidth="0.5" />
            <circle cx="52.5" cy="34" r="9.15" fill="none" stroke="currentColor" strokeWidth="0.5" />
            <rect x="0" y="13.84" width="16.5" height="40.32" fill="none" stroke="currentColor" strokeWidth="0.5" />
            <rect x="88.5" y="13.84" width="16.5" height="40.32" fill="none" stroke="currentColor" strokeWidth="0.5" />
          </svg>
        </div>

        <div className="container relative z-10">
          <div className="max-w-5xl mx-auto text-center">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 text-primary text-sm mb-8 backdrop-blur-sm">
              <Sparkles className="w-4 h-4" />
              <span className="font-medium">AI-Powered Match Analysis</span>
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-8 leading-[1.1]">
              Transform Football Footage into{" "}
              <span className="relative">
                <span className="gradient-text">Tactical Insights</span>
                <svg className="absolute -bottom-2 left-0 w-full h-3 text-primary/30" viewBox="0 0 200 8" preserveAspectRatio="none">
                  <path d="M0 7 Q50 0 100 7 Q150 14 200 7" fill="none" stroke="currentColor" strokeWidth="2" />
                </svg>
              </span>
            </h1>
            
            <p className="text-xl md:text-2xl text-muted-foreground mb-10 max-w-3xl mx-auto leading-relaxed">
              Upload match videos and get real-time player tracking, event detection, 
              heatmaps, pass networks, and AI-generated tactical commentary.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-16">
              {isAuthenticated ? (
                <Link href="/upload">
                  <Button size="lg" className="gap-2 h-14 px-8 text-lg shadow-xl shadow-primary/30 hover:shadow-primary/40 transition-shadow">
                    Get Started <ArrowRight className="w-5 h-5" />
                  </Button>
                </Link>
              ) : (
                <a href={getLoginUrl()}>
                  <Button size="lg" className="gap-2 h-14 px-8 text-lg shadow-xl shadow-primary/30 hover:shadow-primary/40 transition-shadow">
                    Get Started <ArrowRight className="w-5 h-5" />
                  </Button>
                </a>
              )}
              <Link href="/dashboard">
                <Button size="lg" variant="outline" className="gap-2 h-14 px-8 text-lg border-border/50 hover:bg-card">
                  <Play className="w-5 h-5" /> View Demo
                </Button>
              </Link>
            </div>

            {/* Hero Visual - Dashboard Preview */}
            <div className="relative mx-auto max-w-4xl">
              <div className="absolute -inset-4 bg-gradient-to-r from-primary/20 via-emerald-500/20 to-primary/20 rounded-2xl blur-xl opacity-50" />
              <div className="relative rounded-xl overflow-hidden border border-border/50 shadow-2xl bg-card">
                <img 
                  src={IMAGES.aiSports} 
                  alt="AI Sports Analytics" 
                  className="w-full h-auto opacity-90"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-background via-transparent to-transparent" />
                
                {/* Floating Stats Cards */}
                <div className="absolute bottom-4 left-4 right-4 flex gap-3 justify-center flex-wrap">
                  <StatBadge icon={<Users className="w-4 h-4" />} label="22 Players" />
                  <StatBadge icon={<Target className="w-4 h-4" />} label="98.5% Accuracy" />
                  <StatBadge icon={<Zap className="w-4 h-4" />} label="Real-time" />
                  <StatBadge icon={<TrendingUp className="w-4 h-4" />} label="AI Insights" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Tech Stack Banner */}
      <section className="py-8 border-y border-border/50 bg-card/30">
        <div className="container">
          <div className="flex flex-wrap items-center justify-center gap-8 text-muted-foreground">
            <span className="text-sm font-medium">Powered by</span>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-secondary flex items-center justify-center">
                <Cpu className="w-4 h-4" />
              </div>
              <span className="font-medium">YOLOv8</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-secondary flex items-center justify-center">
                <GitBranch className="w-4 h-4" />
              </div>
              <span className="font-medium">ByteTrack</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-secondary flex items-center justify-center">
                <Eye className="w-4 h-4" />
              </div>
              <span className="font-medium">SigLIP</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-secondary flex items-center justify-center">
                <Sparkles className="w-4 h-4" />
              </div>
              <span className="font-medium">Custom Models</span>
            </div>
          </div>
        </div>
      </section>

      {/* Features Grid with Visuals */}
      <section className="py-24">
        <div className="container">
          <div className="text-center mb-16">
            <Badge variant="outline" className="mb-4">Features</Badge>
            <h2 className="text-4xl font-bold mb-4">Comprehensive Analysis Pipeline</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto text-lg">
              Choose from multiple analysis modes to get exactly the insights you need
            </p>
          </div>
          
          {/* Main Feature Cards */}
          <div className="grid lg:grid-cols-2 gap-8 mb-12">
            {/* Heatmap Feature */}
            <div className="group relative bg-card border border-border/50 rounded-2xl overflow-hidden hover:border-primary/30 transition-all duration-300">
              <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
              <div className="relative p-8">
                <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-red-500/20 to-orange-500/20 flex items-center justify-center text-red-400 mb-6">
                  <Map className="w-7 h-7" />
                </div>
                <h3 className="font-bold text-2xl mb-3">Interactive Heatmaps</h3>
                <p className="text-muted-foreground mb-6">
                  Visualize player movement patterns, possession zones, and high-activity areas with dynamic heatmap overlays on the pitch.
                </p>
                <img 
                  src={IMAGES.heatmap} 
                  alt="Heatmap Visualization" 
                  className="w-full h-48 object-cover rounded-lg border border-border/50"
                />
              </div>
            </div>

            {/* Radar Feature */}
            <div className="group relative bg-card border border-border/50 rounded-2xl overflow-hidden hover:border-primary/30 transition-all duration-300">
              <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
              <div className="relative p-8">
                <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-primary/20 to-emerald-500/20 flex items-center justify-center text-primary mb-6">
                  <Radar className="w-7 h-7" />
                </div>
                <h3 className="font-bold text-2xl mb-3">2D Pitch Radar</h3>
                <p className="text-muted-foreground mb-6">
                  Real-time bird's eye view with player positions, ball trajectory, Voronoi diagrams, and team formations.
                </p>
                <div className="w-full h-48 rounded-lg border border-border/50 bg-[#0d1117] flex items-center justify-center overflow-hidden">
                  {/* Mini Pitch Preview */}
                  <svg viewBox="0 0 105 68" className="w-full h-full p-4">
                    <rect x="0" y="0" width="105" height="68" fill="none" stroke="#22c55e" strokeWidth="0.5" opacity="0.5" />
                    <line x1="52.5" y1="0" x2="52.5" y2="68" stroke="#22c55e" strokeWidth="0.5" opacity="0.5" />
                    <circle cx="52.5" cy="34" r="9.15" fill="none" stroke="#22c55e" strokeWidth="0.5" opacity="0.5" />
                    {/* Team 1 dots */}
                    {[{x:15,y:34},{x:30,y:15},{x:30,y:53},{x:45,y:25},{x:45,y:43},{x:60,y:34}].map((p,i) => (
                      <circle key={`t1-${i}`} cx={p.x} cy={p.y} r="2" fill="#22c55e" />
                    ))}
                    {/* Team 2 dots */}
                    {[{x:90,y:34},{x:75,y:15},{x:75,y:53},{x:60,y:20},{x:60,y:48}].map((p,i) => (
                      <circle key={`t2-${i}`} cx={p.x} cy={p.y} r="2" fill="#ef4444" />
                    ))}
                    {/* Ball */}
                    <circle cx="52" cy="34" r="1.5" fill="white" />
                  </svg>
                </div>
              </div>
            </div>
          </div>

          {/* Smaller Feature Cards */}
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <FeatureCard
              icon={<Target className="w-6 h-6" />}
              title="Player Detection"
              description="YOLOv8-powered detection with custom-trained models for ball and pitch"
              gradient="from-blue-500/20 to-cyan-500/20"
              iconColor="text-blue-400"
            />
            <FeatureCard
              icon={<Users className="w-6 h-6" />}
              title="Team Classification"
              description="SigLIP embeddings with UMAP and KMeans clustering"
              gradient="from-purple-500/20 to-pink-500/20"
              iconColor="text-purple-400"
            />
            <FeatureCard
              icon={<Activity className="w-6 h-6" />}
              title="Event Detection"
              description="Automatic detection of passes, shots, and challenges"
              gradient="from-orange-500/20 to-yellow-500/20"
              iconColor="text-orange-400"
            />
            <FeatureCard
              icon={<Sparkles className="w-6 h-6" />}
              title="AI Commentary"
              description="Grounded tactical analysis from tracking data"
              gradient="from-primary/20 to-emerald-500/20"
              iconColor="text-primary"
            />
          </div>
        </div>
      </section>

      {/* Model Selection Section */}
      <section className="py-24 bg-card/30 border-y border-border/50">
        <div className="container">
          <div className="max-w-5xl mx-auto">
            <div className="text-center mb-12">
              <Badge variant="outline" className="mb-4">Flexible Detection</Badge>
              <h2 className="text-4xl font-bold mb-4">Use Your Own Models</h2>
              <p className="text-muted-foreground max-w-2xl mx-auto text-lg">
                Choose between custom-trained models for faster local inference or cloud APIs as fallback
              </p>
            </div>

            <div className="grid md:grid-cols-2 gap-8">
              {/* Custom Models Card */}
              <div className="relative bg-card border border-primary/30 rounded-2xl p-8 overflow-hidden">
                <div className="absolute top-0 right-0 w-32 h-32 bg-primary/10 rounded-full blur-2xl" />
                <Badge className="mb-4 bg-primary/20 text-primary border-primary/30">Recommended</Badge>
                <h3 className="font-bold text-2xl mb-3">Custom Trained Models</h3>
                <p className="text-muted-foreground mb-6">
                  Use your own YOLOv8 models trained specifically for ball and pitch detection. Faster inference, no API costs.
                </p>
                <ul className="space-y-3 mb-6">
                  <li className="flex items-center gap-3 text-sm">
                    <CheckCircle2 className="w-5 h-5 text-primary" />
                    <span>ball_detection.pt - Custom ball tracker</span>
                  </li>
                  <li className="flex items-center gap-3 text-sm">
                    <CheckCircle2 className="w-5 h-5 text-primary" />
                    <span>pitch_detection.pt - Keypoint detection</span>
                  </li>
                  <li className="flex items-center gap-3 text-sm">
                    <CheckCircle2 className="w-5 h-5 text-primary" />
                    <span>Local GPU inference (faster)</span>
                  </li>
                  <li className="flex items-center gap-3 text-sm">
                    <CheckCircle2 className="w-5 h-5 text-primary" />
                    <span>No API rate limits</span>
                  </li>
                </ul>
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Cpu className="w-4 h-4" />
                  <span>Requires GPU for optimal performance</span>
                </div>
              </div>

              {/* Roboflow API Card */}
              <div className="relative bg-card border border-border/50 rounded-2xl p-8 overflow-hidden">
                <div className="absolute top-0 right-0 w-32 h-32 bg-secondary/50 rounded-full blur-2xl" />
                <Badge variant="secondary" className="mb-4">Fallback Option</Badge>
                <h3 className="font-bold text-2xl mb-3">Roboflow API</h3>
                <p className="text-muted-foreground mb-6">
                  Cloud-based detection using Roboflow's hosted models. Good for quick testing without local GPU setup.
                </p>
                <ul className="space-y-3 mb-6">
                  <li className="flex items-center gap-3 text-sm">
                    <CheckCircle2 className="w-5 h-5 text-muted-foreground" />
                    <span>No local GPU required</span>
                  </li>
                  <li className="flex items-center gap-3 text-sm">
                    <CheckCircle2 className="w-5 h-5 text-muted-foreground" />
                    <span>Easy cloud deployment</span>
                  </li>
                  <li className="flex items-center gap-3 text-sm">
                    <CheckCircle2 className="w-5 h-5 text-muted-foreground" />
                    <span>Pre-trained pitch keypoints</span>
                  </li>
                  <li className="flex items-center gap-3 text-sm text-muted-foreground/70">
                    <Circle className="w-5 h-5" />
                    <span>API rate limits apply</span>
                  </li>
                </ul>
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Zap className="w-4 h-4" />
                  <span>Requires API key configuration</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Pipeline Modes */}
      <section className="py-24">
        <div className="container">
          <div className="text-center mb-12">
            <Badge variant="outline" className="mb-4">Pipeline Modes</Badge>
            <h2 className="text-4xl font-bold mb-4">Flexible Analysis Options</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto text-lg">
              Run the full analysis or select specific components based on your needs
            </p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 max-w-5xl mx-auto">
            {[
              { name: "Full Analysis", desc: "Complete pipeline with all features", icon: <Layers className="w-5 h-5" /> },
              { name: "Radar View", desc: "2D pitch visualization only", icon: <Radar className="w-5 h-5" /> },
              { name: "Team Analysis", desc: "Team classification & formations", icon: <Users className="w-5 h-5" /> },
              { name: "Object Tracking", desc: "ByteTrack persistence", icon: <Target className="w-5 h-5" /> },
              { name: "Player Detection", desc: "Bounding boxes only", icon: <Eye className="w-5 h-5" /> },
              { name: "Ball Tracking", desc: "SAHI + interpolation", icon: <Circle className="w-5 h-5" /> },
              { name: "Pitch Mapping", desc: "Homography transform", icon: <Map className="w-5 h-5" /> },
            ].map((mode, i) => (
              <div key={i} className="group bg-card border border-border/50 rounded-xl p-5 hover:border-primary/50 hover:bg-card/80 transition-all duration-300 cursor-pointer">
                <div className="flex items-center gap-3 mb-2">
                  <div className="w-10 h-10 rounded-lg bg-secondary flex items-center justify-center text-muted-foreground group-hover:text-primary group-hover:bg-primary/10 transition-colors">
                    {mode.icon}
                  </div>
                  <span className="font-semibold">{mode.name}</span>
                </div>
                <p className="text-sm text-muted-foreground pl-13">{mode.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-primary/5 via-primary/10 to-background" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[400px] bg-primary/10 rounded-full blur-3xl" />
        
        <div className="container relative z-10">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="text-4xl md:text-5xl font-bold mb-6">Ready to Analyze Your Matches?</h2>
            <p className="text-xl text-muted-foreground mb-10">
              Upload your first video and see the AI-powered analysis in action.
            </p>
            {isAuthenticated ? (
              <Link href="/upload">
                <Button size="lg" className="gap-2 h-14 px-10 text-lg shadow-xl shadow-primary/30">
                  <Upload className="w-5 h-5" /> Upload Video
                </Button>
              </Link>
            ) : (
              <a href={getLoginUrl()}>
                <Button size="lg" className="gap-2 h-14 px-10 text-lg shadow-xl shadow-primary/30">
                  Sign In to Get Started
                </Button>
              </a>
            )}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border/50 py-12 bg-card/30">
        <div className="container">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-emerald-400 flex items-center justify-center shadow-lg shadow-primary/25">
                <Activity className="w-5 h-5 text-white" />
              </div>
              <div>
                <span className="font-bold">Football Analytics</span>
                <p className="text-sm text-muted-foreground">AI-Powered Match Analysis</p>
              </div>
            </div>
            <p className="text-sm text-muted-foreground">
              Powered by YOLOv8, ByteTrack, SigLIP, and custom-trained models
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

function StatBadge({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-card/90 backdrop-blur-sm border border-border/50 text-sm">
      <span className="text-primary">{icon}</span>
      <span className="font-medium">{label}</span>
    </div>
  );
}

function FeatureCard({ 
  icon, 
  title, 
  description, 
  gradient, 
  iconColor 
}: { 
  icon: React.ReactNode; 
  title: string; 
  description: string;
  gradient: string;
  iconColor: string;
}) {
  return (
    <div className="group bg-card border border-border/50 rounded-xl p-6 hover:border-primary/30 transition-all duration-300">
      <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center ${iconColor} mb-4 group-hover:scale-110 transition-transform`}>
        {icon}
      </div>
      <h3 className="font-semibold text-lg mb-2">{title}</h3>
      <p className="text-muted-foreground text-sm">{description}</p>
    </div>
  );
}
