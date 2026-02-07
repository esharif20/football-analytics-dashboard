import { useAuth } from "@/_core/hooks/useAuth";
import { videosApi, analysisApi } from "@/lib/api-local";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { toast } from "sonner";
import { Link, useLocation } from "wouter";
import { getLoginUrl } from "@/const";
import {
  Activity,
  Upload,
  FileVideo,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  Play,
  Trash2,
  BarChart3,
  ArrowRight,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { PIPELINE_MODES, PipelineMode } from "@/shared/types";

export default function Dashboard() {
  const { user, loading: authLoading, isAuthenticated } = useAuth();
  const [, navigate] = useLocation();
  const queryClient = useQueryClient();

  const { data: videos, isLoading: videosLoading } = useQuery({
    queryKey: ["videos"],
    queryFn: () => videosApi.list(),
    enabled: isAuthenticated,
  });

  const { data: analyses, isLoading: analysesLoading } = useQuery({
    queryKey: ["analyses"],
    queryFn: () => analysisApi.list(),
    enabled: isAuthenticated,
  });

  const deleteVideoMutation = useMutation({
    mutationFn: (videoId: number) => videosApi.delete(videoId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["videos"] });
      queryClient.invalidateQueries({ queryKey: ["analyses"] });
      toast.success("Video deleted successfully");
    },
    onError: () => {
      toast.error("Failed to delete video");
    },
  });

  const handleDeleteVideo = async (videoId: number) => {
    if (!confirm("Are you sure you want to delete this video?")) return;
    deleteVideoMutation.mutate(videoId);
  };

  if (authLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardHeader className="text-center">
            <CardTitle>Sign In Required</CardTitle>
            <CardDescription>Please sign in to view your dashboard</CardDescription>
          </CardHeader>
          <CardContent>
            <a href={getLoginUrl()}>
              <Button className="w-full">Sign In</Button>
            </a>
          </CardContent>
        </Card>
      </div>
    );
  }

  const recentAnalyses = analyses?.slice(0, 5) || [];
  const processingAnalyses = analyses?.filter((a: any) => a.status === "processing") || [];

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container flex items-center justify-between h-16">
          <div className="flex items-center gap-2">
            <Link href="/">
              <div className="flex items-center gap-2 cursor-pointer">
                <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
                  <Activity className="w-5 h-5 text-primary-foreground" />
                </div>
                <span className="font-semibold text-lg">Football Analytics</span>
              </div>
            </Link>
          </div>
          <nav className="flex items-center gap-4">
            <span className="text-sm text-muted-foreground">
              Welcome, {user?.name || "User"}
            </span>
            <Link href="/upload">
              <Button>
                <Upload className="w-4 h-4 mr-2" />
                Upload Video
              </Button>
            </Link>
          </nav>
        </div>
      </header>

      <main className="container py-8">
        {/* Stats Overview */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <StatCard
            title="Total Videos"
            value={videos?.length || 0}
            icon={<FileVideo className="w-5 h-5" />}
          />
          <StatCard
            title="Total Analyses"
            value={analyses?.length || 0}
            icon={<BarChart3 className="w-5 h-5" />}
          />
          <StatCard
            title="Processing"
            value={processingAnalyses.length}
            icon={<Loader2 className="w-5 h-5 animate-spin" />}
          />
          <StatCard
            title="Completed"
            value={analyses?.filter((a: any) => a.status === "completed").length || 0}
            icon={<CheckCircle2 className="w-5 h-5" />}
          />
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Recent Analyses */}
          <div className="lg:col-span-2">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle>Recent Analyses</CardTitle>
                  <CardDescription>Your latest video analysis jobs</CardDescription>
                </div>
              </CardHeader>
              <CardContent>
                {analysesLoading ? (
                  <div className="space-y-4">
                    {[1, 2, 3].map(i => (
                      <Skeleton key={i} className="h-20 w-full" />
                    ))}
                  </div>
                ) : recentAnalyses.length === 0 ? (
                  <div className="text-center py-8">
                    <BarChart3 className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
                    <p className="text-muted-foreground mb-4">No analyses yet</p>
                    <Link href="/upload">
                      <Button>Upload Your First Video</Button>
                    </Link>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {recentAnalyses.map((analysis: any) => (
                      <AnalysisCard key={analysis.id} analysis={analysis} />
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Videos List */}
          <div>
            <Card>
              <CardHeader>
                <CardTitle>Your Videos</CardTitle>
                <CardDescription>Uploaded match footage</CardDescription>
              </CardHeader>
              <CardContent>
                {videosLoading ? (
                  <div className="space-y-3">
                    {[1, 2, 3].map(i => (
                      <Skeleton key={i} className="h-16 w-full" />
                    ))}
                  </div>
                ) : videos?.length === 0 ? (
                  <div className="text-center py-6">
                    <FileVideo className="w-10 h-10 mx-auto text-muted-foreground mb-3" />
                    <p className="text-sm text-muted-foreground">No videos uploaded</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {videos?.slice(0, 5).map((video: any) => (
                      <div
                        key={video.id}
                        className="flex items-center justify-between p-3 rounded-lg bg-secondary/50 hover:bg-secondary transition-colors"
                      >
                        <div className="flex items-center gap-3 min-w-0">
                          <div className="w-10 h-10 rounded bg-primary/10 flex items-center justify-center flex-shrink-0">
                            <FileVideo className="w-5 h-5 text-primary" />
                          </div>
                          <div className="min-w-0">
                            <p className="font-medium truncate">{video.title}</p>
                            <p className="text-xs text-muted-foreground">
                              {formatDistanceToNow(new Date(video.createdAt), { addSuffix: true })}
                            </p>
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => handleDeleteVideo(video.id)}
                          className="flex-shrink-0"
                        >
                          <Trash2 className="w-4 h-4 text-muted-foreground hover:text-destructive" />
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}

function StatCard({ title, value, icon }: { title: string; value: number; icon: React.ReactNode }) {
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-muted-foreground">{title}</p>
            <p className="text-3xl font-bold font-mono">{value}</p>
          </div>
          <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
            {icon}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function AnalysisCard({ analysis }: { analysis: any }) {
  const statusConfig = {
    pending: { icon: <Clock className="w-4 h-4" />, color: "bg-yellow-500/10 text-yellow-500" },
    uploading: { icon: <Loader2 className="w-4 h-4 animate-spin" />, color: "bg-blue-500/10 text-blue-500" },
    processing: { icon: <Loader2 className="w-4 h-4 animate-spin" />, color: "bg-blue-500/10 text-blue-500" },
    completed: { icon: <CheckCircle2 className="w-4 h-4" />, color: "bg-green-500/10 text-green-500" },
    failed: { icon: <XCircle className="w-4 h-4" />, color: "bg-red-500/10 text-red-500" },
  };

  const status = statusConfig[analysis.status as keyof typeof statusConfig] || statusConfig.pending;
  const mode = PIPELINE_MODES[analysis.mode as PipelineMode];

  return (
    <Link href={`/analysis/${analysis.id}`}>
      <div className="flex items-center justify-between p-4 rounded-lg border border-border hover:border-primary/50 transition-colors cursor-pointer">
        <div className="flex items-center gap-4">
          <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${status.color}`}>
            {status.icon}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-medium">Analysis #{analysis.id}</span>
              <Badge variant="outline" className="text-xs">
                {mode?.name || analysis.mode}
              </Badge>
            </div>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <span>{analysis.status}</span>
              {analysis.status === "processing" && (
                <span>• {analysis.progress}%</span>
              )}
              <span>•</span>
              <span>{formatDistanceToNow(new Date(analysis.createdAt), { addSuffix: true })}</span>
            </div>
          </div>
        </div>
        <ArrowRight className="w-5 h-5 text-muted-foreground" />
      </div>
    </Link>
  );
}
