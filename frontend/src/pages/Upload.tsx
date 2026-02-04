import { useState, useCallback } from "react";
import { useAuth } from "@/_core/hooks/useAuth";
import { trpc } from "@/lib/trpc";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "sonner";
import { useLocation } from "wouter";
import { getLoginUrl } from "@/const";
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
  CheckCircle2,
} from "lucide-react";
import { Link } from "wouter";
import { PIPELINE_MODES, PipelineMode } from "@/shared/types";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Cpu, Cloud, Sparkles, Video, Camera, Lock } from "lucide-react";

const MODE_ICONS: Record<PipelineMode, React.ReactNode> = {
  all: <Layers className="w-5 h-5" />,
  radar: <Radar className="w-5 h-5" />,
  team: <Users className="w-5 h-5" />,
  track: <Target className="w-5 h-5" />,
  players: <User className="w-5 h-5" />,
  ball: <Circle className="w-5 h-5" />,
  pitch: <Map className="w-5 h-5" />,
};

export default function Upload() {
  const { user, loading: authLoading, isAuthenticated } = useAuth();
  const [, navigate] = useLocation();
  
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [selectedMode, setSelectedMode] = useState<PipelineMode>("all");
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [useCustomModels, setUseCustomModels] = useState(true);
  const [cameraType, setCameraType] = useState<"tactical" | "broadcast">("tactical");

  const uploadMutation = trpc.video.upload.useMutation();
  const createAnalysisMutation = trpc.analysis.create.useMutation();

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith("video/")) {
      setFile(droppedFile);
      if (!title) {
        setTitle(droppedFile.name.replace(/\.[^/.]+$/, ""));
      }
    } else {
      toast.error("Please upload a video file");
    }
  }, [title]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      if (!title) {
        setTitle(selectedFile.name.replace(/\.[^/.]+$/, ""));
      }
    }
  }, [title]);

  const [uploadStage, setUploadStage] = useState<"reading" | "uploading" | "processing" | "done">("reading");
  const [uploadSpeed, setUploadSpeed] = useState<number>(0);
  const [timeRemaining, setTimeRemaining] = useState<number>(0);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!file) {
      toast.error("Please select a video file");
      return;
    }
    
    if (!title.trim()) {
      toast.error("Please enter a title");
      return;
    }

    setUploading(true);
    setUploadProgress(0);
    setUploadStage("reading");
    setUploadSpeed(0);
    setTimeRemaining(0);

    try {
      // Convert file to base64 with progress
      const reader = new FileReader();
      const fileBase64 = await new Promise<string>((resolve, reject) => {
        reader.onload = () => {
          const result = reader.result as string;
          resolve(result.split(",")[1]);
        };
        reader.onerror = reject;
        reader.onprogress = (e) => {
          if (e.lengthComputable) {
            const progress = Math.round((e.loaded / e.total) * 30);
            setUploadProgress(progress);
          }
        };
        reader.readAsDataURL(file);
      });

      setUploadProgress(30);
      setUploadStage("uploading");

      // Upload with real-time progress using XMLHttpRequest
      const startTime = Date.now();
      let lastLoaded = 0;
      let lastTime = startTime;

      const uploadResult = await new Promise<{ id: number }>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        
        xhr.upload.addEventListener("progress", (e) => {
          if (e.lengthComputable) {
            const uploadPercent = (e.loaded / e.total) * 50; // 30-80% range
            setUploadProgress(30 + Math.round(uploadPercent));
            
            // Calculate upload speed
            const now = Date.now();
            const timeDiff = (now - lastTime) / 1000; // seconds
            if (timeDiff > 0.5) { // Update every 500ms
              const bytesDiff = e.loaded - lastLoaded;
              const speed = bytesDiff / timeDiff; // bytes per second
              setUploadSpeed(speed);
              
              // Estimate time remaining
              const remaining = e.total - e.loaded;
              const eta = remaining / speed;
              setTimeRemaining(Math.round(eta));
              
              lastLoaded = e.loaded;
              lastTime = now;
            }
          }
        });

        xhr.addEventListener("load", () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              const response = JSON.parse(xhr.responseText);
              resolve(response.result?.data || response);
            } catch {
              reject(new Error("Invalid response"));
            }
          } else {
            reject(new Error(`Upload failed: ${xhr.status}`));
          }
        });

        xhr.addEventListener("error", () => reject(new Error("Network error")));
        xhr.addEventListener("abort", () => reject(new Error("Upload cancelled")));

        // Use the tRPC endpoint directly
        xhr.open("POST", "/api/trpc/video.upload");
        xhr.setRequestHeader("Content-Type", "application/json");
        
        const payload = JSON.stringify({
          json: {
            title: title.trim(),
            description: description.trim() || undefined,
            fileName: file.name,
            fileSize: file.size,
            mimeType: file.type,
            fileBase64,
          }
        });
        
        xhr.send(payload);
      });

      setUploadProgress(85);
      setUploadStage("processing");
      setUploadSpeed(0);
      setTimeRemaining(0);

      // Create analysis job
      const { id: analysisId } = await createAnalysisMutation.mutateAsync({
        videoId: uploadResult.id,
        mode: selectedMode,
      });

      setUploadProgress(100);
      setUploadStage("done");
      toast.success("Video uploaded successfully! Starting analysis...");
      
      // Navigate to analysis page
      navigate(`/analysis/${analysisId}`);
    } catch (error) {
      console.error("Upload error:", error);
      toast.error("Failed to upload video. Please try again.");
    } finally {
      setUploading(false);
    }
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
            <CardDescription>Please sign in to upload videos</CardDescription>
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

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container flex items-center justify-between h-16">
          <div className="flex items-center gap-4">
            <Link href="/dashboard">
              <Button variant="ghost" size="icon">
                <ArrowLeft className="w-5 h-5" />
              </Button>
            </Link>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
                <Activity className="w-5 h-5 text-primary-foreground" />
              </div>
              <span className="font-semibold text-lg">Upload Video</span>
            </div>
          </div>
        </div>
      </header>

      <main className="container py-8">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto space-y-8">
          {/* Video Upload */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileVideo className="w-5 h-5" />
                Video File
              </CardTitle>
              <CardDescription>
                Upload football match footage for analysis (MP4, MOV, AVI supported)
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div
                className={`dropzone ${isDragging ? "active" : ""} ${file ? "border-primary" : ""}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                {file ? (
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center">
                        <FileVideo className="w-6 h-6 text-primary" />
                      </div>
                      <div className="text-left">
                        <p className="font-medium">{file.name}</p>
                        <p className="text-sm text-muted-foreground">
                          {(file.size / (1024 * 1024)).toFixed(2)} MB
                        </p>
                      </div>
                    </div>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      onClick={() => setFile(null)}
                    >
                      <X className="w-5 h-5" />
                    </Button>
                  </div>
                ) : (
                  <div className="py-8">
                    <UploadIcon className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
                    <p className="text-lg font-medium mb-2">
                      Drag and drop your video here
                    </p>
                    <p className="text-muted-foreground mb-4">
                      or click to browse files
                    </p>
                    <Input
                      type="file"
                      accept="video/*"
                      onChange={handleFileSelect}
                      className="hidden"
                      id="video-upload"
                    />
                    <Label htmlFor="video-upload">
                      <Button type="button" variant="outline" asChild>
                        <span>Browse Files</span>
                      </Button>
                    </Label>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Video Details */}
          <Card>
            <CardHeader>
              <CardTitle>Video Details</CardTitle>
              <CardDescription>
                Add a title and optional description for your video
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="title">Title *</Label>
                <Input
                  id="title"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="e.g., Arsenal vs Chelsea - Premier League"
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="description">Description</Label>
                <Textarea
                  id="description"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Add notes about the match, teams, or specific moments to analyze..."
                  rows={3}
                />
              </div>
            </CardContent>
          </Card>

          {/* Camera Type Selection */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Camera className="w-5 h-5" />
                Camera Angle
              </CardTitle>
              <CardDescription>
                Select the type of camera footage you're uploading
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-4">
                {/* Tactical/Wide Angle Option */}
                <button
                  type="button"
                  onClick={() => setCameraType("tactical")}
                  className={`relative p-4 rounded-xl border-2 text-left transition-all ${
                    cameraType === "tactical" 
                      ? "border-primary bg-primary/5" 
                      : "border-border hover:border-primary/30"
                  }`}
                >
                  {cameraType === "tactical" && (
                    <Badge className="absolute top-2 right-2 bg-primary">Selected</Badge>
                  )}
                  <div className="flex items-center gap-3 mb-3">
                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                      cameraType === "tactical" ? "bg-primary text-primary-foreground" : "bg-secondary"
                    }`}>
                      <Video className="w-5 h-5" />
                    </div>
                    <div>
                      <span className="font-semibold">Tactical View</span>
                      <Badge variant="outline" className="ml-2 text-xs">Supported</Badge>
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground mb-3">
                    Wide-angle footage showing the full pitch (DFL Bundesliga style)
                  </p>
                  <div className="space-y-1 text-xs text-muted-foreground">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-3 h-3 text-primary" />
                      <span>Full pitch visibility</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-3 h-3 text-primary" />
                      <span>Optimal for tactical analysis</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-3 h-3 text-primary" />
                      <span>Accurate homography</span>
                    </div>
                  </div>
                </button>

                {/* Broadcast Angle Option - Coming Soon */}
                <button
                  type="button"
                  disabled
                  className="relative p-4 rounded-xl border-2 text-left transition-all border-border opacity-60 cursor-not-allowed"
                >
                  <Badge className="absolute top-2 right-2 bg-amber-500/80 text-white">Coming Soon</Badge>
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-10 h-10 rounded-lg flex items-center justify-center bg-secondary">
                      <Camera className="w-5 h-5" />
                    </div>
                    <div>
                      <span className="font-semibold">Broadcast View</span>
                      <Lock className="w-3 h-3 ml-2 inline text-muted-foreground" />
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground mb-3">
                    Standard TV broadcast camera angles with dynamic movement
                  </p>
                  <div className="space-y-1 text-xs text-muted-foreground">
                    <div className="flex items-center gap-2">
                      <Lock className="w-3 h-3" />
                      <span>Requires different models</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Lock className="w-3 h-3" />
                      <span>Dynamic camera tracking</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Lock className="w-3 h-3" />
                      <span>Under development</span>
                    </div>
                  </div>
                </button>
              </div>
            </CardContent>
          </Card>

          {/* Model Selection */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="w-5 h-5" />
                Detection Models
              </CardTitle>
              <CardDescription>
                Choose between custom-trained models or cloud API for ball and pitch detection
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-4">
                {/* Custom Models Option */}
                <button
                  type="button"
                  onClick={() => setUseCustomModels(true)}
                  className={`relative p-4 rounded-xl border-2 text-left transition-all ${
                    useCustomModels 
                      ? "border-primary bg-primary/5" 
                      : "border-border hover:border-primary/30"
                  }`}
                >
                  {useCustomModels && (
                    <Badge className="absolute top-2 right-2 bg-primary">Selected</Badge>
                  )}
                  <div className="flex items-center gap-3 mb-3">
                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                      useCustomModels ? "bg-primary text-primary-foreground" : "bg-secondary"
                    }`}>
                      <Cpu className="w-5 h-5" />
                    </div>
                    <div>
                      <span className="font-semibold">Custom Models</span>
                      <Badge variant="outline" className="ml-2 text-xs">Recommended</Badge>
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground mb-3">
                    Use your custom-trained YOLOv8 models for faster local inference
                  </p>
                  <div className="space-y-1 text-xs text-muted-foreground">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-3 h-3 text-primary" />
                      <span>player_detection.pt</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-3 h-3 text-primary" />
                      <span>ball_detection.pt</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-3 h-3 text-primary" />
                      <span>pitch_detection.pt</span>
                    </div>
                  </div>
                </button>

                {/* Roboflow API Option */}
                <button
                  type="button"
                  onClick={() => setUseCustomModels(false)}
                  className={`relative p-4 rounded-xl border-2 text-left transition-all ${
                    !useCustomModels 
                      ? "border-primary bg-primary/5" 
                      : "border-border hover:border-primary/30"
                  }`}
                >
                  {!useCustomModels && (
                    <Badge className="absolute top-2 right-2 bg-primary">Selected</Badge>
                  )}
                  <div className="flex items-center gap-3 mb-3">
                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                      !useCustomModels ? "bg-primary text-primary-foreground" : "bg-secondary"
                    }`}>
                      <Cloud className="w-5 h-5" />
                    </div>
                    <div>
                      <span className="font-semibold">Roboflow API</span>
                      <Badge variant="secondary" className="ml-2 text-xs">Fallback</Badge>
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground mb-3">
                    Cloud-based detection, no local GPU required
                  </p>
                  <div className="space-y-1 text-xs text-muted-foreground">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-3 h-3 text-muted-foreground" />
                      <span>Pre-trained pitch keypoints</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-3 h-3 text-muted-foreground" />
                      <span>API rate limits apply</span>
                    </div>
                  </div>
                </button>
              </div>
            </CardContent>
          </Card>

          {/* Pipeline Mode Selection */}
          <Card>
            <CardHeader>
              <CardTitle>Analysis Mode</CardTitle>
              <CardDescription>
                Select the type of analysis to run on your video
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {(Object.entries(PIPELINE_MODES) as [PipelineMode, typeof PIPELINE_MODES[PipelineMode]][]).map(
                  ([mode, config]) => (
                    <button
                      key={mode}
                      type="button"
                      onClick={() => setSelectedMode(mode)}
                      className={`mode-card text-left ${selectedMode === mode ? "selected" : ""}`}
                    >
                      <div className="flex items-start gap-3">
                        <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                          selectedMode === mode ? "bg-primary text-primary-foreground" : "bg-secondary text-foreground"
                        }`}>
                          {MODE_ICONS[mode]}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="font-medium">{config.name}</span>
                            {selectedMode === mode && (
                              <CheckCircle2 className="w-4 h-4 text-primary" />
                            )}
                          </div>
                          <p className="text-sm text-muted-foreground line-clamp-2">
                            {config.description}
                          </p>
                        </div>
                      </div>
                    </button>
                  )
                )}
              </div>
            </CardContent>
          </Card>

          {/* Submit */}
          <div className="flex items-center justify-between">
            <Link href="/dashboard">
              <Button type="button" variant="outline">
                Cancel
              </Button>
            </Link>
            <Button type="submit" disabled={!file || !title.trim() || uploading} size="lg">
              {uploading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Uploading... {uploadProgress}%
                </>
              ) : (
                <>
                  <UploadIcon className="w-4 h-4 mr-2" />
                  Start Analysis
                </>
              )}
            </Button>
          </div>

          {/* Upload Progress */}
          {uploading && (
            <Card className="border-primary/50 bg-primary/5">
              <CardContent className="pt-6">
                <div className="space-y-4">
                  {/* Stage indicator */}
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin text-primary" />
                      <span className="font-medium">
                        {uploadStage === "reading" && "Reading file..."}
                        {uploadStage === "uploading" && "Uploading to server..."}
                        {uploadStage === "processing" && "Creating analysis job..."}
                        {uploadStage === "done" && "Complete!"}
                      </span>
                    </div>
                    <span className="font-mono text-primary">{uploadProgress}%</span>
                  </div>
                  
                  {/* Progress bar */}
                  <div className="relative h-3 bg-secondary rounded-full overflow-hidden">
                    <div 
                      className="absolute inset-y-0 left-0 bg-gradient-to-r from-primary to-primary/80 rounded-full transition-all duration-300 ease-out"
                      style={{ width: `${uploadProgress}%` }}
                    />
                    <div 
                      className="absolute inset-y-0 left-0 bg-white/20 rounded-full animate-pulse"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                  
                  {/* Stats row */}
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <div className="flex items-center gap-4">
                      {uploadSpeed > 0 && (
                        <span>
                          Speed: {uploadSpeed > 1024 * 1024 
                            ? `${(uploadSpeed / (1024 * 1024)).toFixed(1)} MB/s`
                            : `${(uploadSpeed / 1024).toFixed(0)} KB/s`
                          }
                        </span>
                      )}
                      {file && (
                        <span>
                          {((file.size * uploadProgress / 100) / (1024 * 1024)).toFixed(1)} / {(file.size / (1024 * 1024)).toFixed(1)} MB
                        </span>
                      )}
                    </div>
                    {timeRemaining > 0 && uploadStage === "uploading" && (
                      <span>
                        {timeRemaining > 60 
                          ? `~${Math.floor(timeRemaining / 60)}m ${timeRemaining % 60}s remaining`
                          : `~${timeRemaining}s remaining`
                        }
                      </span>
                    )}
                  </div>
                  
                  {/* Stage steps */}
                  <div className="flex items-center justify-between pt-2">
                    {["reading", "uploading", "processing", "done"].map((stage, i) => (
                      <div key={stage} className="flex items-center">
                        <div className={`w-2 h-2 rounded-full ${
                          ["reading", "uploading", "processing", "done"].indexOf(uploadStage) >= i
                            ? "bg-primary"
                            : "bg-muted"
                        }`} />
                        {i < 3 && (
                          <div className={`w-16 h-0.5 ${
                            ["reading", "uploading", "processing", "done"].indexOf(uploadStage) > i
                              ? "bg-primary"
                              : "bg-muted"
                          }`} />
                        )}
                      </div>
                    ))}
                  </div>
                  <div className="flex items-center justify-between text-[10px] text-muted-foreground -mt-2">
                    <span>Read</span>
                    <span>Upload</span>
                    <span>Process</span>
                    <span>Done</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </form>
      </main>
    </div>
  );
}
