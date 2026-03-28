import { CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import {
  Sparkles,
  TrendingUp,
  MessageSquare,
  Bot,
  Lock,
  Loader2,
} from "lucide-react";
import { Streamdown } from "streamdown";

export function AICommentarySection({
  aiTab, setAiTab, commentaryList, generateCommentaryMutation, handleGenerateCommentary,
}: {
  aiTab: string;
  setAiTab: (t: string) => void;
  commentaryList: any;
  generateCommentaryMutation: any;
  handleGenerateCommentary: (type: "match_summary" | "tactical_analysis") => void;
}) {
  return (
    <div className="glass-card overflow-hidden h-full hover-lift group">
      <div className="absolute -top-10 -right-10 w-32 h-32 rounded-full bg-violet-500/5 blur-3xl group-hover:bg-violet-500/10 transition-colors duration-700" />
      <CardHeader className="pb-0 pt-5 px-5 relative">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-9 h-9 rounded-xl bg-violet-500/10 flex items-center justify-center border border-violet-500/20">
            <Sparkles className="w-4 h-4 text-violet-400" />
          </div>
          <div>
            <CardTitle className="text-sm font-semibold">AI Commentary</CardTitle>
            <CardDescription className="text-xs mt-0.5">Tactical analysis grounded in tracking data</CardDescription>
          </div>
        </div>
        {/* AI Tabs */}
        <div className="flex gap-1 bg-black/20 rounded-xl p-1 border border-white/[0.04]">
          {[
            { id: "tactical", label: "Tactical Analysis", icon: <TrendingUp className="w-3.5 h-3.5" /> },
            { id: "commentary", label: "Commentary", icon: <MessageSquare className="w-3.5 h-3.5" /> },
            { id: "chat", label: "Chat Agent", icon: <Bot className="w-3.5 h-3.5" />, comingSoon: true },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => !tab.comingSoon && setAiTab(tab.id)}
              className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-2 rounded-lg text-[11px] font-medium transition-all duration-300 relative ${
                aiTab === tab.id && !tab.comingSoon
                  ? "bg-violet-500/15 text-violet-400 shadow-[0_0_10px_rgba(139,92,246,0.1)] border border-violet-500/20"
                  : tab.comingSoon
                  ? "text-muted-foreground/50 cursor-not-allowed"
                  : "text-muted-foreground hover:text-foreground hover:bg-white/[0.04]"
              }`}
            >
              {tab.icon}
              <span className="hidden sm:inline">{tab.label}</span>
              {tab.comingSoon && (
                <span className="absolute -top-1.5 -right-1 px-1 py-0.5 rounded text-[7px] font-bold bg-amber-500/20 text-amber-400 border border-amber-500/20 leading-none">
                  SOON
                </span>
              )}
            </button>
          ))}
        </div>
      </CardHeader>
      <CardContent className="p-4">
        {aiTab === "chat" ? (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <div className="w-16 h-16 rounded-2xl bg-amber-500/5 border border-amber-500/10 flex items-center justify-center mb-4 relative">
              <Bot className="w-8 h-8 text-amber-400/60" />
              <Lock className="w-4 h-4 text-amber-400/80 absolute -bottom-1 -right-1" />
            </div>
            <h4 className="font-semibold text-sm mb-1">Chat Agent</h4>
            <p className="text-xs text-muted-foreground max-w-[200px]">
              Interactive AI chat for real-time tactical Q&amp;A is coming soon.
            </p>
            <Badge variant="outline" className="mt-3 text-[10px] border-amber-400/20 text-amber-400/70 bg-amber-500/5">
              Coming Soon
            </Badge>
          </div>
        ) : (
          <>
            {commentaryList && commentaryList.length > 0 ? (
              <ScrollArea className="h-[350px]">
                <div className="space-y-3">
                  {commentaryList
                    .filter((c: any) =>
                      aiTab === "tactical"
                        ? c.type === "tactical_analysis"
                        : c.type === "match_summary"
                    )
                    .map((c: any) => (
                      <div key={c.id} className="p-4 rounded-xl bg-secondary/30 border border-border/20 hover:border-violet-500/20 transition-all duration-300">
                        <Badge variant="outline" className="mb-2 text-xs border-violet-500/30 text-violet-400">{c.type}</Badge>
                        <div className="text-sm prose prose-sm dark:prose-invert max-w-none leading-relaxed">
                          <Streamdown>{c.content}</Streamdown>
                        </div>
                      </div>
                    ))}
                  {commentaryList.filter((c: any) =>
                    aiTab === "tactical" ? c.type === "tactical_analysis" : c.type === "match_summary"
                  ).length === 0 && (
                    <EmptyCommentaryState
                      type={aiTab === "tactical" ? "tactical_analysis" : "match_summary"}
                      handleGenerate={handleGenerateCommentary}
                      isPending={generateCommentaryMutation.isPending}
                    />
                  )}
                </div>
              </ScrollArea>
            ) : (
              <EmptyCommentaryState
                type={aiTab === "tactical" ? "tactical_analysis" : "match_summary"}
                handleGenerate={handleGenerateCommentary}
                isPending={generateCommentaryMutation.isPending}
              />
            )}
          </>
        )}
      </CardContent>
    </div>
  );
}

export function EmptyCommentaryState({
  type, handleGenerate, isPending,
}: {
  type: "match_summary" | "tactical_analysis";
  handleGenerate: (t: "match_summary" | "tactical_analysis") => void;
  isPending: boolean;
}) {
  return (
    <div className="space-y-4">
      <div className="text-center py-6">
        <div className="w-12 h-12 rounded-2xl bg-violet-500/5 border border-violet-500/10 flex items-center justify-center mx-auto mb-3">
          {type === "tactical_analysis" ? (
            <TrendingUp className="w-6 h-6 text-violet-400/60" />
          ) : (
            <MessageSquare className="w-6 h-6 text-violet-400/60" />
          )}
        </div>
        <p className="text-sm text-muted-foreground">
          {type === "tactical_analysis"
            ? "Generate AI-powered tactical breakdown based on tracking data."
            : "Generate a match summary with key moments and insights."}
        </p>
      </div>
      <Button
        variant="outline"
        className="w-full justify-center gap-2 h-11 rounded-xl border-violet-500/20 hover:border-violet-500/40 hover:bg-violet-500/5 transition-all duration-300 group"
        onClick={() => handleGenerate(type)}
        disabled={isPending}
      >
        {isPending ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : (
          <>
            <Sparkles className="w-4 h-4 text-violet-400 group-hover:scale-110 transition-transform duration-300" />
            <span className="text-sm font-medium">Generate {type === "tactical_analysis" ? "Tactical Analysis" : "Match Summary"}</span>
          </>
        )}
      </Button>
    </div>
  );
}
