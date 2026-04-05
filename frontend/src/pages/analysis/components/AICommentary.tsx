import { CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Button } from '@/components/ui/button'
import { Sparkles, TrendingUp, MessageSquare, Bot, Loader2 } from 'lucide-react'
import { Streamdown } from 'streamdown'
import { AIChatBox } from '@/components/AIChatBox'
import type { Message } from '@/components/AIChatBox'

export function AICommentarySection({
  aiTab,
  setAiTab,
  commentaryList,
  generateCommentaryMutation,
  handleGenerateCommentary,
  streamingContent,
  streamingType,
  chatMessages,
  isChatLoading,
  onSendChatMessage,
}: {
  aiTab: string
  setAiTab: (t: string) => void
  commentaryList: any
  generateCommentaryMutation: any
  handleGenerateCommentary: (type: 'match_summary' | 'tactical_analysis') => void
  streamingContent: string
  streamingType: string | null
  chatMessages: Message[]
  isChatLoading: boolean
  onSendChatMessage: (content: string) => void
}) {
  // Determine if we're currently streaming for the active tab
  const activeType = aiTab === 'tactical' ? 'tactical_analysis' : 'match_summary'
  const isStreaming = streamingType === activeType && streamingContent.length > 0
  const isPending =
    generateCommentaryMutation.isPending || (streamingType === activeType && !isStreaming)

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
            <CardDescription className="text-xs mt-0.5">
              Tactical analysis grounded in tracking data
            </CardDescription>
          </div>
        </div>
        {/* AI Tabs */}
        <div className="flex gap-1 bg-black/20 rounded-xl p-1 border border-white/[0.04]">
          {[
            {
              id: 'tactical',
              label: 'Tactical Analysis',
              icon: <TrendingUp className="w-3.5 h-3.5" />,
            },
            {
              id: 'commentary',
              label: 'Commentary',
              icon: <MessageSquare className="w-3.5 h-3.5" />,
            },
            {
              id: 'chat',
              label: 'Chat Agent',
              icon: <Bot className="w-3.5 h-3.5" />,
            },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setAiTab(tab.id)}
              className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-2 rounded-lg text-[11px] font-medium transition-all duration-300 relative ${
                aiTab === tab.id
                  ? 'bg-violet-500/15 text-violet-400 shadow-[0_0_10px_rgba(139,92,246,0.1)] border border-violet-500/20'
                  : 'text-muted-foreground hover:text-foreground hover:bg-white/[0.04]'
              }`}
            >
              {tab.icon}
              <span className="hidden sm:inline">{tab.label}</span>
            </button>
          ))}
        </div>
      </CardHeader>
      <CardContent className="p-4">
        {aiTab === 'chat' ? (
          <AIChatBox
            messages={chatMessages}
            onSendMessage={onSendChatMessage}
            isLoading={isChatLoading}
            height="350px"
            placeholder="Ask about tactics, players, pressing..."
            emptyStateMessage="Ask anything about this match"
            suggestedPrompts={[
              'What were the key tactical patterns?',
              'Which players covered the most distance?',
              'Analyze the pressing intensity',
              'Who dominated possession and how?',
            ]}
          />
        ) : (
          <>
            {/* Streaming preview — shows text as it arrives */}
            {isStreaming && (
              <div className="mb-3 p-4 rounded-xl bg-violet-500/5 border border-violet-500/20 animate-pulse-subtle">
                <div className="flex items-center gap-2 mb-2">
                  <Loader2 className="w-3.5 h-3.5 text-violet-400 animate-spin" />
                  <span className="text-xs text-violet-400 font-medium">Generating…</span>
                </div>
                <div className="text-sm prose prose-sm dark:prose-invert max-w-none leading-relaxed">
                  <Streamdown>{streamingContent}</Streamdown>
                </div>
              </div>
            )}

            {commentaryList && commentaryList.length > 0 ? (
              <ScrollArea className="h-[350px]">
                <div className="space-y-3">
                  {commentaryList
                    .filter((c: any) =>
                      aiTab === 'tactical'
                        ? c.type === 'tactical_analysis' || c.type === 'tactical_deep_dive'
                        : c.type === 'match_summary' || c.type === 'match_overview'
                    )
                    .map((c: any) => (
                      <div
                        key={c.id}
                        className="p-4 rounded-xl bg-secondary/30 border border-border/20 hover:border-violet-500/20 transition-all duration-300"
                      >
                        <div className="flex items-center gap-2 mb-2">
                          <Badge
                            variant="outline"
                            className="text-xs border-violet-500/30 text-violet-400"
                          >
                            {c.type}
                          </Badge>
                          {c.groundingData?.vision_augmented && (
                            <Badge
                              variant="outline"
                              className="text-xs border-emerald-500/30 text-emerald-400"
                            >
                              vision
                            </Badge>
                          )}
                          {c.groundingData?.provider && (
                            <span className="text-[10px] text-muted-foreground/50 ml-auto">
                              {c.groundingData.provider.replace('Provider', '')}
                            </span>
                          )}
                        </div>
                        <div className="text-sm prose prose-sm dark:prose-invert max-w-none leading-relaxed">
                          <Streamdown>{c.content}</Streamdown>
                        </div>
                      </div>
                    ))}
                  {!isStreaming &&
                    commentaryList.filter((c: any) =>
                      aiTab === 'tactical'
                        ? c.type === 'tactical_analysis' || c.type === 'tactical_deep_dive'
                        : c.type === 'match_summary' || c.type === 'match_overview'
                    ).length === 0 && (
                      <EmptyCommentaryState
                        type={activeType}
                        handleGenerate={handleGenerateCommentary}
                        isPending={isPending}
                      />
                    )}
                </div>
              </ScrollArea>
            ) : (
              !isStreaming && (
                <EmptyCommentaryState
                  type={activeType}
                  handleGenerate={handleGenerateCommentary}
                  isPending={isPending}
                />
              )
            )}
          </>
        )}
      </CardContent>
    </div>
  )
}

export function EmptyCommentaryState({
  type,
  handleGenerate,
  isPending,
}: {
  type: 'match_summary' | 'tactical_analysis'
  handleGenerate: (t: 'match_summary' | 'tactical_analysis') => void
  isPending: boolean
}) {
  return (
    <div className="space-y-4">
      <div className="text-center py-6">
        <div className="w-12 h-12 rounded-2xl bg-violet-500/5 border border-violet-500/10 flex items-center justify-center mx-auto mb-3">
          {type === 'tactical_analysis' ? (
            <TrendingUp className="w-6 h-6 text-violet-400/60" />
          ) : (
            <MessageSquare className="w-6 h-6 text-violet-400/60" />
          )}
        </div>
        <p className="text-sm text-muted-foreground">
          {type === 'tactical_analysis'
            ? 'Generate AI-powered tactical breakdown based on tracking data.'
            : 'Generate a match summary with key moments and insights.'}
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
            <span className="text-sm font-medium">
              Generate {type === 'tactical_analysis' ? 'Tactical Analysis' : 'Match Summary'}
            </span>
          </>
        )}
      </Button>
    </div>
  )
}
