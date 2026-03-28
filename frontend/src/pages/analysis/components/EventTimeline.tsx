import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { EVENT_TYPES } from "../context";

export function EventTimeline({ events }: { events: any[] }) {
  const maxTimestamp = Math.max(...events.map(e => e.timestamp || 0), 1);
  return (
    <div className="space-y-4">
      <div className="relative h-12 bg-secondary/30 rounded-xl overflow-hidden border border-border/20">
        {events.map((event, i) => {
          const position = (event.timestamp / maxTimestamp) * 100;
          const eventConfig = EVENT_TYPES[event.type as keyof typeof EVENT_TYPES];
          return <div key={i} className="event-marker" style={{ left: `${Math.min(position, 98)}%`, backgroundColor: eventConfig?.color || "#666" }} title={`${event.type} at ${event.timestamp.toFixed(1)}s`} />;
        })}
      </div>
      <ScrollArea className="h-48">
        <div className="space-y-2">
          {events.slice(0, 10).map((event, i) => {
            const eventConfig = EVENT_TYPES[event.type as keyof typeof EVENT_TYPES];
            return (
              <div key={i} className="flex items-center gap-3 p-3 rounded-xl bg-secondary/20 border border-border/10 hover:border-border/30 hover:bg-secondary/30 transition-all duration-300">
                <div className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: eventConfig?.color || "#666" }} />
                <div className="flex-1 min-w-0">
                  <span className="font-medium capitalize text-sm">{event.type}</span>
                  <span className="text-muted-foreground text-xs ml-2">Team {event.teamId} &middot; {event.timestamp}s</span>
                </div>
                <Badge variant="outline" className={`text-xs shrink-0 ${event.success ? "border-emerald-500/30 text-emerald-400 bg-emerald-500/5" : "border-border/30 text-muted-foreground"}`}>
                  {event.success ? "Success" : "Failed"}
                </Badge>
              </div>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
}
