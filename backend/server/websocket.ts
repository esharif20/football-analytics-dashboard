/**
 * WebSocket server for real-time progress updates
 * Replaces polling with instant push notifications
 */
import { WebSocketServer, WebSocket } from "ws";
import type { Server } from "http";

// Message types for WebSocket communication
export interface WSMessage {
  type: "progress" | "status" | "complete" | "error" | "connected";
  analysisId?: number;
  data?: {
    status?: string;
    progress?: number;
    currentStage?: string;
    eta?: number;
    error?: string;
    result?: unknown;
  };
}

// Store connected clients by analysis ID they're watching
const clients = new Map<number, Set<WebSocket>>();
const allClients = new Set<WebSocket>();

let wss: WebSocketServer | null = null;

/**
 * Initialize WebSocket server attached to HTTP server
 */
export function initWebSocket(server: Server): WebSocketServer {
  wss = new WebSocketServer({ server, path: "/ws" });

  wss.on("connection", (ws: WebSocket) => {
    allClients.add(ws);
    
    // Send connected confirmation
    ws.send(JSON.stringify({ type: "connected" } as WSMessage));

    ws.on("message", (data: Buffer) => {
      try {
        const message = JSON.parse(data.toString());
        
        // Handle subscription to analysis updates
        if (message.type === "subscribe" && message.analysisId) {
          const analysisId = message.analysisId;
          if (!clients.has(analysisId)) {
            clients.set(analysisId, new Set());
          }
          clients.get(analysisId)!.add(ws);
          
          // Confirm subscription
          ws.send(JSON.stringify({
            type: "status",
            analysisId,
            data: { status: "subscribed" }
          } as WSMessage));
        }
        
        // Handle unsubscription
        if (message.type === "unsubscribe" && message.analysisId) {
          const analysisId = message.analysisId;
          clients.get(analysisId)?.delete(ws);
        }
      } catch (e) {
        console.error("WebSocket message parse error:", e);
      }
    });

    ws.on("close", () => {
      allClients.delete(ws);
      // Remove from all analysis subscriptions
      clients.forEach((clientSet) => {
        clientSet.delete(ws);
      });
    });

    ws.on("error", (error) => {
      console.error("WebSocket error:", error);
      allClients.delete(ws);
    });
  });

  console.log("WebSocket server initialized on /ws");
  return wss;
}

/**
 * Broadcast progress update to all clients watching an analysis
 */
export function broadcastProgress(
  analysisId: number,
  status: string,
  progress: number,
  currentStage: string,
  eta?: number
): void {
  const message: WSMessage = {
    type: "progress",
    analysisId,
    data: {
      status,
      progress,
      currentStage,
      eta,
    },
  };

  const messageStr = JSON.stringify(message);
  const subscribers = clients.get(analysisId);
  
  if (subscribers) {
    subscribers.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(messageStr);
      }
    });
  }
}

/**
 * Broadcast completion to all clients watching an analysis
 */
export function broadcastComplete(analysisId: number, result?: unknown): void {
  const message: WSMessage = {
    type: "complete",
    analysisId,
    data: {
      status: "completed",
      progress: 100,
      currentStage: "done",
      result,
    },
  };

  const messageStr = JSON.stringify(message);
  const subscribers = clients.get(analysisId);
  
  if (subscribers) {
    subscribers.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(messageStr);
      }
    });
  }
  
  // Clean up subscriptions for completed analysis
  clients.delete(analysisId);
}

/**
 * Broadcast error to all clients watching an analysis
 */
export function broadcastError(analysisId: number, error: string): void {
  const message: WSMessage = {
    type: "error",
    analysisId,
    data: {
      status: "failed",
      error,
    },
  };

  const messageStr = JSON.stringify(message);
  const subscribers = clients.get(analysisId);
  
  if (subscribers) {
    subscribers.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(messageStr);
      }
    });
  }
  
  // Clean up subscriptions for failed analysis
  clients.delete(analysisId);
}

/**
 * Get count of connected clients
 */
export function getClientCount(): number {
  return allClients.size;
}

/**
 * Get count of subscribers for an analysis
 */
export function getSubscriberCount(analysisId: number): number {
  return clients.get(analysisId)?.size || 0;
}
