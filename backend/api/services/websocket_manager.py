"""
WebSocket Connection Manager for Real-time Updates
"""
from fastapi import WebSocket
from typing import Dict, Set, Any
import json
import asyncio

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # client_id -> set of topics
        self.topic_subscribers: Dict[str, Set[str]] = {}  # topic -> set of client_ids
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = set()
        print(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        # Clean up subscriptions
        if client_id in self.subscriptions:
            for topic in self.subscriptions[client_id]:
                if topic in self.topic_subscribers:
                    self.topic_subscribers[topic].discard(client_id)
            del self.subscriptions[client_id]
        
        print(f"Client {client_id} disconnected")
    
    def subscribe(self, client_id: str, topic: str):
        if client_id in self.subscriptions:
            self.subscriptions[client_id].add(topic)
        
        if topic not in self.topic_subscribers:
            self.topic_subscribers[topic] = set()
        self.topic_subscribers[topic].add(client_id)
        
        print(f"Client {client_id} subscribed to {topic}")
    
    def unsubscribe(self, client_id: str, topic: str):
        if client_id in self.subscriptions:
            self.subscriptions[client_id].discard(topic)
        
        if topic in self.topic_subscribers:
            self.topic_subscribers[topic].discard(client_id)
        
        print(f"Client {client_id} unsubscribed from {topic}")
    
    async def send_personal_message(self, message: Any, client_id: str):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                print(f"Error sending to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast_to_topic(self, topic: str, message: Any):
        """Broadcast message to all subscribers of a topic"""
        if topic not in self.topic_subscribers:
            return
        
        disconnected = []
        for client_id in self.topic_subscribers[topic]:
            if client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_json(message)
                except Exception as e:
                    print(f"Error broadcasting to {client_id}: {e}")
                    disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)
    
    async def broadcast_analysis_progress(self, analysis_id: int, status: str, 
                                          progress: int, current_stage: str = None,
                                          eta_seconds: float = None):
        """Broadcast analysis progress update to subscribers"""
        topic = f"analysis:{analysis_id}"
        message = {
            "type": "progress",
            "analysisId": analysis_id,
            "status": status,
            "progress": progress,
            "currentStage": current_stage,
            "etaSeconds": eta_seconds,
            "timestamp": asyncio.get_event_loop().time()
        }
        await self.broadcast_to_topic(topic, message)
    
    async def broadcast_analysis_complete(self, analysis_id: int, results: Dict):
        """Broadcast analysis completion to subscribers"""
        topic = f"analysis:{analysis_id}"
        message = {
            "type": "complete",
            "analysisId": analysis_id,
            "results": results,
            "timestamp": asyncio.get_event_loop().time()
        }
        await self.broadcast_to_topic(topic, message)
    
    async def broadcast_analysis_error(self, analysis_id: int, error: str):
        """Broadcast analysis error to subscribers"""
        topic = f"analysis:{analysis_id}"
        message = {
            "type": "error",
            "analysisId": analysis_id,
            "error": error,
            "timestamp": asyncio.get_event_loop().time()
        }
        await self.broadcast_to_topic(topic, message)

# Global manager instance
manager = ConnectionManager()
