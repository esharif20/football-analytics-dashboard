#!/usr/bin/env python3
"""
Generate demo tracking data for Test6.mp4

This creates pre-processed sample data so users can explore
the dashboard visualizations without waiting for processing.
"""

import json
import random
import math
from pathlib import Path
from datetime import datetime

# Output paths
OUTPUT_DIR = Path(__file__).parent / "demo_data"
OUTPUT_DIR.mkdir(exist_ok=True)

# Video parameters (based on Test6.mp4)
FPS = 25
DURATION = 30  # seconds
TOTAL_FRAMES = FPS * DURATION

# Pitch dimensions (in meters, standard football pitch)
PITCH_LENGTH = 105
PITCH_WIDTH = 68


def generate_player_positions(frame: int, team: int) -> list:
    """Generate realistic player positions for a team."""
    positions = []
    
    # Base formation (4-3-3)
    if team == 1:
        # Team 1 attacking left to right
        base_positions = [
            (10, 34),   # GK
            (25, 10), (25, 25), (25, 43), (25, 58),  # Defenders
            (45, 20), (45, 34), (45, 48),  # Midfielders
            (70, 15), (70, 34), (70, 53),  # Forwards
        ]
    else:
        # Team 2 attacking right to left
        base_positions = [
            (95, 34),   # GK
            (80, 10), (80, 25), (80, 43), (80, 58),  # Defenders
            (60, 20), (60, 34), (60, 48),  # Midfielders
            (35, 15), (35, 34), (35, 53),  # Forwards
        ]
    
    # Add movement based on frame
    t = frame / TOTAL_FRAMES
    wave = math.sin(t * 2 * math.pi * 3)  # Oscillating movement
    
    for i, (x, y) in enumerate(base_positions):
        # Add some random movement
        dx = wave * 5 + random.uniform(-2, 2)
        dy = random.uniform(-3, 3)
        
        # Constrain to pitch
        new_x = max(0, min(PITCH_LENGTH, x + dx))
        new_y = max(0, min(PITCH_WIDTH, y + dy))
        
        positions.append({
            "id": i + 1 + (10 if team == 2 else 0),
            "x": round(new_x, 2),
            "y": round(new_y, 2),
            "team_id": team,
            "confidence": round(random.uniform(0.85, 0.99), 3),
            "speed": round(random.uniform(0, 8), 2),  # m/s
        })
    
    return positions


def generate_ball_position(frame: int) -> dict:
    """Generate ball position with realistic movement."""
    t = frame / TOTAL_FRAMES
    
    # Ball moves across the pitch with some variation
    base_x = 20 + t * 65 + math.sin(t * 10) * 15
    base_y = 34 + math.sin(t * 7) * 20
    
    # Occasionally "lose" the ball (simulating occlusion)
    if random.random() < 0.05:
        return None
    
    return {
        "x": round(max(0, min(PITCH_LENGTH, base_x)), 2),
        "y": round(max(0, min(PITCH_WIDTH, base_y)), 2),
        "confidence": round(random.uniform(0.7, 0.95), 3),
        "speed": round(random.uniform(0, 25), 2),  # m/s
    }


def generate_events() -> list:
    """Generate sample events (passes, shots, etc.)."""
    events = []
    event_types = ["pass", "shot", "challenge", "interception", "possession_change"]
    
    for i in range(50):  # Generate 50 events
        frame = random.randint(0, TOTAL_FRAMES - 1)
        event_type = random.choice(event_types)
        team = random.choice([1, 2])
        player = random.randint(1, 11) + (10 if team == 2 else 0)
        
        event = {
            "id": i + 1,
            "type": event_type,
            "frame_number": frame,
            "timestamp": round(frame / FPS, 2),
            "player_id": player,
            "team_id": team,
            "start_x": round(random.uniform(0, PITCH_LENGTH), 2),
            "start_y": round(random.uniform(0, PITCH_WIDTH), 2),
            "confidence": round(random.uniform(0.7, 0.95), 3),
        }
        
        if event_type == "pass":
            event["target_player_id"] = random.randint(1, 11) + (10 if team == 2 else 0)
            event["end_x"] = round(random.uniform(0, PITCH_LENGTH), 2)
            event["end_y"] = round(random.uniform(0, PITCH_WIDTH), 2)
            event["success"] = random.random() > 0.2
        elif event_type == "shot":
            event["end_x"] = 0 if team == 2 else PITCH_LENGTH
            event["end_y"] = round(random.uniform(25, 43), 2)
            event["success"] = random.random() > 0.7
        
        events.append(event)
    
    # Sort by frame number
    events.sort(key=lambda e: e["frame_number"])
    return events


def generate_statistics() -> dict:
    """Generate match statistics."""
    possession_1 = random.randint(40, 60)
    
    return {
        "possession": {
            "team1": possession_1,
            "team2": 100 - possession_1,
        },
        "passes": {
            "team1": random.randint(150, 250),
            "team2": random.randint(150, 250),
        },
        "pass_accuracy": {
            "team1": round(random.uniform(0.75, 0.90), 2),
            "team2": round(random.uniform(0.75, 0.90), 2),
        },
        "shots": {
            "team1": random.randint(5, 15),
            "team2": random.randint(5, 15),
        },
        "distance_covered": {
            "team1": round(random.uniform(8000, 12000), 0),  # meters
            "team2": round(random.uniform(8000, 12000), 0),
        },
        "avg_speed": {
            "team1": round(random.uniform(5, 7), 2),  # m/s
            "team2": round(random.uniform(5, 7), 2),
        },
    }


def generate_heatmap_data(team: int) -> list:
    """Generate heatmap data for a team."""
    heatmap = []
    grid_size = 10
    
    for x in range(0, PITCH_LENGTH, grid_size):
        for y in range(0, PITCH_WIDTH, grid_size):
            # Higher intensity in team's attacking half
            if team == 1:
                intensity = 0.3 + 0.7 * (x / PITCH_LENGTH)
            else:
                intensity = 0.3 + 0.7 * (1 - x / PITCH_LENGTH)
            
            intensity *= random.uniform(0.5, 1.5)
            intensity = min(1.0, max(0, intensity))
            
            heatmap.append({
                "x": x + grid_size / 2,
                "y": y + grid_size / 2,
                "intensity": round(intensity, 3),
            })
    
    return heatmap


def generate_pass_network(team: int) -> dict:
    """Generate pass network data for a team."""
    players = list(range(1, 12)) if team == 1 else list(range(11, 22))
    
    nodes = []
    for p in players:
        nodes.append({
            "id": p,
            "passes_made": random.randint(10, 50),
            "passes_received": random.randint(10, 50),
            "x": round(random.uniform(10, 90), 2),
            "y": round(random.uniform(10, 58), 2),
        })
    
    edges = []
    for i, p1 in enumerate(players):
        for p2 in players[i+1:]:
            if random.random() > 0.5:
                edges.append({
                    "source": p1,
                    "target": p2,
                    "weight": random.randint(1, 15),
                    "success_rate": round(random.uniform(0.6, 0.95), 2),
                })
    
    return {"nodes": nodes, "edges": edges}


def main():
    print("Generating demo data for Test6.mp4...")
    
    # Generate tracking data (sample every 5 frames to reduce size)
    tracks = []
    for frame in range(0, TOTAL_FRAMES, 5):
        track = {
            "frame_number": frame,
            "timestamp": round(frame / FPS, 3),
            "player_positions": (
                generate_player_positions(frame, 1) +
                generate_player_positions(frame, 2)
            ),
            "ball_position": generate_ball_position(frame),
        }
        tracks.append(track)
    
    # Generate events
    events = generate_events()
    
    # Generate statistics
    statistics = generate_statistics()
    statistics["heatmap_team1"] = generate_heatmap_data(1)
    statistics["heatmap_team2"] = generate_heatmap_data(2)
    statistics["pass_network_team1"] = generate_pass_network(1)
    statistics["pass_network_team2"] = generate_pass_network(2)
    
    # Generate analytics summary
    analytics = {
        "video_info": {
            "fps": FPS,
            "duration": DURATION,
            "total_frames": TOTAL_FRAMES,
            "width": 1920,
            "height": 1080,
        },
        "statistics": statistics,
        "events_summary": {
            "total": len(events),
            "by_type": {},
        },
        "generated_at": datetime.now().isoformat(),
    }
    
    # Count events by type
    for event in events:
        t = event["type"]
        analytics["events_summary"]["by_type"][t] = analytics["events_summary"]["by_type"].get(t, 0) + 1
    
    # Save files
    with open(OUTPUT_DIR / "tracks.json", "w") as f:
        json.dump(tracks, f, indent=2)
    print(f"  Saved tracks.json ({len(tracks)} frames)")
    
    with open(OUTPUT_DIR / "events.json", "w") as f:
        json.dump(events, f, indent=2)
    print(f"  Saved events.json ({len(events)} events)")
    
    with open(OUTPUT_DIR / "statistics.json", "w") as f:
        json.dump(statistics, f, indent=2)
    print(f"  Saved statistics.json")
    
    with open(OUTPUT_DIR / "analytics.json", "w") as f:
        json.dump(analytics, f, indent=2)
    print(f"  Saved analytics.json")
    
    print("\nDemo data generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nTo import into dashboard:")
    print("  1. Start the dashboard: pnpm run standalone")
    print("  2. Upload Test6.mp4 through the UI")
    print("  3. The demo data will be automatically loaded")


if __name__ == "__main__":
    main()
