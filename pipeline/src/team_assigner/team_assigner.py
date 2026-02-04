"""
Team classification module using SigLIP embeddings + UMAP + KMeans.
Matches the original repo's team_assigner.py structure.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import Counter

try:
    from sklearn.cluster import KMeans
    import umap
except ImportError:
    raise ImportError("Please install scikit-learn and umap-learn: pip install scikit-learn umap-learn")


@dataclass
class TeamColors:
    """Team color information."""
    team_1_color: Tuple[int, int, int]  # BGR
    team_2_color: Tuple[int, int, int]  # BGR
    team_1_name: str = "Team 1"
    team_2_name: str = "Team 2"


class TeamAssigner:
    """
    Assigns players to teams using visual embeddings.
    
    Pipeline:
    1. Extract player crops from frames
    2. Get SigLIP embeddings for each crop
    3. Reduce dimensions with UMAP
    4. Cluster with KMeans (k=2)
    5. Assign team IDs based on cluster membership
    """
    
    def __init__(
        self,
        embedding_model: str = "google/siglip-base-patch16-224",
        n_clusters: int = 2,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_n_components: int = 3,
        device: str = "auto"
    ):
        """
        Initialize the team assigner.
        
        Args:
            embedding_model: HuggingFace model ID for embeddings
            n_clusters: Number of teams (default 2)
            umap_n_neighbors: UMAP neighbors parameter
            umap_min_dist: UMAP minimum distance
            umap_n_components: UMAP output dimensions
            device: Device for inference
        """
        self.n_clusters = n_clusters
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_n_components = umap_n_components
        self.device = self._resolve_device(device)
        
        # Lazy load models
        self._processor = None
        self._model = None
        self._embedding_model_name = embedding_model
        
        # Results
        self.team_colors: Optional[TeamColors] = None
        self.track_to_team: Dict[int, int] = {}
    
    def _resolve_device(self, device: str) -> str:
        """Resolve 'auto' device to actual device."""
        if device != "auto":
            return device
        
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _load_model(self):
        """Lazy load the SigLIP model."""
        if self._model is None:
            print(f"Loading SigLIP model: {self._embedding_model_name}...")
            try:
                from transformers import AutoProcessor, AutoModel
                import torch
                
                self._processor = AutoProcessor.from_pretrained(self._embedding_model_name)
                self._model = AutoModel.from_pretrained(self._embedding_model_name)
                self._model = self._model.to(self.device)
                self._model.eval()
            except Exception as e:
                print(f"Warning: Could not load SigLIP model: {e}")
                print("Falling back to color-based team assignment")
                self._model = "fallback"
    
    def _get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Get SigLIP embedding for an image crop."""
        self._load_model()
        
        if self._model == "fallback":
            return None
        
        try:
            import torch
            from PIL import Image
            
            # Convert BGR to RGB
            rgb_image = image[:, :, ::-1]
            pil_image = Image.fromarray(rgb_image)
            
            # Process and get embedding
            inputs = self._processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model.get_image_features(**inputs)
            
            return outputs.cpu().numpy().flatten()
        except Exception as e:
            print(f"Warning: Embedding extraction failed: {e}")
            return None
    
    def _get_dominant_color(self, image: np.ndarray) -> Tuple[int, int, int]:
        """Get dominant color from image crop (fallback method)."""
        # Focus on center region (jersey area)
        h, w = image.shape[:2]
        center = image[h//4:3*h//4, w//4:3*w//4]
        
        # Calculate mean color
        mean_color = np.mean(center, axis=(0, 1)).astype(int)
        return tuple(mean_color)
    
    def _extract_crops(
        self,
        frames: List[np.ndarray],
        tracks: List[Dict],
        sample_rate: int = 5,
        max_samples_per_track: int = 10
    ) -> Dict[int, List[np.ndarray]]:
        """
        Extract player crops from frames.
        
        Args:
            frames: List of BGR frames
            tracks: List of frame tracks
            sample_rate: Sample every N frames
            max_samples_per_track: Maximum crops per track ID
            
        Returns:
            Dictionary mapping track_id to list of crops
        """
        crops_by_track = {}
        
        for frame_idx in range(0, len(frames), sample_rate):
            frame = frames[frame_idx]
            frame_tracks = tracks[frame_idx]
            
            for track_id, track in frame_tracks.get("players", {}).items():
                if track_id not in crops_by_track:
                    crops_by_track[track_id] = []
                
                if len(crops_by_track[track_id]) >= max_samples_per_track:
                    continue
                
                # Extract crop
                x1, y1, x2, y2 = map(int, track.bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crops_by_track[track_id].append(crop)
        
        return crops_by_track
    
    def classify_teams(
        self,
        frames: List[np.ndarray],
        tracks: List[Dict],
        progress_callback: Optional[callable] = None
    ) -> Dict[int, int]:
        """
        Classify players into teams.
        
        Args:
            frames: List of BGR frames
            tracks: List of frame tracks
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary mapping track_id to team_id (0 or 1)
        """
        print("Extracting player crops...")
        crops_by_track = self._extract_crops(frames, tracks)
        
        if not crops_by_track:
            print("Warning: No player crops found")
            return {}
        
        print(f"Found {len(crops_by_track)} unique player tracks")
        
        # Get embeddings for each track
        print("Computing embeddings...")
        track_embeddings = {}
        track_colors = {}
        
        for i, (track_id, crops) in enumerate(crops_by_track.items()):
            if progress_callback:
                progress_callback(i / len(crops_by_track), f"Processing track {track_id}")
            
            embeddings = []
            colors = []
            
            for crop in crops:
                emb = self._get_embedding(crop)
                if emb is not None:
                    embeddings.append(emb)
                colors.append(self._get_dominant_color(crop))
            
            if embeddings:
                track_embeddings[track_id] = np.mean(embeddings, axis=0)
            track_colors[track_id] = np.mean(colors, axis=0).astype(int)
        
        # Cluster using embeddings or colors
        if track_embeddings and len(track_embeddings) >= self.n_clusters:
            print("Clustering with SigLIP embeddings + UMAP + KMeans...")
            self.track_to_team = self._cluster_embeddings(track_embeddings)
        else:
            print("Clustering with color features...")
            self.track_to_team = self._cluster_colors(track_colors)
        
        # Extract team colors
        self._extract_team_colors(track_colors)
        
        return self.track_to_team
    
    def _cluster_embeddings(self, track_embeddings: Dict[int, np.ndarray]) -> Dict[int, int]:
        """Cluster tracks using embeddings."""
        track_ids = list(track_embeddings.keys())
        embeddings = np.array([track_embeddings[tid] for tid in track_ids])
        
        # UMAP dimensionality reduction
        n_samples = len(embeddings)
        n_neighbors = min(self.umap_n_neighbors, n_samples - 1)
        
        if n_neighbors < 2:
            # Not enough samples for UMAP, use raw embeddings
            reduced = embeddings
        else:
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=self.umap_min_dist,
                n_components=min(self.umap_n_components, n_samples - 1),
                random_state=42
            )
            reduced = reducer.fit_transform(embeddings)
        
        # KMeans clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(reduced)
        
        return {track_ids[i]: int(labels[i]) for i in range(len(track_ids))}
    
    def _cluster_colors(self, track_colors: Dict[int, np.ndarray]) -> Dict[int, int]:
        """Cluster tracks using color features (fallback)."""
        track_ids = list(track_colors.keys())
        colors = np.array([track_colors[tid] for tid in track_ids])
        
        if len(colors) < self.n_clusters:
            return {tid: 0 for tid in track_ids}
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(colors)
        
        return {track_ids[i]: int(labels[i]) for i in range(len(track_ids))}
    
    def _extract_team_colors(self, track_colors: Dict[int, np.ndarray]):
        """Extract representative colors for each team."""
        team_1_colors = []
        team_2_colors = []
        
        for track_id, color in track_colors.items():
            team_id = self.track_to_team.get(track_id, 0)
            if team_id == 0:
                team_1_colors.append(color)
            else:
                team_2_colors.append(color)
        
        team_1_color = tuple(np.mean(team_1_colors, axis=0).astype(int)) if team_1_colors else (255, 0, 0)
        team_2_color = tuple(np.mean(team_2_colors, axis=0).astype(int)) if team_2_colors else (0, 0, 255)
        
        self.team_colors = TeamColors(
            team_1_color=team_1_color,
            team_2_color=team_2_color
        )
    
    def assign_goalkeepers(
        self,
        frames: List[np.ndarray],
        tracks: List[Dict]
    ) -> Dict[int, int]:
        """
        Assign goalkeepers to teams based on position.
        
        Goalkeepers are assigned to the team whose players are
        predominantly on the same side of the pitch.
        """
        # Get goalkeeper positions
        gk_positions = {}
        for frame_tracks in tracks:
            for gk_id, gk in frame_tracks.get("goalkeepers", {}).items():
                if gk_id not in gk_positions:
                    gk_positions[gk_id] = []
                cx = (gk.bbox[0] + gk.bbox[2]) / 2
                gk_positions[gk_id].append(cx)
        
        if not gk_positions:
            return {}
        
        # Get average x position for each team
        team_x_positions = {0: [], 1: []}
        for frame_tracks in tracks:
            for player_id, player in frame_tracks.get("players", {}).items():
                team_id = self.track_to_team.get(player_id)
                if team_id is not None:
                    cx = (player.bbox[0] + player.bbox[2]) / 2
                    team_x_positions[team_id].append(cx)
        
        team_avg_x = {
            0: np.mean(team_x_positions[0]) if team_x_positions[0] else 0,
            1: np.mean(team_x_positions[1]) if team_x_positions[1] else 0
        }
        
        # Assign goalkeepers
        gk_to_team = {}
        for gk_id, positions in gk_positions.items():
            avg_x = np.mean(positions)
            # Assign to team on same side
            if abs(avg_x - team_avg_x[0]) < abs(avg_x - team_avg_x[1]):
                gk_to_team[gk_id] = 0
            else:
                gk_to_team[gk_id] = 1
        
        return gk_to_team
    
    def apply_teams_to_tracks(
        self,
        tracks: List[Dict],
        gk_teams: Optional[Dict[int, int]] = None
    ) -> List[Dict]:
        """
        Apply team assignments to all tracks.
        
        Args:
            tracks: List of frame tracks
            gk_teams: Optional goalkeeper team assignments
            
        Returns:
            Tracks with team_id assigned
        """
        updated_tracks = []
        
        for frame_tracks in tracks:
            updated_frame = {
                "players": {},
                "goalkeepers": {},
                "referees": dict(frame_tracks.get("referees", {})),
                "ball": dict(frame_tracks.get("ball", {}))
            }
            
            # Assign player teams
            for player_id, player in frame_tracks.get("players", {}).items():
                player.team_id = self.track_to_team.get(player_id)
                updated_frame["players"][player_id] = player
            
            # Assign goalkeeper teams
            for gk_id, gk in frame_tracks.get("goalkeepers", {}).items():
                if gk_teams:
                    gk.team_id = gk_teams.get(gk_id)
                updated_frame["goalkeepers"][gk_id] = gk
            
            updated_tracks.append(updated_frame)
        
        return updated_tracks
