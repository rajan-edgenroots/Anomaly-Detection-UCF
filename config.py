"""
Configuration settings for Federated Video Anomaly Detection inference.
Matches exact preprocessing, hyperparameter, and class settings from training notebook.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set


@dataclass(frozen=True)
class Config:
    """Centralized application configuration parameters."""
    
    # Model Preprocessing & Architecture Specs (EXACT MATCH TO NOTEBOOK)
    image_size: int = 64
    channels: int = 3
    sequence_length: int = 16
    sequence_stride: int = 8
    width_multiplier: float = 0.5
    temporal_units: int = 64
    weight_decay: float = 1e-4
    batch_size: int = 32
    
    # Inference Strides (EXACT MATCH TO NOTEBOOK)
    inference_frame_stride: int = 10
    inference_sequence_stride: int = 8

    # Classes in natural sorted order matching dataset scanning in notebook
    class_names: List[str] = field(
        default_factory=lambda: [
            "Abuse",
            "Arrest",
            "Arson",
            "Assault",
            "Burglary",
            "Explosion",
            "Fighting",
            "NormalVideos",
            "RoadAccidents",
            "Robbery",
            "Shooting",
            "Shoplifting",
            "Stealing",
            "Vandalism",
        ]
    )

    # Allowed inputs & limits
    allowed_extensions: Set[str] = field(
        default_factory=lambda: {"mp4", "avi", "mov", "mkv"}
    )
    max_content_length: int = 500 * 1024 * 1024  # 500 MB max file size

    # Base Paths
    base_dir: Path = Path(__file__).resolve().parent
    upload_folder: Path = base_dir / "uploads"
    temp_folder: Path = base_dir / "temp"

    @property
    def model_weights_path(self) -> Path:
        """
        Dynamically locate model weights in models/ or model/ directory.
        """
        candidates = [
            self.base_dir / "models" / "best_model.weights.h5",
            self.base_dir / "models" / "best_se_resnet18_fedavg.weights.h5",
            self.base_dir / "model" / "best_model.weights.h5",
            self.base_dir / "model" / "best_se_resnet18_fedavg.weights.h5",
            self.base_dir / "best_se_resnet18_fedavg.weights.h5",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return self.base_dir / "models" / "best_model.weights.h5"

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def class_to_id(self) -> Dict[str, int]:
        return {name: idx for idx, name in enumerate(self.class_names)}

    @property
    def id_to_class(self) -> Dict[int, str]:
        return {idx: name for idx, name in enumerate(self.class_names)}


# Global instance
CFG = Config()
