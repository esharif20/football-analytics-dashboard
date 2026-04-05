"""ML-based football event detection module.

Provides sliding-window inference using an EfficientNetV2-B0 + 3D conv head
trained on the DFL competition dataset.

Detected event types: challenge, play, throwin.
These supplement the heuristic events (pass, shot, tackle) from the main pipeline.

Typical usage in the pipeline::

    from event_detector import EventModelInference, SlidingWindowConfig, config_from_env

    cfg = config_from_env()
    inf = EventModelInference.from_config(cfg)
    if inf is not None:
        ml_events = inf.detect_events(frames, fps)
"""

from .config import SlidingWindowConfig, config_from_env
from .inference import EventModelInference

__all__ = ["EventModelInference", "SlidingWindowConfig", "config_from_env"]
