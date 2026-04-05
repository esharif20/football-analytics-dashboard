"""EfficientNetV2-B0 + 3D conv head for football event classification.

Architecture is identical to training so checkpoint weights load cleanly.
All training-only artefacts (Lightning boilerplate, loss config, mode flags)
have been removed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Freeze-through index used during training (kept for architectural parity).
_FREEZE_THRU = -5


class EfficientNetBCE_v3(nn.Module):
    """4-class event classifier with EfficientNetV2-B0 backbone + Conv3D head."""

    def __init__(self) -> None:
        super().__init__()
        import timm  # lazy — not a hard dep for the rest of the pipeline

        self.num_classes = 4
        self.backbone = timm.create_model(
            "tf_efficientnetv2_b0", pretrained=False, num_classes=0
        )
        feat_dim = self.backbone.num_features

        n_blocks = len(self.backbone.blocks)
        self._frozen_thru = n_blocks + _FREEZE_THRU

        self.reduce = nn.Conv2d(feat_dim, 64, 1)
        self.conv3d = nn.Sequential(
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, self.num_classes),
        )
        # pos_weight buffer — overwritten by checkpoint
        self.register_buffer("pos_weight", torch.ones(self.num_classes))

    def _backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        if hasattr(self.backbone, "act1"):
            x = self.backbone.act1(x)
        for block in self.backbone.blocks:
            x = block(x)
        x = self.backbone.conv_head(x)
        x = self.backbone.bn2(x)
        if hasattr(self.backbone, "act2"):
            x = self.backbone.act2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self._backbone_forward(x)           # (B*T, feat_dim, h, w)
        feat = self.reduce(feat)                    # (B*T, 64, h, w)
        _, c, h, w = feat.shape
        feat = feat.view(B, T, c, h, w).permute(0, 2, 1, 3, 4)  # (B, 64, T, h, w)
        feat = self.conv3d(feat)
        feat = feat.mean(dim=[2, 3, 4])            # GAP → (B, 64)
        return self.head(feat)                     # (B, 4)

    # ------------------------------------------------------------------
    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        device: str = "cpu",
    ) -> Optional["EfficientNetBCE_v3"]:
        """Load model weights from a Lightning or plain PyTorch checkpoint.

        Returns None and logs a warning if the file is missing or corrupt.
        Strips ``model.`` / ``module.`` prefixes added by Lightning / DDP.
        """
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            logger.warning(
                "Event detection checkpoint not found at %s — ML events disabled", ckpt_path
            )
            return None

        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        except Exception as exc:
            logger.warning("Failed to load event checkpoint %s: %s", ckpt_path, exc)
            return None

        state = ckpt.get("state_dict", ckpt)
        cleaned: dict = {}
        for k, v in state.items():
            for prefix in ("model.", "module."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
            cleaned[k] = v

        model = cls()
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            logger.debug("Checkpoint missing keys (%d): %s …", len(missing), missing[:3])
        if unexpected:
            logger.debug("Checkpoint unexpected keys (%d): %s …", len(unexpected), unexpected[:3])

        logger.info(
            "Event detection model loaded: %d params, checkpoint=%s",
            sum(p.numel() for p in model.parameters()),
            ckpt_path.name,
        )
        return model
