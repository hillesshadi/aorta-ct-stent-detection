"""
models.py
=========
Deep-learning segmentation architectures – all implemented in PyTorch.

Four variants are provided so that their performance can be compared:

  1. ``UNet2D``    – Classic 4-level encoder-decoder U-Net [Ronneberger 2015]
  2. ``ResUNet2D`` – U-Net with residual skip connections [Zhang 2018]
  3. ``AttUNet2D`` – Attention U-Net with gated attention gates [Oktay 2018]
  4. ``NNUNet2D``  – nnU-Net style (instance norm instead of batch norm)

All models operate on **2-D slices** (B, 1, H, W) → (B, 1, H, W).
A sigmoid is NOT applied inside the forward pass; use BCEWithLogitsLoss
during training and torch.sigmoid() at inference time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
# ADD after imports, before DoubleConv class:

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Works on raw logits — applies sigmoid internally.
    Handles extreme class imbalance (stent << background).
    """
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_f   = probs.view(probs.size(0), -1)
        targets_f = targets.view(targets.size(0), -1)
        intersection = (probs_f * targets_f).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs_f.sum(dim=1) + targets_f.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Dice + BCE combined loss — standard for medical image segmentation.

    For stent segmentation (extreme imbalance), alpha=0.7 Dice dominant.
    For aorta segmentation (moderate imbalance), alpha=0.5 balanced.

    Parameters
    ----------
    alpha : float
        Weight of Dice loss. (1-alpha) is weight of BCE loss.
    """
    def __init__(self, alpha: float = 0.7) -> None:
        super().__init__()
        self.alpha   = alpha
        self.dice    = DiceLoss()
        self.bce     = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.dice(logits, targets) + \
               (1.0 - self.alpha) * self.bce(logits, targets)
    
class DoubleConv(nn.Module):
    """Two consecutive Conv2d → BN → ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int, norm: str = "batch") -> None:
        super().__init__()
        NormLayer = nn.BatchNorm2d if norm == "batch" else nn.InstanceNorm2d
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=(norm == "instance")),
            NormLayer(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=(norm == "instance")),
            NormLayer(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.conv(x)


class ResBlock(nn.Module):
    """Residual block with a 1×1 projection shortcut when channels differ."""

    def __init__(self, in_ch: int, out_ch: int, norm: str = "batch") -> None:
        super().__init__()
        NormLayer = nn.BatchNorm2d if norm == "batch" else nn.InstanceNorm2d
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = NormLayer(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = NormLayer(out_ch)
        self.skip  = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), NormLayer(out_ch))
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x), inplace=True)


class AttGate(nn.Module):
    """Gated attention module from Oktay et al. (Attention U-Net, 2018).

    Produces a soft attention map ∈ (0,1) that is element-wise multiplied
    onto the skip-connection feature map, suppressing irrelevant regions.

    Parameters
    ----------
    Fg : int  feature channels of the gating signal (decoder stream)
    Fl : int  feature channels of the skip-connection (encoder stream)
    Fint : int  intermediate channel size
    """

    def __init__(self, Fg: int, Fl: int, Fint: int) -> None:
        super().__init__()
        self.Wg  = nn.Conv2d(Fg, Fint, kernel_size=1, bias=True)
        self.Wx  = nn.Conv2d(Fl, Fint, kernel_size=1, bias=True)
        self.psi = nn.Sequential(
            nn.Conv2d(Fint, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """
        Parameters
        ----------
        g : gating signal  (from decoder, possibly smaller spatial resolution)
        x : skip feature  (from encoder, same spatial size as *x*)
        """
        # Align spatial dims of g to x via bilinear up-sampling if needed
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        att = self.psi(F.relu(self.Wg(g) + self.Wx(x), inplace=True))
        return x * att


# ---------------------------------------------------------------------------
# 1. Classic U-Net 2D
# ---------------------------------------------------------------------------

class UNet2D(nn.Module):
    """Standard U-Net with 4 encoder + 4 decoder levels."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32) -> None:
        super().__init__()
        b = base
        # Encoder
        self.enc1 = DoubleConv(in_ch, b);     self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(b,     b*2);   self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(b*2,   b*4);   self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(b*4,   b*8);   self.pool4 = nn.MaxPool2d(2)
        # Bridge
        self.bridge = DoubleConv(b*8, b*16)
        # Decoder
        self.up4  = nn.ConvTranspose2d(b*16, b*8,  2, 2); self.dec4 = DoubleConv(b*16, b*8)
        self.up3  = nn.ConvTranspose2d(b*8,  b*4,  2, 2); self.dec3 = DoubleConv(b*8,  b*4)
        self.up2  = nn.ConvTranspose2d(b*4,  b*2,  2, 2); self.dec2 = DoubleConv(b*4,  b*2)
        self.up1  = nn.ConvTranspose2d(b*2,  b,    2, 2); self.dec1 = DoubleConv(b*2,  b)
        self.out  = nn.Conv2d(b, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        e1 = self.enc1(x);  e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2)); e4 = self.enc4(self.pool3(e3))
        b  = self.bridge(self.pool4(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


# ---------------------------------------------------------------------------
# 2. Residual U-Net 2D
# ---------------------------------------------------------------------------

class ResUNet2D(nn.Module):
    """U-Net with residual encoder/decoder blocks."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32) -> None:
        super().__init__()
        b = base
        self.enc1 = ResBlock(in_ch, b);   self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResBlock(b,     b*2); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResBlock(b*2,   b*4); self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResBlock(b*4,   b*8); self.pool4 = nn.MaxPool2d(2)
        self.bridge = ResBlock(b*8, b*16)
        self.up4  = nn.ConvTranspose2d(b*16, b*8,  2, 2); self.dec4 = ResBlock(b*16, b*8)
        self.up3  = nn.ConvTranspose2d(b*8,  b*4,  2, 2); self.dec3 = ResBlock(b*8,  b*4)
        self.up2  = nn.ConvTranspose2d(b*4,  b*2,  2, 2); self.dec2 = ResBlock(b*4,  b*2)
        self.up1  = nn.ConvTranspose2d(b*2,  b,    2, 2); self.dec1 = ResBlock(b*2,  b)
        self.out  = nn.Conv2d(b, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        e1 = self.enc1(x);  e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2)); e4 = self.enc4(self.pool3(e3))
        b  = self.bridge(self.pool4(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


# ---------------------------------------------------------------------------
# 3. Attention U-Net 2D
# ---------------------------------------------------------------------------

class AttUNet2D(nn.Module):
    """Attention U-Net — gated attention on every skip connection.

    Particularly effective for small, high-contrast targets like
    metal stent struts inside the aortic lumen.
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32) -> None:
        super().__init__()
        b = base
        self.enc1 = DoubleConv(in_ch, b);   self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(b,     b*2); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(b*2,   b*4); self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(b*4,   b*8); self.pool4 = nn.MaxPool2d(2)
        self.bridge = DoubleConv(b*8, b*16)

        self.up4   = nn.ConvTranspose2d(b*16, b*8, 2, 2)
        self.att4  = AttGate(b*8,  b*8,  b*4); self.dec4 = DoubleConv(b*16, b*8)
        self.up3   = nn.ConvTranspose2d(b*8,  b*4, 2, 2)
        self.att3  = AttGate(b*4,  b*4,  b*2); self.dec3 = DoubleConv(b*8,  b*4)
        self.up2   = nn.ConvTranspose2d(b*4,  b*2, 2, 2)
        self.att2  = AttGate(b*2,  b*2,  b);   self.dec2 = DoubleConv(b*4,  b*2)
        self.up1   = nn.ConvTranspose2d(b*2,  b,   2, 2)
        self.att1  = AttGate(b,    b,    b//2); self.dec1 = DoubleConv(b*2,  b)
        self.out   = nn.Conv2d(b, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        e1 = self.enc1(x);  e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2)); e4 = self.enc4(self.pool3(e3))
        b  = self.bridge(self.pool4(e4))

        u4 = self.up4(b);  a4 = self.att4(u4, e4); d4 = self.dec4(torch.cat([u4, a4], 1))
        u3 = self.up3(d4); a3 = self.att3(u3, e3); d3 = self.dec3(torch.cat([u3, a3], 1))
        u2 = self.up2(d3); a2 = self.att2(u2, e2); d2 = self.dec2(torch.cat([u2, a2], 1))
        u1 = self.up1(d2); a1 = self.att1(u1, e1); d1 = self.dec1(torch.cat([u1, a1], 1))
        return self.out(d1)


# ---------------------------------------------------------------------------
# 4. nnU-Net 2D (instance norm variant)
# ---------------------------------------------------------------------------

class NNUNet2D(nn.Module):
    """nnU-Net-style model: same topology as U-Net but uses Instance Norm.

    Instance Norm is preferred in nnU-Net for small batch sizes and varied
    patch sizes; it also avoids leakage of statistics across patients.
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32) -> None:
        super().__init__()
        b = base
        self.enc1 = DoubleConv(in_ch, b,   norm="instance"); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(b,     b*2, norm="instance"); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(b*2,   b*4, norm="instance"); self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(b*4,   b*8, norm="instance"); self.pool4 = nn.MaxPool2d(2)
        self.bridge = DoubleConv(b*8,  b*16, norm="instance")
        self.up4  = nn.ConvTranspose2d(b*16, b*8,  2, 2)
        self.dec4 = DoubleConv(b*16, b*8,  norm="instance")
        self.up3  = nn.ConvTranspose2d(b*8,  b*4,  2, 2)
        self.dec3 = DoubleConv(b*8,  b*4,  norm="instance")
        self.up2  = nn.ConvTranspose2d(b*4,  b*2,  2, 2)
        self.dec2 = DoubleConv(b*4,  b*2,  norm="instance")
        self.up1  = nn.ConvTranspose2d(b*2,  b,    2, 2)
        self.dec1 = DoubleConv(b*2,  b,    norm="instance")
        self.out  = nn.Conv2d(b, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        e1 = self.enc1(x);  e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2)); e4 = self.enc4(self.pool3(e3))
        b  = self.bridge(self.pool4(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

_MODEL_REGISTRY = {
    "UNet":    UNet2D,
    "ResUNet": ResUNet2D,
    "AttUNet": AttUNet2D,
    "nnUNet":  NNUNet2D,
}


# REPLACE build_all_models WITH:

def build_segmentation_models(base: int = 32) -> dict:
    """
    Build separate model sets for aorta and stent segmentation.

    Why separate models?
    - Aorta: in_ch=1 (aorta-windowed channel), moderate imbalance
    - Stent: in_ch=2 (stent-windowed channel + metal prior mask),
             extreme imbalance, needs attention mechanism

    The stent model receives 2 input channels:
      Channel 0: stent-windowed CT (center=1500, width=2000, normalised)
      Channel 1: binary metal prior from HU thresholding (float, 0.0/1.0)
    The metal prior acts as a spatial hint — it tells the network where
    metal is likely to be, which dramatically reduces false negatives.

    Returns
    -------
    dict with keys:
      "aorta_models": {name: model}  — all 4 architectures for aorta
      "stent_models": {name: model}  — AttUNet + NNUNet for stent (best for small structures)
      "aorta_loss":   CombinedLoss(alpha=0.5)
      "stent_loss":   CombinedLoss(alpha=0.8)  — Dice-dominant for imbalance
    """
    aorta_models = {
        name: cls(in_ch=1, out_ch=1, base=base)
        for name, cls in _MODEL_REGISTRY.items()
    }

    # Stent: only attention-based models — better for thin, sparse structures
    stent_model_classes = {
        "AttUNet": AttUNet2D,
        "nnUNet":  NNUNet2D,
    }
    stent_models = {
        name: cls(in_ch=2, out_ch=1, base=base)   # in_ch=2: stent channel + metal prior
        for name, cls in stent_model_classes.items()
    }

    return {
        "aorta_models": aorta_models,
        "stent_models": stent_models,
        "aorta_loss":   CombinedLoss(alpha=0.5),
        "stent_loss":   CombinedLoss(alpha=0.8),
    }


# Keep build_all_models as alias for backward compatibility
def build_all_models(base: int = 32) -> dict:
    return build_segmentation_models(base)["aorta_models"]
