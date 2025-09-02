import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class MobileViTBlock(nn.Module):
    def __init__(self, dim, ffn_dim, fusion_in_channels=None,
                 n_transformer_blocks=4, patch_size=(2, 2), num_heads=2):
        super().__init__()
        ph, pw = patch_size

        self.local_rep = ConvBNReLU(dim, dim, kernel_size=3, padding=1)
        self.project_patches = nn.Linear(ph * pw * dim, dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads,
            dim_feedforward=ffn_dim,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_blocks)

        self.fusion = ConvBNReLU(2 * dim, dim, kernel_size=1)
        self.ph, self.pw = ph, pw

    def unfold(self, x):
        B, C, H, W = x.shape
        ph, pw = self.ph, self.pw
        new_H = ((H + ph - 1) // ph) * ph
        new_W = ((W + pw - 1) // pw) * pw
        x = F.pad(x, (0, new_W - W, 0, new_H - H))
        patches = x.unfold(2, ph, ph).unfold(3, pw, pw)
        patches = patches.contiguous().view(B, C, -1, ph * pw)
        patches = patches.permute(0, 2, 3, 1).contiguous().view(B, -1, ph * pw * C)
        return patches, new_H, new_W

    def fold(self, patches, H, W):
        B, N, D = patches.shape
        ph, pw = self.ph, self.pw
        nH, nW = H // ph, W // pw
        x = patches.view(B, nH, nW, D).permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=(H, W), mode='nearest')
        return x

    def forward(self, x):
        y = self.local_rep(x)
        patches, H, W = self.unfold(y)
        patches = self.project_patches(patches)
        patches = self.transformer(patches)
        global_rep = self.fold(patches, H, W)
        out = self.fusion(torch.cat([y, global_rep], dim=1))
        return out

class MobileViTUNet(nn.Module):
    def __init__(
        self,
        image_size=64,
        in_channels=1,
        out_channels=1,
        model_channels=96,
        cond_lq: bool = False,
        lq_size: int = None,
    ):
        super().__init__()
        self.cond_lq = cond_lq
        self.lq_size = lq_size or image_size
        C = model_channels

        # Build encoder for low-quality input if requested
        if cond_lq:
            ds_factor = self.lq_size // image_size if self.lq_size != image_size else 1
            self.lq_encoder = nn.Sequential(
                nn.Conv2d(in_channels, C, kernel_size=3, stride=ds_factor, padding=1, bias=False),
                nn.BatchNorm2d(C),
                nn.SiLU(),
            )

        # Encoder blocks for primary input
        self.enc1 = ConvBNReLU(in_channels, C, 3, padding=1)
        self.enc2 = ConvBNReLU(C, 2 * C, 3, stride=2, padding=1)
        self.enc3 = MobileViTBlock(2 * C, 6 * C)
        self.enc4 = MobileViTBlock(2 * C, 6 * C)
        self.bottleneck = MobileViTBlock(2 * C, 6 * C)

        # Decoder blocks
        self.dec4 = ConvBNReLU(4 * C, 2 * C, 3, padding=1)
        self.dec3 = ConvBNReLU(4 * C, 2 * C, 3, padding=1)
        self.dec2 = ConvBNReLU(3 * C, C, 3, padding=1)
        self.dec1 = nn.Conv2d(C, out_channels, 3, padding=1)

    def forward(self, x, t=None, lq=None):
        # Primary encoding
        e1 = self.enc1(x)

        # Fuse low-quality features if enabled
        if self.cond_lq and lq is not None:
            lq_resized = F.interpolate(lq, size=e1.shape[2:], mode='bilinear', align_corners=False)
            lq_feat = self.lq_encoder(lq_resized)
            e1 = e1 + lq_feat

        # Continue encoding
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)

        # Decoding path
        d4 = F.interpolate(b, size=e4.shape[2:], mode='nearest')
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = F.interpolate(d4, size=e3.shape[2:], mode='nearest')
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = F.interpolate(d3, size=e1.shape[2:], mode='nearest')
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        out = self.dec1(d2)
        return out

# === High-Fidelity Version ===
class ConvGNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, num_groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.gn = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class UpConvGNReLU(nn.Module):
    """Bilinear upsample + conv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvGNReLU(in_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.up(x)


class SE(nn.Module):
    """Squeeze-Excitation channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(mid, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


# ---------------------------
# High-fidelity MobileViT block
#   - exact (un)patchify (no interpolate hacks)
#   - pre-norm Transformer
#   - SE attention on fused features
#   - residual around the block
# ---------------------------

class MobileViTBlockHF(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        n_transformer_blocks=6,
        patch_size=(2, 2),
        num_heads=4,
    ):
        super().__init__()
        ph, pw = patch_size
        self.ph, self.pw = ph, pw
        self.dim = dim

        # Local conv features
        self.local_rep = ConvGNReLU(dim, dim, kernel_size=3, padding=1)

        # Patch projection <-> recovery
        self.project_patches = nn.Linear(ph * pw * dim, dim, bias=True)
        self.recover_patches = nn.Linear(dim, ph * pw * dim, bias=True)

        # Pre-norm Transformer (norm_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_blocks)

        # Fuse local + global
        self.fuse_conv = ConvGNReLU(2 * dim, dim, kernel_size=1)
        self.se = SE(dim, reduction=16)

    @torch.no_grad()
    def _pad_to_multiple(self, x):
        B, C, H, W = x.shape
        H_pad = (self.ph - H % self.ph) % self.ph
        W_pad = (self.pw - W % self.pw) % self.pw
        x = F.pad(x, (0, W_pad, 0, H_pad))
        return x, H, W  # keep originals to crop back later

    def _patchify(self, x):
        """x: [B, C, H, W] -> patches: [B, N, ph*pw*C] (non-overlapping)"""
        B, C, H, W = x.shape
        ph, pw = self.ph, self.pw
        patches = x.unfold(2, ph, ph).unfold(3, pw, pw)          # [B, C, nH, nW, ph, pw]
        patches = patches.permute(0, 2, 3, 4, 5, 1).contiguous() # [B, nH, nW, ph, pw, C]
        patches = patches.view(B, -1, ph * pw * C)               # [B, N, ph*pw*C]
        return patches

    def _unpatchify(self, patches, H, W):
        """patches: [B, N, ph*pw*dim] -> [B, dim, H, W] exact stitch"""
        B, N, D = patches.shape
        ph, pw, C = self.ph, self.pw, self.dim
        nH, nW = H // ph, W // pw

        x = self.recover_patches(patches)                        # [B, N, ph*pw*C]
        x = x.view(B, nH, nW, ph, pw, C)                         # [B, nH, nW, ph, pw, C]
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()             # [B, C, nH, ph, nW, pw]
        x = x.view(B, C, H, W)                                   # [B, C, H, W]
        return x

    def forward(self, x):
        # x: [B, dim, H, W]
        identity = x
        y = self.local_rep(x)

        y_pad, H0, W0 = self._pad_to_multiple(y)                 # pad so H,W % patch == 0
        H, W = y_pad.shape[-2:]

        patches = self._patchify(y_pad)                          # [B, N, ph*pw*dim]
        tokens  = self.project_patches(patches)                  # [B, N, dim]
        tokens  = self.transformer(tokens)                       # global reasoning
        global_rep = self._unpatchify(tokens, H, W)              # [B, dim, H, W]

        # remove padding
        global_rep = global_rep[..., :H0, :W0]

        fused = self.fuse_conv(torch.cat([y, global_rep], dim=1))
        fused = self.se(fused)
        return fused + identity


# ---------------------------
# High-fidelity UNet with HF blocks and residual output
# (decoder edited as requested)
# ---------------------------

class MobileViTUNetHighFidelity(nn.Module):
    def __init__(
        self,
        image_size=64,
        in_channels=1,
        out_channels=1,
        model_channels=96,
        cond_lq: bool = False,
        lq_size: int = None,
    ):
        super().__init__()
        self.cond_lq = cond_lq
        self.lq_size = lq_size or image_size
        C = model_channels

        # Optional encoder for low-quality conditioning
        if cond_lq:
            ds_factor = self.lq_size // image_size if self.lq_size != image_size else 1
            self.lq_encoder = nn.Sequential(
                nn.Conv2d(in_channels, C, kernel_size=3, stride=ds_factor, padding=1, bias=False),
                nn.GroupNorm(8, C),
                nn.GELU(),
            )

        # Encoder
        self.enc1 = ConvGNReLU(in_channels, C, 3, padding=1)          # H, W
        self.enc2 = ConvGNReLU(C, 2 * C, 3, stride=2, padding=1)      # H/2, W/2

        # HF MobileViT blocks keep spatial size
        self.enc3 = MobileViTBlockHF(2 * C, 6 * C, n_transformer_blocks=6, num_heads=4)  # H/2
        self.enc4 = MobileViTBlockHF(2 * C, 6 * C, n_transformer_blocks=6, num_heads=4)  # H/2
        self.bottleneck = MobileViTBlockHF(2 * C, 6 * C, n_transformer_blocks=6, num_heads=4)  # H/2

        # --- Decoder (edited) ---
        # Only the first decoder stage upsamples; the next two are same-res convs.
        self.dec4 = UpConvGNReLU(4 * C, 2 * C)                # upsample here (H/2 -> H)
        self.dec3 = ConvGNReLU(4 * C, 2 * C, 3, padding=1)    # no upsample
        self.dec2 = ConvGNReLU(3 * C, C, 3, padding=1)        # no upsample

        self.refine = nn.Sequential(
            ConvGNReLU(C, C, 3, padding=1),
            nn.Conv2d(C, out_channels, 3, padding=1)
        )

    def forward(self, x, t=None, lq=None):
        inp = x

        # enc1 (optionally fuse LQ conditioning)
        e1 = self.enc1(x)
        if self.cond_lq and lq is not None:
            lq_resized = F.interpolate(lq, size=e1.shape[2:], mode='bilinear', align_corners=False)
            lq_feat = self.lq_encoder(lq_resized)
            e1 = e1 + lq_feat

        # deeper encoder
        e2 = self.enc2(e1)   # H/2
        e3 = self.enc3(e2)   # H/2
        e4 = self.enc4(e3)   # H/2
        b  = self.bottleneck(e4)  # H/2

        # --- Decoder (edited) ---
        # 1) First stage: concat at H/2, then upsample inside dec4 -> H
        d4_in = torch.cat([b, e4], dim=1)           # both H/2
        d4 = self.dec4(d4_in)                       # -> H

        # 2) Align skip (e3) to H and fuse; stay at H
        e3_up = F.interpolate(e3, size=d4.shape[2:], mode='bilinear', align_corners=False)
        d3_in = torch.cat([d4, e3_up], dim=1)       # H
        d3 = self.dec3(d3_in)                       # H

        # 3) Final fuse with e1 at H
        d2_in = torch.cat([d3, e1], dim=1)          # H
        d2 = self.dec2(d2_in)                       # H

        # refine + residual output
        res = self.refine(d2)
        if res.shape[-2:] != inp.shape[-2:]:
            res = F.interpolate(res, size=inp.shape[-2:], mode='bilinear', align_corners=False)

        out = inp + res
        return out
