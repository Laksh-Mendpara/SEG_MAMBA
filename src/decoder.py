"""
SDBC-Mamba Decoder — full implementation.

Stages:
  I.   Multi-Scale Pyramid Alignment & Local Inductive Encoding
  II.  Parameter-Free Laplacian Spectral-Morphological Decoupling
  III. Anisotropic Boundary-Clamped 4-Way Selective Scan (BC-SS2D)
  IV.  Gated Spectral Re-Integration & Dense Prediction Heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── optional fast CUDA selective scan ───────────────────────────────────────
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA_CUDA = True
except ImportError:
    HAS_MAMBA_CUDA = False


# ════════════════════════════════════════════════════════════════════════════
# PyTorch reference fallback for selective scan (sequential, used on CPU/test)
# ════════════════════════════════════════════════════════════════════════════

def _selective_scan_ref(u, delta, A, B, C, D):
    """
    Pure-PyTorch reference selective scan (ZOH discretization).

    Shapes (K = 4 scan directions, interleaved in dim-1):
        u     : [B, K*D, L]
        delta : [B, K*D, L]   — already softplus-activated
        A     : [K*D, N]      — negative
        B     : [B, K, N, L]
        C     : [B, K, N, L]
        D     : [K*D]
    Returns:
        y     : [B, K*D, L]
    """
    B_sz, KD, L = u.shape
    K = B.shape[1]
    N = B.shape[2]
    D_dim = KD // K

    # [B, KD, L, N]
    dA = torch.exp(
        delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2)
    )
    # Expand B/C from [B,K,N,L] to [B,KD,L,N]
    B_exp = (B.unsqueeze(2)
              .expand(-1, -1, D_dim, -1, -1)
              .reshape(B_sz, KD, N, L)
              .permute(0, 1, 3, 2))                    # [B,KD,L,N]
    C_exp = (C.unsqueeze(2)
              .expand(-1, -1, D_dim, -1, -1)
              .reshape(B_sz, KD, N, L)
              .permute(0, 1, 3, 2))                    # [B,KD,L,N]

    dBu = (delta * u).unsqueeze(-1) * B_exp           # [B,KD,L,N]

    h = torch.zeros(B_sz, KD, N, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        h = dA[:, :, t] * h + dBu[:, :, t]
        y_t = (h * C_exp[:, :, t]).sum(-1)            # [B,KD]
        ys.append(y_t)

    y = torch.stack(ys, dim=-1)                        # [B,KD,L]
    y = y + u * D.view(1, -1, 1)
    return y


# ════════════════════════════════════════════════════════════════════════════
# BC-SS2D — Boundary-Clamped 4-Way 2D Selective Scan Block
# ════════════════════════════════════════════════════════════════════════════

class BoundaryClampedSS2D(nn.Module):
    """
    One BC-SS2D residual block.

    Runs the standard 4-way selective scan on X_low (stride-8) and modulates
    the step-size Δ_t anisotropically using directional Sobel edge maps
    extracted from X_high (stride-4, detached).

    Horizontal scans (L→R, R→L) are gated by Ex (vertical boundary map).
    Vertical  scans (T→B, B→T) are gated by Ey (horizontal boundary map).

    At training step 0 all α,β=0 → pure vanilla Mamba (stable init).
    """

    K = 4  # number of scan directions

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = max(16, d_model // 16)

        # ── input projection: X_in → [U_0 | Z]  (2*d_inner channels) ───
        self.in_proj = nn.Conv2d(d_model, self.d_inner * 2, 1, bias=False)
        self.dw_conv = nn.Conv2d(
            self.d_inner, self.d_inner, 3, padding=1,
            groups=self.d_inner, bias=True
        )

        # ── grouped projection to (dt_rank, N, N) for 4 directions ─────
        self.x_proj = nn.Conv1d(
            self.K * self.d_inner,
            self.K * (self.dt_rank + d_state * 2),
            1, groups=self.K, bias=False
        )
        self.dt_proj = nn.Conv1d(
            self.K * self.dt_rank,
            self.K * self.d_inner,
            1, groups=self.K, bias=True
        )

        # ── SSM parameters ───────────────────────────────────────────────
        # HiPPO S4D-Real init: A_n = -(n+1) → stored as log(|A|)
        A_base = torch.arange(1, d_state + 1, dtype=torch.float32)
        A_full = A_base.repeat(self.K * self.d_inner, 1)   # [K*D, N]
        self.A_log = nn.Parameter(torch.log(A_full))        # log of positive values
        self.D_skip = nn.Parameter(torch.ones(self.K * self.d_inner))

        # ── Anisotropic Δ-Firewall params (init=0 → stable start) ───────
        # Shape [1, D, 1] for broadcast over [B, D, L]
        self.alpha_h = nn.Parameter(torch.zeros(1, self.d_inner, 1))
        self.beta_h  = nn.Parameter(torch.zeros(1, self.d_inner, 1))
        self.alpha_v = nn.Parameter(torch.zeros(1, self.d_inner, 1))
        self.beta_v  = nn.Parameter(torch.zeros(1, self.d_inner, 1))

        # ── output ───────────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(32, self.d_inner)
        self.out_proj = nn.Conv2d(self.d_inner, d_model, 1, bias=False)

    # ── forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,         # [B, C, H, W]  — stride-8 low-freq input
        edge_x: torch.Tensor,    # [B, 1, H, W]  — Ex ∈ [0,1], peaks at vert. edges
        edge_y: torch.Tensor,    # [B, 1, H, W]  — Ey ∈ [0,1], peaks at horiz. edges
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        L = H * W

        # ── in-projection + DWConv ───────────────────────────────────────
        xz = self.in_proj(x)                                 # [B, 2*D, H, W]
        x_branch, z_gate = xz.chunk(2, dim=1)               # each [B, D, H, W]
        x_branch = F.silu(self.dw_conv(x_branch))           # [B, D, H, W]

        # ── 4-way spatial serialization ──────────────────────────────────
        # d=1 L→R : row-major flatten
        u_lr = x_branch.reshape(B, self.d_inner, L)
        # d=2 R→L : reverse of L→R
        u_rl = u_lr.flip(-1)
        # d=3 T→B : column-major (transpose H,W then flatten)
        u_tb = x_branch.permute(0, 1, 3, 2).contiguous().reshape(B, self.d_inner, L)
        # d=4 B→T : reverse of T→B
        u_bt = u_tb.flip(-1)

        xs = torch.cat([u_lr, u_rl, u_tb, u_bt], dim=1)    # [B, 4*D, L]

        # ── serialize edge maps to match each scan ───────────────────────
        ex_lr = edge_x.reshape(B, 1, L)
        ex_rl = ex_lr.flip(-1)
        ey_tb = edge_y.permute(0, 1, 3, 2).contiguous().reshape(B, 1, L)
        ey_bt = ey_tb.flip(-1)

        # ── project to (dt_raw, B_ssm, C_ssm) ───────────────────────────
        x_dbl = self.x_proj(xs)                              # [B, K*(R+2N), L]
        x_dbl = x_dbl.reshape(B, self.K, self.dt_rank + 2 * self.d_state, L)
        dt_raw_r, B_ssm, C_ssm = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=2
        )   # each [B, K, *, L]

        # Expand dt_raw from dt_rank → d_inner
        dt_raw = self.dt_proj(
            dt_raw_r.reshape(B, self.K * self.dt_rank, L)
        ).reshape(B, self.K, self.d_inner, L)                # [B, K, D, L]

        # ── Anisotropic Δ-Firewall ───────────────────────────────────────
        # horizontal scans: modulated by Ex (vertical boundary energy)
        dt_lr = dt_raw[:, 0] + self.alpha_h * ex_lr - self.beta_h * (1.0 - ex_lr)
        dt_rl = dt_raw[:, 1] + self.alpha_h * ex_rl - self.beta_h * (1.0 - ex_rl)
        # vertical scans: modulated by Ey (horizontal boundary energy)
        dt_tb = dt_raw[:, 2] + self.alpha_v * ey_tb - self.beta_v * (1.0 - ey_tb)
        dt_bt = dt_raw[:, 3] + self.alpha_v * ey_bt - self.beta_v * (1.0 - ey_bt)

        # Apply softplus AFTER firewall injection (must NOT use delta_softplus=True in CUDA)
        deltas = F.softplus(
            torch.cat([dt_lr, dt_rl, dt_tb, dt_bt], dim=1)  # [B, 4*D, L]
        )

        # ── build A (always negative) ────────────────────────────────────
        A = -torch.exp(self.A_log.float())                   # [K*D, N]

        # ── selective scan (CUDA kernel or reference) ────────────────────
        # Reshape B_ssm, C_ssm: [B, K, N, L]
        B_ssm = B_ssm.permute(0, 1, 3, 2).contiguous()      # [B, K, L, N] → need [B,K,N,L]
        B_ssm = B_ssm.permute(0, 1, 3, 2)                   # [B, K, N, L]
        C_ssm = C_ssm.permute(0, 1, 3, 2).contiguous()
        C_ssm = C_ssm.permute(0, 1, 3, 2)

        if HAS_MAMBA_CUDA and x.is_cuda:
            ys = selective_scan_fn(
                xs.float(), deltas.float(), A,
                B_ssm.float(), C_ssm.float(), self.D_skip.float(),
                z=None, delta_bias=None,
                delta_softplus=False,  # softplus already applied above
                return_last_state=False,
            ).to(x.dtype)
        else:
            ys = _selective_scan_ref(
                xs, deltas, A, B_ssm, C_ssm, self.D_skip
            )

        # ── un-flatten and merge 4 directional outputs → 2D ─────────────
        ys = ys.reshape(B, self.K, self.d_inner, L)

        y_lr = ys[:, 0]                                      # [B, D, L]
        y_rl = ys[:, 1].flip(-1)                             # undo R→L flip

        # undo T→B transpose: L → (W, H) → permute → (H, W) → flatten
        y_tb = ys[:, 2].reshape(B, self.d_inner, W, H).permute(0, 1, 3, 2).contiguous().reshape(B, self.d_inner, L)
        y_bt = ys[:, 3].flip(-1).reshape(B, self.d_inner, W, H).permute(0, 1, 3, 2).contiguous().reshape(B, self.d_inner, L)

        y_merged = (y_lr + y_rl + y_tb + y_bt).reshape(B, self.d_inner, H, W)

        # ── gate + output projection + residual ──────────────────────────
        y_out = self.out_norm(y_merged) * F.silu(z_gate)
        return x + self.out_proj(y_out)


# ════════════════════════════════════════════════════════════════════════════
# SDBCMambaDecoder — full 4-stage decoder
# ════════════════════════════════════════════════════════════════════════════

class SDBCMambaDecoder(nn.Module):
    """
    Spectrally Decoupled, Boundary-Clamped 4-Way SSM Segmentation Decoder.

    Args:
        in_channels:  Tuple of backbone output channel dims (C1, C2, C3, C4)
        embed_dim:    Unified decoder embedding dimension C (128 or 256)
        num_classes:  Number of segmentation classes K
        num_blocks:   Number of stacked BC-SS2D blocks M
        d_state:      SSM state dimension N
        expand:       Channel expansion factor inside BC-SS2D (D = expand*C)
    """

    def __init__(
        self,
        in_channels: tuple = (96, 192, 384, 768),
        embed_dim: int = 128,
        num_classes: int = 2,
        num_blocks: int = 2,
        d_state: int = 16,
        expand: int = 2,
    ):
        super().__init__()
        C = embed_dim

        # ── Stage I: lateral projections ─────────────────────────────────
        # Stride-4 feature: project + local DWConv residual
        self.lat_s4 = nn.Sequential(
            nn.Conv2d(in_channels[0], C, 1, bias=False),
            nn.GroupNorm(32, C),
            nn.SiLU(),
            nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),   # DWConv
            nn.GroupNorm(32, C),
            nn.SiLU(),
        )
        # Deep semantic stages (stride 8/16/32): simple 1×1 projection
        self.lat_deep = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, C, 1, bias=False),
                nn.GroupNorm(32, C),
                nn.SiLU(),
            )
            for ch in in_channels[1:]
        ])

        # ── Fixed Sobel kernels (registered as buffers, not params) ──────
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        )

        # ── Stage III: stack of BC-SS2D blocks ───────────────────────────
        self.ssm_blocks = nn.ModuleList([
            BoundaryClampedSS2D(d_model=C, d_state=d_state, expand=expand)
            for _ in range(num_blocks)
        ])

        # ── Stage IV: learnable spectral fusion gate ──────────────────────
        self.gamma_freq = nn.Parameter(torch.ones(1, C, 1, 1))

        # ── Segmentation head ─────────────────────────────────────────────
        self.seg_head = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, bias=False),
            nn.GroupNorm(32, C),
            nn.SiLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(C, num_classes, 1),
        )

        # ── Auxiliary edge head (training only) ───────────────────────────
        self.aux_edge_head = nn.Conv2d(C, 1, 1)

    # ── Sobel edge extraction (no gradient, detached) ───────────────────

    @torch.no_grad()
    def _extract_edges(self, x_high: torch.Tensor):
        """
        Extract normalized directional Sobel edge maps from x_high at stride-4.

        Returns:
            Ex: [B, 1, H/8, W/8]  — peaks at vertical object boundaries
            Ey: [B, 1, H/8, W/8]  — peaks at horizontal object boundaries
        """
        # Channel-mean magnitude → pool down to stride-8
        mag = F.avg_pool2d(
            x_high.detach().abs().mean(dim=1, keepdim=True),
            kernel_size=2, stride=2
        )   # [B, 1, H/8, W/8]

        Gx = F.conv2d(mag, self.sobel_x, padding=1).abs()
        Gy = F.conv2d(mag, self.sobel_y, padding=1).abs()

        # Per-sample normalization to [0, 1]
        Ex = Gx / (Gx.amax(dim=(2, 3), keepdim=True) + 1e-6)
        Ey = Gy / (Gy.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return Ex, Ey

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, features: list, out_size: tuple):
        """
        Args:
            features:  [F1, F2, F3, F4] from backbone at strides [4, 8, 16, 32]
            out_size:  (H, W) of the original input image (for final upsample)

        Returns:
            logits    : [B, K, H, W]  — segmentation logits
            edge_pred : [B, 1, H, W]  — binary edge logits (training supervision)
        """
        F1, F2, F3, F4 = features
        H8, W8 = F2.shape[2], F2.shape[3]

        # ── Stage I ───────────────────────────────────────────────────────
        # Deep semantic context fused at stride-8
        P2 = self.lat_deep[0](F2)                            # [B, C, H/8, W/8]
        P3 = F.interpolate(self.lat_deep[1](F3), size=(H8, W8), mode="bilinear", align_corners=False)
        P4 = F.interpolate(self.lat_deep[2](F4), size=(H8, W8), mode="bilinear", align_corners=False)
        F_deep = P2 + P3 + P4                               # [B, C, H/8, W/8]

        # High-res stride-4 feature with local DWConv inductive bias
        X_s4 = self.lat_s4(F1)                              # [B, C, H/4, W/4]

        # ── Stage II: Parameter-Free Laplacian Frequency Split ───────────
        X_pool = F.avg_pool2d(X_s4, kernel_size=2, stride=2)   # [B, C, H/8, W/8]
        X_high = X_s4 - F.interpolate(                          # [B, C, H/4, W/4]
            X_pool, size=X_s4.shape[2:], mode="bilinear", align_corners=False
        )
        X_low = X_pool + F_deep                                 # [B, C, H/8, W/8]

        # ── Stage III: Boundary-Clamped 4-Way Scan ───────────────────────
        Ex, Ey = self._extract_edges(X_high)                    # [B,1,H/8,W/8] each
        Y_ssm = X_low
        for block in self.ssm_blocks:
            Y_ssm = block(Y_ssm, Ex, Ey)                        # [B, C, H/8, W/8]

        # ── Stage IV: Spectral Re-Integration ────────────────────────────
        Y_ssm_up = F.interpolate(
            Y_ssm, size=X_s4.shape[2:], mode="bilinear", align_corners=False
        )
        Y_fused = X_high + self.gamma_freq * Y_ssm_up           # [B, C, H/4, W/4]

        # Segmentation logits → upsample 4× to full resolution
        logits = F.interpolate(
            self.seg_head(Y_fused), size=out_size,
            mode="bilinear", align_corners=False
        )   # [B, K, H, W]

        # Auxiliary edge logits → upsample 4× (discarded at inference)
        edge_pred = F.interpolate(
            self.aux_edge_head(X_high), size=out_size,
            mode="bilinear", align_corners=False
        )   # [B, 1, H, W]

        return logits, edge_pred


# ════════════════════════════════════════════════════════════════════════════
# Full model (backbone + decoder)
# ════════════════════════════════════════════════════════════════════════════

class SDBCMambaSegModel(nn.Module):
    """Backbone + SDBCMambaDecoder combined model."""

    def __init__(self, backbone: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.decoder  = decoder

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        return self.decoder(features, out_size=x.shape[2:])


def build_model(cfg: dict) -> SDBCMambaSegModel:
    from .backbone import build_backbone
    mcfg = cfg["model"]
    backbone = build_backbone(cfg)
    decoder  = SDBCMambaDecoder(
        in_channels  = backbone.out_channels,
        embed_dim    = mcfg.get("embed_dim", 128),
        num_classes  = mcfg.get("num_classes", 2),
        num_blocks   = mcfg.get("num_ssm_blocks", 2),
        d_state      = mcfg.get("d_state", 16),
        expand       = mcfg.get("expand", 2),
    )
    return SDBCMambaSegModel(backbone, decoder)

