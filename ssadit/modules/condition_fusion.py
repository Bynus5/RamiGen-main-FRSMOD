import torch
import torch.nn as nn
import torch.nn.functional as F



class MultiScaleConditionFusion(nn.Module):
    """
    Latent-space multi-conditional fusion module.

    Input:
        B x C x H x W

    For edge/depth/mask:
        edge  : 1 channel
        depth : 3 channels
        mask  : 3 channels
        total : 7 channels

    Output:
        B x 4 x (H/8) x (W/8)

    This follows the idea of CRS-Diff:
        multiple image conditions
        -> convolutional feature extractor
        -> latent/noise-aligned condition feature.

    The output is directly fed into ControlNet and should NOT be
    passed through the VAE encoder again.
    """

    def __init__(self, in_channels=7, base_channels=32, out_channels=4):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.feature_channels = base_channels * 4

        # H, W -> H/2, W/2 1024*1024 -> 512*512
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )

        # H/2, W/2 -> H/4, W/4 512*512 -> 256*256
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
        )

        # H/4, W/4 -> H/8, W/8 256*256 -> 128*128
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
        )

        # Multi-scale feature extraction at latent resolution.
        self.branch1 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
        )

        self.branch2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(base_channels * 12, base_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),

            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),

            nn.Conv2d(base_channels * 2, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """Fuse raw conditions [B,C,H,W] into [B,4,H/8,W/8]."""
        if not torch.is_tensor(x):
            raise TypeError(
                "MultiScaleConditionFusion expects a torch.Tensor, "
                f"but got {type(x)}."
            )

        if x.ndim != 4:
            raise ValueError(
                "MultiScaleConditionFusion expects [B, C, H, W], "
                f"but got shape {tuple(x.shape)}."
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"MultiScaleConditionFusion expects {self.in_channels} channels, "
                f"but got {x.shape[1]} channels. "
                f"Full shape: {tuple(x.shape)}."
            )

        input_height, input_width = x.shape[-2:]

        if input_height % 8 != 0 or input_width % 8 != 0:
            raise ValueError(
                "Condition height and width must be divisible by 8, "
                f"but got H={input_height}, W={input_width}."
            )

        # ------------------------------------------------------------
        # Shared convolutional feature extractor
        # ------------------------------------------------------------
        x = self.stem(x)      # [B, 32,  H/2, W/2]
        x = self.down1(x)     # [B, 64,  H/4, W/4]
        x = self.down2(x)     # [B, 128, H/8, W/8]

        latent_height, latent_width = x.shape[-2:]

        # ------------------------------------------------------------
        # Fine-scale branch
        # Native latent resolution; focuses more on local structures.
        # ------------------------------------------------------------
        fine = self.branch1(x)

        # ------------------------------------------------------------
        # Medium-scale branch
        # Larger receptive field, then aligned to latent resolution.
        # ------------------------------------------------------------
        medium_native = self.branch2(x)

        medium = F.interpolate(
            medium_native,
            size=(latent_height, latent_width),
            mode="bilinear",
            align_corners=False,
        )

        # ------------------------------------------------------------
        # Coarse-scale branch
        # Largest receptive field, then aligned to latent resolution.
        # ------------------------------------------------------------
        coarse_native = self.branch4(x)

        coarse = F.interpolate(
            coarse_native,
            size=(latent_height, latent_width),
            mode="bilinear",
            align_corners=False,
        )

        # ------------------------------------------------------------
        # Existing unified latent condition
        # ------------------------------------------------------------
        latent = self.fuse(
            torch.cat(
                [fine, medium, coarse],
                dim=1,
            )
        )

        if latent.shape[1] != self.out_channels:
            raise ValueError(
                f"Fusion output should have {self.out_channels} channels, "
                f"but got shape {tuple(latent.shape)}."
            )

        expected_spatial_shape = (
            input_height // 8,
            input_width // 8,
        )

        if latent.shape[-2:] != expected_spatial_shape:
            raise ValueError(
                "Fusion output spatial size is incorrect: "
                f"expected {expected_spatial_shape}, "
                f"got {tuple(latent.shape[-2:])}."
            )

        return latent
