from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from peft.utils import (
    ModulesToSaveWrapper,
    _get_submodules,
)
from timm.models.vision_transformer import Mlp
from torch.utils import checkpoint
from tqdm import tqdm
from transformers.integrations import PeftAdapterMixin

from .attn_layers import (
    Attention,
    FlashCrossMHAModified,
    FlashSelfMHAModified,
    CrossAttention,
)
from .embedders import TimestepEmbedder, PatchEmbed, timestep_embedding
from .norm_layers import RMSNorm
from .poolers import AttentionPool

from .models import FP32_Layernorm, FP32_SiLU, HunYuanDiTBlock
from .condition_fusion import MultiScaleConditionFusion


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class HunYuanControlNet(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    HunYuanDiT: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Inherit PeftAdapterMixin to be compatible with the PEFT training pipeline.

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    input_size: tuple
        The size of the input image.
    patch_size: int
        The size of the patch.
    in_channels: int
        The number of input channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    depth: int
        The number of transformer blocks.
    num_heads: int
        The number of attention heads.
    mlp_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    log_fn: callable
        The logging function.
    """

    @register_to_config
    def __init__(
        self,
        args: Any,
        input_size: tuple = (32, 32),
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        log_fn: callable = print,
    ):
        super().__init__()
        self.args = args
        self.log_fn = log_fn
        self.depth = depth
        self.learn_sigma = args.learn_sigma
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.text_states_dim = args.text_states_dim
        self.text_states_dim_t5 = args.text_states_dim_t5
        self.text_len = args.text_len
        self.text_len_t5 = args.text_len_t5
        self.norm = args.norm

        # ========================多尺度特征融合模块==============================
        self.use_condition_fusion = getattr(args, "use_condition_fusion", False)
        self.condition_fusion_in_channels = getattr(
            args,
            "condition_fusion_in_channels",
            7,
        )

        if self.use_condition_fusion:
            self.condition_fusion = MultiScaleConditionFusion(
                in_channels=self.condition_fusion_in_channels,
                base_channels=32,
                out_channels=4,
            )
            log_fn(
                f"    Enable MultiScaleConditionFusion: "
                f"{self.condition_fusion_in_channels}-channel condition -> 4-channel condition."
            )
        else:
            self.condition_fusion = None
        # =======================================================================

        use_flash_attn = args.infer_mode == "fa" or args.use_flash_attn
        if use_flash_attn:
            log_fn(f"    Enable Flash Attention.")
        qk_norm = args.qk_norm  # See http://arxiv.org/abs/2302.05442 for details.

        self.mlp_t5 = nn.Sequential(
            nn.Linear(self.text_states_dim_t5, self.text_states_dim_t5 * 4, bias=True),
            FP32_SiLU(),
            nn.Linear(self.text_states_dim_t5 * 4, self.text_states_dim, bias=True),
        )
        # learnable replace
        self.text_embedding_padding = nn.Parameter(
            torch.randn(
                self.text_len + self.text_len_t5,
                self.text_states_dim,
                dtype=torch.float32,
            )
        )

        # Attention pooling
        pooler_out_dim = 1024
        self.pooler = AttentionPool(
            self.text_len_t5,
            self.text_states_dim_t5,
            num_heads=8,
            output_dim=pooler_out_dim,
        )

        # Dimension of the extra input vectors
        self.extra_in_dim = pooler_out_dim

        if args.size_cond:
            # Image size and crop size conditions
            self.extra_in_dim += 6 * 256

        if args.use_style_cond:
            # Here we use a default learned embedder layer for future extension.
            self.style_embedder = nn.Embedding(1, hidden_size)
            self.extra_in_dim += hidden_size

        # Text embedding for `add`
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.extra_embedder = nn.Sequential(
            nn.Linear(self.extra_in_dim, hidden_size * 4),
            FP32_SiLU(),
            nn.Linear(hidden_size * 4, hidden_size, bias=True),
        )

        # Image embedding
        num_patches = self.x_embedder.num_patches
        log_fn(f"    Number of tokens: {num_patches}")

        # HUnYuanDiT Blocks
        self.blocks = nn.ModuleList(
            [
                HunYuanDiTBlock(
                    hidden_size=hidden_size,
                    c_emb_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    text_states_dim=self.text_states_dim,
                    use_flash_attn=use_flash_attn,
                    qk_norm=qk_norm,
                    norm_type=self.norm,
                    skip=False,
                )
                for _ in range(19)
            ]
        )

        # Input zero linear for the first block
        self.before_proj = zero_module(nn.Linear(self.hidden_size, self.hidden_size))

        # Output zero linear for the every block
        self.after_proj_list = nn.ModuleList(
            [
                zero_module(nn.Linear(self.hidden_size, self.hidden_size))
                for _ in range(len(self.blocks))
            ]
        )

        self.fix_weight_modules = [
            "mlp_t5",
            "text_embedding_padding",
            "pooler",
            "style_embedder",
            "x_embedder",
            "t_embedder",
            "extra_embedder",
        ]

    def log_trainable_parameter_summary(self):
        groups = {
            "condition_fusion": 0,
            "before_proj": 0,
            "after_proj_list": 0,
            "attention": 0,
            "norm": 0,
            "mlp": 0,
            "embedder_or_pooler": 0,
            "others": 0,
        }

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            n = param.numel()

            if name.startswith("condition_fusion"):
                groups["condition_fusion"] += n
            elif name.startswith("before_proj"):
                groups["before_proj"] += n
            elif name.startswith("after_proj_list"):
                groups["after_proj_list"] += n
            elif ".attn" in name or "Wqkv" in name or "q_proj" in name or "kv_proj" in name or "out_proj" in name:
                groups["attention"] += n
            elif "norm" in name.lower() or "layernorm" in name.lower():
                groups["norm"] += n
            elif "mlp" in name.lower():
                groups["mlp"] += n
            elif (
                "embedder" in name
                or "embedding" in name
                or "pooler" in name
                or "x_embedder" in name
                or "t_embedder" in name
            ):
                groups["embedder_or_pooler"] += n
            else:
                groups["others"] += n

        total_trainable = sum(groups.values())
        self.log_fn("=" * 80)
        self.log_fn("Trainable parameter summary by group:")
        self.log_fn("=" * 80)

        if total_trainable == 0:
            self.log_fn("No trainable parameters found.")
            self.log_fn("=" * 80)
            return

        for k, v in groups.items():
            if v > 0:
                self.log_fn(f"{k:24s}: {v:,} ({100.0 * v / total_trainable:.2f}%)")

        self.log_fn(f"{'TOTAL':24s}: {total_trainable:,}")
        self.log_fn("=" * 80)

    def log_trainable_parameters(self, max_lines=300):
        trainable_params = []
        frozen_params = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param.numel()))
            else:
                frozen_params.append((name, param.numel()))

        trainable_num = sum(n for _, n in trainable_params)
        frozen_num = sum(n for _, n in frozen_params)
        total_num = trainable_num + frozen_num

        self.log_fn("=" * 80)
        self.log_fn("Trainable parameter details:")
        self.log_fn("=" * 80)

        for i, (name, numel) in enumerate(trainable_params):
            if i >= max_lines:
                self.log_fn(
                    f"... {len(trainable_params) - max_lines} more trainable parameter tensors omitted."
                )
                break
            self.log_fn(f"[TRAIN] {name:90s} {numel:,}")

        self.log_fn("=" * 80)
        self.log_fn(
            f"Trainable params: {trainable_num:,} / {total_num:,} "
            f"({100.0 * trainable_num / total_num:.2f}%)"
        )
        self.log_fn(
            f"Frozen params:    {frozen_num:,} / {total_num:,} "
            f"({100.0 * frozen_num / total_num:.2f}%)"
        )
        self.log_fn("=" * 80)

    def _set_blocks_train_for_checkpointing(self):
        if getattr(self.args, "gradient_checkpointing", False):
            for block in self.blocks:
                block.train()
            self.log_fn(
                "    Keep ControlNet blocks in train() mode to enable gradient checkpointing."
            )

    def _keep_fixed_modules_eval(self):
        self.mlp_t5.eval()
        self.pooler.eval()
        self.x_embedder.eval()
        self.t_embedder.eval()
        self.extra_embedder.eval()

        if self.args.use_style_cond:
            self.style_embedder.eval()


# ======================ControlNet微调模式==============================
    def _set_all_requires_grad(self, flag: bool):
        for p in self.parameters():
            p.requires_grad_(flag)


    def _set_module_trainable(self, module, train: bool = True):
        if module is None:
            return
        module.requires_grad_(train)
        if train:
            module.train()
        else:
            module.eval()


    def _enable_attention_trainable(self):
        attn_classes = (
            Attention,
            FlashCrossMHAModified,
            FlashSelfMHAModified,
            CrossAttention,
        )

        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, attn_classes):
                    module.requires_grad_(True)
                    module.train()


    def _enable_norm_trainable(self):
        norm_classes = (
            nn.LayerNorm,
            RMSNorm,
            FP32_Layernorm,
        )

        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, norm_classes):
                    module.requires_grad_(True)
                    module.train()


# ======================================================================




    def check_condition_validation(self, image_meta_size, style):
        if self.args.size_cond is None and image_meta_size is not None:
            raise ValueError(
                f"When `size_cond` is None, `image_meta_size` should be None, but got "
                f"{type(image_meta_size)}. "
            )
        if self.args.size_cond is not None and image_meta_size is None:
            raise ValueError(
                f"When `size_cond` is not None, `image_meta_size` should not be None. "
            )
        if not self.args.use_style_cond and style is not None:
            raise ValueError(
                f"When `use_style_cond` is False, `style` should be None, but got {type(style)}. "
            )
        if self.args.use_style_cond and style is None:
            raise ValueError(
                f"When `use_style_cond` is True, `style` should be not None."
            )

    def enable_gradient_checkpointing(self):
        for block in self.blocks:
            block.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        for block in self.blocks:
            block.gradient_checkpointing = False

    def from_dit(self, dit):
        """
        Load the parameters from a pre-trained HunYuanDiT model.

        Parameters
        ----------
        dit: HunYuanDiT
            The pre-trained HunYuanDiT model.
        """

        self.mlp_t5.load_state_dict(dit.mlp_t5.state_dict())

        self.text_embedding_padding.data = dit.text_embedding_padding.data
        self.pooler.load_state_dict(dit.pooler.state_dict())
        if self.args.use_style_cond:
            self.style_embedder.load_state_dict(dit.style_embedder.state_dict())
        self.x_embedder.load_state_dict(dit.x_embedder.state_dict())
        self.t_embedder.load_state_dict(dit.t_embedder.state_dict())
        self.extra_embedder.load_state_dict(dit.extra_embedder.state_dict())

        for i, block in enumerate(self.blocks):
            block.load_state_dict(dit.blocks[i].state_dict())

    def set_trainable(self):
        mode = getattr(self.args, "controlnet_train_mode", "all")

        # First freeze everything.
        self._set_all_requires_grad(False)
        self.eval()
        self._set_blocks_train_for_checkpointing()
        self._keep_fixed_modules_eval()

        if mode == "all":
            # Keep the original behavior.
            self.blocks.requires_grad_(True)
            self.before_proj.requires_grad_(True)
            self.after_proj_list.requires_grad_(True)

            self.blocks.train()
            self.before_proj.train()
            self.after_proj_list.train()

            if self.condition_fusion is not None:
                self.condition_fusion.requires_grad_(True)
                self.condition_fusion.train()


        elif mode == "attn_zero":
            # Train only attention layers and newly initialized ControlNet layers.
            self._enable_attention_trainable()

            self.before_proj.requires_grad_(True)
            self.after_proj_list.requires_grad_(True)
            self.before_proj.train()
            self.after_proj_list.train()

            if self.condition_fusion is not None:
                self.condition_fusion.requires_grad_(True)
                self.condition_fusion.train()


        elif mode == "attn_norm_zero":
            # Slightly stronger version: attention + norm + zero/new layers.
            self._enable_attention_trainable()
            self._enable_norm_trainable()

            self.before_proj.requires_grad_(True)
            self.after_proj_list.requires_grad_(True)
            self.before_proj.train()
            self.after_proj_list.train()

            if self.condition_fusion is not None:
                self.condition_fusion.requires_grad_(True)
                self.condition_fusion.train()



        elif mode == "zero_only":
            # Minimum trainable mode.
            self.before_proj.requires_grad_(True)
            self.after_proj_list.requires_grad_(True)
            self.before_proj.train()
            self.after_proj_list.train()

            if self.condition_fusion is not None:
                self.condition_fusion.requires_grad_(True)
                self.condition_fusion.train()



        else:
            raise ValueError(f"Unknown controlnet_train_mode: {mode}")
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())

        self.log_fn(f"    ControlNet train mode: {mode}")
        self.log_fn(
            f"    ControlNet trainable params: {trainable:,} / {total:,} "
            f"({100.0 * trainable / total:.2f}%)"
        )
        self.log_trainable_parameter_summary()
        self.log_trainable_parameters(max_lines=300)

    def forward(
        self,
        x,
        t,
        condition,
        encoder_hidden_states=None,
        text_embedding_mask=None,
        encoder_hidden_states_t5=None,
        text_embedding_mask_t5=None,
        image_meta_size=None,
        style=None,
        cos_cis_img=None,
        sin_cis_img=None,
        return_dict=True,
    ):
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x: torch.Tensor
            (B, D, H, W)
        t: torch.Tensor
            (B)
        encoder_hidden_states: torch.Tensor
            CLIP text embedding, (B, L_clip, D)
        text_embedding_mask: torch.Tensor
            CLIP text embedding mask, (B, L_clip)
        encoder_hidden_states_t5: torch.Tensor
            T5 text embedding, (B, L_t5, D)
        text_embedding_mask_t5: torch.Tensor
            T5 text embedding mask, (B, L_t5)
        image_meta_size: torch.Tensor
            (B, 6)
        style: torch.Tensor
            (B)
        cos_cis_img: torch.Tensor
        sin_cis_img: torch.Tensor
        return_dict: bool
            Whether to return a dictionary.
        """
        text_states = encoder_hidden_states  # 2,77,1024
        text_states_t5 = encoder_hidden_states_t5  # 2,256,2048
        text_states_mask = text_embedding_mask.bool()  # 2,77
        text_states_t5_mask = text_embedding_mask_t5.bool()  # 2,256
        b_t5, l_t5, c_t5 = text_states_t5.shape
        text_states_t5 = self.mlp_t5(text_states_t5.view(-1, c_t5))
        text_states = torch.cat(
            [text_states, text_states_t5.view(b_t5, l_t5, -1)], dim=1
        )  # 2,205，1024
        clip_t5_mask = torch.cat([text_states_mask, text_states_t5_mask], dim=-1)

        clip_t5_mask = clip_t5_mask
        text_states = torch.where(
            clip_t5_mask.unsqueeze(2),
            text_states,
            self.text_embedding_padding.to(text_states),
        )

        _, _, oh, ow = x.shape
        th, tw = oh // self.patch_size, ow // self.patch_size

        # ========================= Build time and image embedding =========================
        t = self.t_embedder(t)
        x = self.x_embedder(x)

        # ============================================================
        # Build condition tokens
        # ============================================================
        if condition.ndim != 4:
            raise ValueError(
                "ControlNet latent condition must have shape [B, C, H, W], "
                f"but got {tuple(condition.shape)}."
            )

        if condition.shape[1] != self.in_channels:
            raise ValueError(
                f"ControlNet expects {self.in_channels}-channel latent condition, "
                f"but got shape {tuple(condition.shape)}."
            )

        condition_tokens = self.x_embedder(
            condition
        )

        if condition_tokens.shape != x.shape:
            raise ValueError(
                "Condition token shape does not match noisy token shape: "
                f"x={tuple(x.shape)}, "
                f"condition={tuple(condition_tokens.shape)}."
            )

        # Get image RoPE embedding according to `reso`lution.
        freqs_cis_img = (cos_cis_img, sin_cis_img)

        # ========================= Concatenate all extra vectors =========================
        # Build text tokens with pooling
        extra_vec = self.pooler(encoder_hidden_states_t5)

        self.check_condition_validation(image_meta_size, style)
        # Build image meta size tokens if applicable
        if image_meta_size is not None:
            image_meta_size = timestep_embedding(
                image_meta_size.view(-1), 256
            )  # [B * 6, 256]
            if self.args.use_fp16:
                image_meta_size = image_meta_size.half()
            image_meta_size = image_meta_size.view(-1, 6 * 256)
            extra_vec = torch.cat(
                [extra_vec, image_meta_size], dim=1
            )  # [B, D + 6 * 256]

        # Build style tokens
        if style is not None:
            style_embedding = self.style_embedder(style)
            extra_vec = torch.cat([extra_vec, style_embedding], dim=1)

        # Concatenate all extra vectors
        c = t + self.extra_embedder(extra_vec)  # [B, D]

        # ========================= Forward pass through HunYuanDiT blocks =========================
        controls = []
        x = x + self.before_proj(condition_tokens)  # add condition
        for layer, block in enumerate(self.blocks):
            # --------------------------------------------------------
            # Existing ControlNet Transformer block
            # --------------------------------------------------------
            x = block(
                x=x,
                c=c,
                text_states=text_states,
                freq_cis_img=freqs_cis_img,
            )

            # --------------------------------------------------------
            # Existing zero-convolution control output
            # --------------------------------------------------------
            controls.append(
                self.after_proj_list[layer](x)
            )


        if return_dict:
            return {"controls": controls}
        return controls
