import os
import random
from pathlib import Path
import ast
import numpy as np
import re
import json
import time
from functools import partial
from PIL import Image

import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset

from IndexKits.index_kits import (
    ArrowIndexV2,
    MultiResolutionBucketIndexV2,
    MultiIndexV2,
)


class TextImageArrowStream(Dataset):
    def __init__(
        self,
        args,
        resolution=512,
        random_flip=None,
        enable_CN=True,
        log_fn=print,
        index_file=None,
        multireso=False,
        batch_size=-1,
        world_size=1,
        random_shrink_size_cond=False,
        merge_src_cond=False,
        uncond_p=0.0,
        uncond_p_img=0.0,
        text_ctx_len=77,
        tokenizer=None,
        uncond_p_t5=0.0,
        text_ctx_len_t5=256,
        tokenizer_t5=None,
    ):
        self.args = args
        self.resolution = resolution
        self.log_fn = lambda x: log_fn(f"    {Path(__file__).stem} | " + x)

        self.random_flip = random_flip
        self.enable_CN = enable_CN
        self.index_file = index_file
        self.multireso = multireso
        self.batch_size = batch_size
        self.world_size = world_size
        self.index_manager = self.load_index()

        # clip params
        self.uncond_p = uncond_p
        self.text_ctx_len = text_ctx_len
        self.tokenizer = tokenizer
        self.uncond_p_img = uncond_p_img

        # t5 params
        self.uncond_p_t5 = uncond_p_t5
        self.text_ctx_len_t5 = text_ctx_len_t5
        self.tokenizer_t5 = tokenizer_t5

        # size condition
        self.random_shrink_size_cond = random_shrink_size_cond
        self.merge_src_cond = merge_src_cond
        self.is_ipa = getattr(args, 'is_ipa', False)

        # control switch
        self.use_control = getattr(args, "use_control", False)
        self.batchwise_racem = self.use_control and getattr(args, "racem_mode", "preprocess") == "batch_wise"
        if self.batchwise_racem:
            label_dir = getattr(args, "racem_label_dir", None)
            if not label_dir or not Path(label_dir).is_dir():
                raise FileNotFoundError("Online RACEM requires an existing --racem-label-dir")
            self.racem_label_dir = Path(label_dir)

        # supported control-image extensions, in priority order
        self.control_image_exts = getattr(
            args,
            "control_image_exts",
            [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"],
        )
        self.control_image_exts = [
            ext if ext.startswith(".") else f".{ext}"
            for ext in self.control_image_exts
        ]

        assert isinstance(
            resolution, int
        ), f"resolution must be an integer, got {resolution}"

        self.ti2i_transform = T.Compose([
                T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                lambda x: x.convert("RGB"),
                T.ToTensor(),
                T.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        if self.merge_src_cond:
            self.log_fn("Enable merging src condition: (oriW, oriH) --> ((WH)**0.5, (WH)**0.5)")
        self.log_fn("Enable image_meta_size condition (original_size, target_size, crop_coords)")
        self.log_fn(f"Control image extensions: {self.control_image_exts}")


# ============多尺度特征融合数据读取======================

    def _get_control_dir(self, control_type):
        cond_base_dir = os.environ.get(
            "COND_BASE_DIR",
            "/data/private/nfs_ds1/yyh/dataset/DIOR/control_condition"
        )
        sub_type = os.environ.get("SUB_TYPE", "full")

        cond_folder = f"{control_type}_{sub_type}"
        return os.path.join(cond_base_dir, control_type, cond_folder)


    def _load_control_image(
        self,
        cond_dir,
        base_name,
        origin_size,
        mode="RGB",
        control_type=None,
    ):
        """
        Load one control image in strict mode.

        Requirements:
        1. The control image must exist.
        2. The control image must be readable.
        3. Its original spatial size must exactly match the target image.
        4. The returned channel mode must match the requested mode.

        Args:
            cond_dir:
                Directory containing the control images.

            base_name:
                Filename stem shared with the target image.

            origin_size:
                Original target-image size in PIL format: (width, height).

            mode:
                PIL conversion mode:
                    "L"   -> one channel
                    "RGB" -> three channels

            control_type:
                Optional name used for clearer error messages.
        """
        cond_path = self._resolve_control_image_path(
            cond_dir,
            base_name,
        )

        type_name = (
            str(control_type)
            if control_type is not None
            else str(mode)
        )

        if cond_path is None:
            expected_extensions = ", ".join(self.control_image_exts)

            raise FileNotFoundError(
                f"Missing {type_name} control image.\n"
                f"Base name: {base_name}\n"
                f"Directory: {cond_dir}\n"
                f"Supported extensions: {expected_extensions}"
            )

        try:
            # Use a context manager so the underlying file handle is closed.
            with Image.open(cond_path) as control_image:
                control_size = tuple(control_image.size)
                expected_size = tuple(origin_size)

                if control_size != expected_size:
                    raise ValueError(
                        f"Spatial-size mismatch for {type_name} control image.\n"
                        f"Control path: {cond_path}\n"
                        f"Control size: {control_size}\n"
                        f"Target-image size: {expected_size}\n"
                        "The target image and its condition images must be "
                        "pixel-aligned before resizing and cropping."
                    )

                # copy() detaches the returned PIL image from the closed file.
                control_image = control_image.convert(mode).copy()

        except ValueError:
            # Preserve the detailed spatial-alignment error above.
            raise

        except Exception as e:
            raise RuntimeError(
                f"Failed to read {type_name} control image.\n"
                f"Path: {cond_path}\n"
                f"Original error: {type(e).__name__}: {e}"
            ) from e

        expected_mode = str(mode)

        if control_image.mode != expected_mode:
            raise RuntimeError(
                f"Unexpected image mode after conversion for {type_name}.\n"
                f"Path: {cond_path}\n"
                f"Expected mode: {expected_mode}\n"
                f"Actual mode: {control_image.mode}"
            )

        return control_image


    def _load_fusion_control_images(
        self,
        base_name,
        origin_size,
    ):
        edge_dir = self._get_control_dir("edge")
        depth_dir = self._get_control_dir("depth")
        mask_dir = self._get_control_dir("mask")

        edge_img = self._load_control_image(
            cond_dir=edge_dir,
            base_name=base_name,
            origin_size=origin_size,
            mode="L",
            control_type="edge",
        )

        depth_img = self._load_control_image(
            cond_dir=depth_dir,
            base_name=base_name,
            origin_size=origin_size,
            mode="RGB",
            control_type="depth",
        )

        mask_img = self._load_control_image(
            cond_dir=mask_dir,
            base_name=base_name,
            origin_size=origin_size,
            mode="RGB",
            control_type="mask",
        )

        return {
            "edge": edge_img,
            "depth": depth_img,
            "mask": mask_img,
        }


    def _resize_and_crop_control(
        self,
        cond_img,
        origin_size,
        target_size,
        crops_coords_top_left,
        resample,
    ):
        aspect_ratio = float(origin_size[0]) / float(origin_size[1])

        if origin_size[0] < origin_size[1]:
            new_width = target_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_size[1]
            new_width = int(new_height * aspect_ratio)

        cond_img = cond_img.resize((new_width, new_height), resample=resample)

        x_start, y_start = crops_coords_top_left
        cond_img = cond_img.crop(
            (x_start, y_start, x_start + target_size[0], y_start + target_size[1])
        )

        return cond_img


    def _fusion_controls_to_tensor(self, cond_imgs):
        edge = TF.to_tensor(cond_imgs["edge"])      # 1,H,W
        depth = TF.to_tensor(cond_imgs["depth"])    # 3,H,W
        mask = TF.to_tensor(cond_imgs["mask"])      # 3,H,W

        cond_tensor = torch.cat([edge, depth, mask], dim=0)  # 7,H,W
        cond_tensor = cond_tensor * 2.0 - 1.0

        return cond_tensor

    def load_index(self):
        multireso = self.multireso
        index_file = self.index_file
        batch_size = self.batch_size
        world_size = self.world_size

        if multireso:
            if isinstance(index_file, (list, tuple)):
                if len(index_file) > 1:
                    raise ValueError(f"When enabling multireso, index_file should be a single file, but got {index_file}")
                index_file = index_file[0]
            index_manager = MultiResolutionBucketIndexV2(index_file, batch_size, world_size)
            self.log_fn(f"Using MultiResolutionBucketIndexV2: {len(index_manager):,}")
        else:
            if isinstance(index_file, str):
                index_file = [index_file]
            if len(index_file) == 1:
                index_manager = ArrowIndexV2(index_file[0])
                self.log_fn(f"Using ArrowIndexV2: {len(index_manager):,}")
            else:
                index_manager = MultiIndexV2(index_file)
                self.log_fn(f"Using MultiIndexV2: {len(index_manager):,}")

        return index_manager

    def shuffle(self, seed, fast=False):
        self.index_manager.shuffle(seed, fast=fast)

    def get_raw_image(self, index, image_key="image"):
        try:
            image = self.index_manager.get_image(
                index,
                image_key,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to read target image from Arrow index.\n"
                f"Index: {index}\n"
                f"Image key: {image_key}\n"
                f"Original error: {type(e).__name__}: {e}"
            ) from e

        if image is None:
            raise RuntimeError(
                f"Arrow index returned None for target image.\n"
                f"Index: {index}\n"
                f"Image key: {image_key}"
            )

        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Expected a PIL.Image.Image from Arrow index, "
                f"but got {type(image)} at index={index}."
            )

        return image.convert("RGB")

    @staticmethod
    def random_crop_image(image, origin_size, target_size):
        aspect_ratio = float(origin_size[0]) / float(origin_size[1])
        if origin_size[0] < origin_size[1]:
            new_width = target_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_size[1]
            new_width = int(new_height * aspect_ratio)

        image = image.resize((new_width, new_height), Image.LANCZOS)

        if new_width > target_size[0]:
            x_start = random.randint(0, new_width - target_size[0])
            y_start = 0
        else:
            x_start = 0
            y_start = random.randint(0, new_height - target_size[1])
        image_crop = image.crop(
            (x_start, y_start, x_start + target_size[0], y_start + target_size[1])
        )
        crops_coords_top_left = (x_start, y_start)
        return image_crop, crops_coords_top_left

    def get_style(self, index):
        return 0

    def _resolve_control_image_path(self, cond_dir, base_name):
        for ext in self.control_image_exts:
            candidate = os.path.join(cond_dir, f"{base_name}{ext}")
            if os.path.exists(candidate):
                return candidate
            upper_candidate = os.path.join(cond_dir, f"{base_name}{ext.upper()}")
            if os.path.exists(upper_candidate):
                return upper_candidate

        cond_dir_path = Path(cond_dir)
        if cond_dir_path.exists():
            matched = sorted(cond_dir_path.glob(f"{base_name}.*"))
            image_like = [
                str(p) for p in matched
                if p.suffix.lower() in {ext.lower() for ext in self.control_image_exts}
            ]
            if image_like:
                return image_like[0]

        return None

    # def _load_control_image(self, cond_dir, base_name, origin_size):
    #     cond_path = self._resolve_control_image_path(cond_dir, base_name)
    #     if cond_path is None:
    #         self.log_fn(
    #             f"Warning: 找不到控制图 {os.path.join(cond_dir, base_name)}"
    #             f"{{{', '.join(self.control_image_exts)}}}，已使用全黑图替代。"
    #         )
    #         return Image.new("RGB", origin_size, (0, 0, 0))

    #     try:
    #         return Image.open(cond_path).convert("RGB")
    #     except Exception as e:
    #         self.log_fn(
    #             f"Warning: 控制图读取失败 {cond_path} | {e}，已使用全黑图替代。"
    #         )
    #         return Image.new("RGB", origin_size, (0, 0, 0))

    def get_image_with_hwxy(self, index, image_key="image"):
        image = self.get_raw_image(index, image_key=image_key)
        origin_size = image.size
        decoded_origin_size = tuple(origin_size)
        try:
            meta_width = int(
                self.index_manager.get_attribute(index, "width")
            )
            meta_height = int(
                self.index_manager.get_attribute(index, "height")
            )
        except Exception:
            meta_width = None
            meta_height = None

        if meta_width is not None and meta_height is not None:
            metadata_size = (meta_width, meta_height)

            if metadata_size != tuple(origin_size):
                raise ValueError(
                    f"Arrow image metadata does not match decoded image size.\n"
                    f"Index: {index}\n"
                    f"Metadata size: {metadata_size}\n"
                    f"Decoded image size: {tuple(origin_size)}"
                )

        cond_img = None
        cond_imgs = None

        if self.use_control:
            try:
                img_path_str = self.index_manager.get_attribute(
                    index,
                    "img_path",
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read 'img_path' from Arrow index.\n"
                    f"Index: {index}\n"
                    f"Original error: {type(e).__name__}: {e}"
                ) from e

            if isinstance(img_path_str, bytes):
                img_path_str = img_path_str.decode(
                    "utf-8",
                    errors="strict",
                )

            img_path_str = str(img_path_str).strip()

            if not img_path_str:
                raise ValueError(
                    f"Empty 'img_path' in Arrow index at index={index}."
                )

            base_name = Path(img_path_str).stem

            if not base_name:
                raise ValueError(
                    f"Cannot derive basename from img_path={img_path_str!r} "
                    f"at index={index}."
                )

            if getattr(self, "batchwise_racem", False):
                from RACEM.algorithms.labels import read_labels
                racem_labels = read_labels(self.racem_label_dir / f"{base_name}.txt")
            elif getattr(self.args, "use_condition_fusion", False):
                cond_imgs = self._load_fusion_control_images(base_name, origin_size)
            else:
                control_type = str(self.args.control_type)

                cond_dir = self._get_control_dir(
                    control_type
                )

                cond_img = self._load_control_image(
                    cond_dir=cond_dir,
                    base_name=base_name,
                    origin_size=origin_size,
                    mode="RGB",
                    control_type=control_type,
                )
        if self.multireso:
            target_size = self.index_manager.get_target_size(index)
            image, crops_coords_top_left = self.index_manager.resize_and_crop(
                image, target_size, resample=Image.LANCZOS, crop_type="random"
            )
        else:
            target_size = (self.resolution, self.resolution)
            image, crops_coords_top_left = self.random_crop_image(
                image, origin_size, target_size
            )

        if self.use_control and not getattr(self, "batchwise_racem", False):
            if getattr(self.args, "use_condition_fusion", False):
                cond_imgs["edge"] = self._resize_and_crop_control(
                    cond_imgs["edge"],
                    origin_size,
                    target_size,
                    crops_coords_top_left,
                    resample=Image.LANCZOS,
                )
                cond_imgs["depth"] = self._resize_and_crop_control(
                    cond_imgs["depth"],
                    origin_size,
                    target_size,
                    crops_coords_top_left,
                    resample=Image.LANCZOS,
                )
                cond_imgs["mask"] = self._resize_and_crop_control(
                    cond_imgs["mask"],
                    origin_size,
                    target_size,
                    crops_coords_top_left,
                    resample=Image.NEAREST,
                )
            else:
                cond_img = self._resize_and_crop_control(
                    cond_img,
                    origin_size,
                    target_size,
                    crops_coords_top_left,
                    resample=Image.LANCZOS,
                )

        did_horizontal_flip = False
        if self.random_flip and random.random() < 0.5:
            did_horizontal_flip = True
            image = TF.hflip(image)

            if self.use_control and not getattr(self, "batchwise_racem", False):
                if getattr(self.args, "use_condition_fusion", False):
                    cond_imgs = {
                        k: TF.hflip(v)
                        for k, v in cond_imgs.items()
                    }
                else:
                    cond_img = TF.hflip(cond_img)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, [0.5], [0.5])

        cond_tensor = None
        if self.use_control:
            if getattr(self, "batchwise_racem", False):
                # Only CPU geometry/labels live in workers. Models run once per
                # batch in the trainer, never in Dataset.__getitem__.
                cond_tensor = torch.empty((0, image_tensor.shape[-2], image_tensor.shape[-1]))
            elif getattr(self.args, "use_condition_fusion", False):
                cond_tensor = self._fusion_controls_to_tensor(cond_imgs)
            else:
                cond_tensor = TF.to_tensor(cond_img)
                cond_tensor = TF.normalize(cond_tensor, [0.5], [0.5])

        if self.use_control and not getattr(self, "batchwise_racem", False) and getattr(self.args, "use_condition_fusion", False):
            expected_channels = getattr(self.args, "condition_fusion_in_channels", 7)
            assert cond_tensor.shape[0] == expected_channels, (
                f"Fusion condition channel mismatch: "
                f"expected {expected_channels}, got {cond_tensor.shape[0]}, "
                f"shape={cond_tensor.shape}"
            )

        if self.is_ipa:
            img_for_clip_tensor = self.ti2i_transform(image)

        if self.random_shrink_size_cond:
            origin_size = (
                1024 if origin_size[0] < 1024 else origin_size[0],
                1024 if origin_size[1] < 1024 else origin_size[1],
            )

        if self.merge_src_cond:
            val = (origin_size[0] * origin_size[1]) ** 0.5
            origin_size = (val, val)

        image_meta_size = (
            tuple(origin_size) + tuple(target_size) + tuple(crops_coords_top_left)
        )

        kwargs = {
            "image_meta_size": image_meta_size,
            "style": self.get_style(index),
        }
        if getattr(self, "batchwise_racem", False):
            from RACEM.algorithms.labels import transform_labels
            kwargs["racem_labels"] = transform_labels(
                racem_labels, source_size=decoded_origin_size, target_size=target_size,
                crop_xy=crops_coords_top_left, horizontal_flip=did_horizontal_flip,
                bbox_type=getattr(self.args, "racem_bbox_type", "obb"), multireso=self.multireso,
            )

        if self.use_control:
            if self.is_ipa:
                return image_tensor, cond_tensor, img_for_clip_tensor, kwargs
            else:
                return image_tensor, cond_tensor, kwargs
        else:
            if self.is_ipa:
                return image_tensor, img_for_clip_tensor, kwargs
            else:
                return image_tensor, kwargs

    def get_text_info_with_encoder(self, description):
        pad_num = 0
        text_inputs = self.tokenizer(
            description,
            padding="max_length",
            max_length=self.text_ctx_len,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids[0]
        attention_mask = text_inputs.attention_mask[0].bool()
        if pad_num > 0:
            attention_mask[1 : pad_num + 1] = False
        return description, text_input_ids, attention_mask

    def fill_t5_token_mask(self, fill_tensor, fill_number, setting_length):
        fill_length = setting_length - fill_tensor.shape[1]
        if fill_length > 0:
            fill_tensor = torch.cat(
                (fill_tensor, fill_number * torch.ones(1, fill_length)), dim=1
            )
        return fill_tensor

    def get_text_info_with_encoder_t5(self, description_t5):
        text_tokens_and_mask = self.tokenizer_t5(
            description_t5,
            max_length=self.text_ctx_len_t5,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids_t5 = self.fill_t5_token_mask(
            text_tokens_and_mask["input_ids"],
            fill_number=1,
            setting_length=self.text_ctx_len_t5,
        ).long()
        attention_mask_t5 = self.fill_t5_token_mask(
            text_tokens_and_mask["attention_mask"],
            fill_number=0,
            setting_length=self.text_ctx_len_t5,
        ).bool()
        return description_t5, text_input_ids_t5, attention_mask_t5

    def get_original_text(self, ind):
        text = ""
        try:
            text = self.index_manager.get_attribute(
                ind, "text_zh" if self.enable_CN else "text_en"
            )
        except Exception:
            self.log_fn(f"Warning! Fail get text_zh columns")
        text = str(text).strip()
        return text

    def get_text(self, ind):
        text = self.get_original_text(ind)
        if text == "":
            text = "随机生成一张图片"
        return text

    def __getitem__(self, ind):
        if random.random() < self.uncond_p:
            description = ""
        else:
            description = self.get_text(ind)

        if random.random() < self.uncond_p_t5:
            description_t5 = ""
        else:
            description_t5 = self.get_text(ind)

        if self.use_control:
            if self.is_ipa:
                original_pil_image, cond_tensor, img_for_clip_tensor, kwargs = self.get_image_with_hwxy(ind)
                if random.random() < self.uncond_p_img:
                    img_for_clip_tensor = torch.zeros_like(img_for_clip_tensor)
            else:
                original_pil_image, cond_tensor, kwargs = self.get_image_with_hwxy(ind)
        else:
            if self.is_ipa:
                original_pil_image, img_for_clip_tensor, kwargs = self.get_image_with_hwxy(ind)
                if random.random() < self.uncond_p_img:
                    img_for_clip_tensor = torch.zeros_like(img_for_clip_tensor)
            else:
                original_pil_image, kwargs = self.get_image_with_hwxy(ind)

        text, text_embedding, text_embedding_mask = self.get_text_info_with_encoder(description)
        text_t5, text_embedding_t5, text_embedding_mask_t5 = self.get_text_info_with_encoder_t5(description_t5)

        if self.use_control:
            if self.is_ipa:
                return (
                    original_pil_image,
                    cond_tensor,
                    text_embedding.clone().detach(),
                    text_embedding_mask.clone().detach(),
                    text_embedding_t5.clone().detach(),
                    text_embedding_mask_t5.clone().detach(),
                    img_for_clip_tensor.clone().detach(),
                    {k: v if k == "racem_labels" else torch.tensor(np.array(v)).clone().detach() for k, v in kwargs.items()},
                )
            else:
                return (
                    original_pil_image,
                    cond_tensor,
                    text_embedding.clone().detach(),
                    text_embedding_mask.clone().detach(),
                    text_embedding_t5.clone().detach(),
                    text_embedding_mask_t5.clone().detach(),
                    {k: v if k == "racem_labels" else torch.tensor(np.array(v)).clone().detach() for k, v in kwargs.items()},
                )
        else:
            if self.is_ipa:
                return (
                    original_pil_image,
                    text_embedding.clone().detach(),
                    text_embedding_mask.clone().detach(),
                    text_embedding_t5.clone().detach(),
                    text_embedding_mask_t5.clone().detach(),
                    img_for_clip_tensor.clone().detach(),
                    {k: v if k == "racem_labels" else torch.tensor(np.array(v)).clone().detach() for k, v in kwargs.items()},
                )
            else:
                return (
                    original_pil_image,
                    text_embedding.clone().detach(),
                    text_embedding_mask.clone().detach(),
                    text_embedding_t5.clone().detach(),
                    text_embedding_mask_t5.clone().detach(),
                    {k: v if k == "racem_labels" else torch.tensor(np.array(v)).clone().detach() for k, v in kwargs.items()},
                )

    def __len__(self):
        return len(self.index_manager)
