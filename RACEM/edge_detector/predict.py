# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import paddle
import tifffile as tiff
import paddle.nn.functional as F

if __package__:
    from .hed_model import HED
    from .utils import load_pretrained_model
else:
    from hed_model import HED
    from utils import load_pretrained_model


# ============================================================
# Paddle 设备信息
# ============================================================

if __name__ == "__main__":
    print("paddle version:", paddle.__version__)
    print("compiled with cuda:", paddle.is_compiled_with_cuda())
    paddle.set_device("gpu:0" if paddle.is_compiled_with_cuda() else "cpu")
    print("current device:", paddle.get_device())


# ============================================================
# 基础配置
# ============================================================

SUPPORTED_IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff"
}

TIFF_EXTS = {
    ".tif", ".tiff"
}

# HED 多尺度推理尺度
HED_SCALES = [0.5, 1.0, 1.5]

# 是否跳过已经生成过的结果
# 建议 True，方便断点续跑
SKIP_EXISTING = True

# TIFF 前三通道解释方式：
# "RGB"：tifffile 读取的多通道 TIFF 默认按 RGB 转 BGR
# "BGR"：如果你的 TIFF 本身就是 BGR 顺序，可以改成 BGR
TIFF_CHANNEL_ORDER = "RGB"


# ============================================================
# 图像读取与 TIFF 转换工具
# ============================================================

def normalize_to_uint8(img, per_channel=False):
    """
    将 uint16 / int16 / float32 / float64 等图像归一化到 uint8。

    参数:
        img: 输入图像
        per_channel:
            False：整体拉伸
            True ：每个通道单独拉伸，适合 RGB / 多通道图像
    """
    if img is None:
        return None

    img = np.asarray(img)

    if img.dtype == np.uint8:
        return img

    img = img.astype(np.float32)

    if per_channel and img.ndim == 3:
        out = np.zeros_like(img, dtype=np.uint8)

        for c in range(img.shape[2]):
            channel = img[:, :, c]
            valid = np.isfinite(channel)

            if not np.any(valid):
                out[:, :, c] = 0
                continue

            vmin = np.percentile(channel[valid], 1)
            vmax = np.percentile(channel[valid], 99)

            if vmax <= vmin:
                vmin = float(np.min(channel[valid]))
                vmax = float(np.max(channel[valid]))

            if vmax <= vmin:
                out[:, :, c] = 0
                continue

            channel = np.clip(channel, vmin, vmax)
            channel = (channel - vmin) / (vmax - vmin + 1e-6)
            channel = np.nan_to_num(channel, nan=0.0, posinf=1.0, neginf=0.0)
            out[:, :, c] = (channel * 255.0).astype(np.uint8)

        return out

    valid = np.isfinite(img)

    if not np.any(valid):
        return np.zeros_like(img, dtype=np.uint8)

    vmin = np.percentile(img[valid], 1)
    vmax = np.percentile(img[valid], 99)

    if vmax <= vmin:
        vmin = float(np.min(img[valid]))
        vmax = float(np.max(img[valid]))

    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.uint8)

    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin + 1e-6)
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    img = img * 255.0

    return img.astype(np.uint8)


def array_to_bgr_uint8(img, source="tifffile"):
    """
    将 tifffile 或 cv2 读取出来的图像统一转换为 HED 需要的 BGR uint8 三通道。

    支持：
        1. H x W
        2. H x W x C
        3. C x H x W
        4. 多页 TIFF，默认取第一页

    参数:
        source:
            "tifffile"：通常多通道按 RGB 解释
            "cv2"     ：OpenCV 读出的 3 通道默认已经是 BGR
    """
    if img is None:
        return None

    img = np.asarray(img)

    # 多页 / 多维 TIFF，例如 N x H x W 或 N x H x W x C
    # 默认取第一页
    if img.ndim == 4:
        print(f"检测到多页/多维 TIFF，shape={img.shape}，默认使用第 1 页")
        img = img[0]

    # C x H x W 转 H x W x C
    # 常见遥感多波段可能是 bands x H x W
    if img.ndim == 3:
        # 如果第一维像通道数，后两维像高宽，则转置
        if img.shape[0] <= 16 and img.shape[1] > 16 and img.shape[2] > 16:
            if img.shape[-1] not in [1, 3, 4]:
                img = np.transpose(img, (1, 2, 0))

    # 单通道 H x W
    if img.ndim == 2:
        img_u8 = normalize_to_uint8(img, per_channel=False)
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        return img_bgr

    # 多通道 H x W x C
    if img.ndim == 3:
        c = img.shape[2]

        if c == 1:
            img_u8 = normalize_to_uint8(img[:, :, 0], per_channel=False)
            img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
            return img_bgr

        if c == 2:
            # 两通道不常见，默认取第一个通道转灰度
            img_u8 = normalize_to_uint8(img[:, :, 0], per_channel=False)
            img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
            return img_bgr

        if c >= 3:
            img_3c = img[:, :, :3]
            img_3c = normalize_to_uint8(img_3c, per_channel=True)

            if source == "cv2":
                # cv2.imread 读取出来默认是 BGR
                img_bgr = img_3c
                return img_bgr

            # tifffile 读取的多通道一般按 RGB 理解
            if TIFF_CHANNEL_ORDER.upper() == "RGB":
                img_bgr = cv2.cvtColor(img_3c, cv2.COLOR_RGB2BGR)
                return img_bgr

            elif TIFF_CHANNEL_ORDER.upper() == "BGR":
                img_bgr = img_3c
                return img_bgr

            else:
                raise ValueError("TIFF_CHANNEL_ORDER 只支持 'RGB' 或 'BGR'")

    raise ValueError(f"不支持的图像维度: shape={img.shape}")


def read_tif_with_fallback(img_path):
    """
    TIFF 读取逻辑：
        1. 优先使用 tifffile
        2. 若 tifffile 因 LZW/JPEG/Deflate 压缩失败，尝试 cv2.imread 兜底
        3. 两者都失败时，提示安装 imagecodecs
    """
    # 1. 优先 tifffile
    try:
        img = tiff.imread(img_path)
        img_bgr = array_to_bgr_uint8(img, source="tifffile")
        return img_bgr

    except Exception as e:
        print(f"tifffile 读取失败: {img_path}")
        print(f"错误信息: {e}")

        # 2. OpenCV 兜底读取
        print("尝试使用 OpenCV 兜底读取 TIFF...")

        img_cv = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img_cv is not None:
            try:
                img_bgr = array_to_bgr_uint8(img_cv, source="cv2")
                print("OpenCV 兜底读取成功。")
                return img_bgr
            except Exception as e2:
                print(f"OpenCV 读取后转换失败: {img_path}")
                print(f"错误信息: {e2}")

        # 3. 两种方式都失败
        print("\n该 TIFF 很可能是 LZW/JPEG/Deflate 等压缩格式。")
        print("请在当前 conda 环境中安装 imagecodecs：")
        print(r"D:\miniconda3\envs\landslide\python.exe -m pip install imagecodecs -i https://pypi.tuna.tsinghua.edu.cn/simple")
        print("安装完成后重新运行本脚本即可。")

        return None


def read_image_any_format(img_path):
    """
    统一读取 jpg/png/bmp/tif/tiff。
    输出一定是：
        uint8, H x W x 3, BGR
    """
    ext = os.path.splitext(img_path)[1].lower()

    if ext in TIFF_EXTS:
        return read_tif_with_fallback(img_path)

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"图像读取失败: {img_path}")
        return None

    try:
        img_bgr = array_to_bgr_uint8(img, source="cv2")
        return img_bgr
    except Exception as e:
        print(f"普通图像转换失败: {img_path}")
        print(f"错误信息: {e}")
        return None


def get_all_image_paths(img_dir):
    """
    获取所有支持格式的图像路径，支持大小写后缀。
    """
    img_paths = []

    for filename in os.listdir(img_dir):
        path = os.path.join(img_dir, filename)

        if not os.path.isfile(path):
            continue

        ext = os.path.splitext(filename)[1].lower()

        if ext in SUPPORTED_IMAGE_EXTS:
            img_paths.append(path)

    img_paths = sorted(img_paths)

    return img_paths


# ============================================================
# 标签与几何工具
# ============================================================

def tensor_flip(x, flip):
    """
    同时兼容 numpy 和 paddle tensor 的翻转。
    当前 flips 默认是 [[False, False]]，一般不会实际触发。
    """
    if flip[0]:
        x = x[:, :, :, ::-1]
    if flip[1]:
        x = x[:, :, ::-1, :]
    return x


def yolo_to_rbox(yolo_line, W, H, bbox_type="obb"):
    """
    YOLO-OBB:
        class x1 y1 x2 y2 x3 y3 x4 y4

    YOLO-HBB:
        class cx cy w h

    返回:
        class_id, cv2.minAreaRect
    """
    parts = yolo_line.strip().split()

    if len(parts) < 5:
        return None, None

    try:
        class_id = int(float(parts[0]))
    except ValueError:
        return None, None

    if bbox_type.lower() == "obb":
        if len(parts) < 9:
            return None, None

        try:
            coords = np.array(
                [float(p) for p in parts[1:9]],
                dtype=np.float32
            ).reshape(4, 2)
        except ValueError:
            return None, None

        coords *= np.array([W, H], dtype=np.float32)
        rect = cv2.minAreaRect(coords.astype(np.float32))

        return class_id, rect

    elif bbox_type.lower() == "hbb":
        try:
            cx, cy, w, h = map(float, parts[1:5])
        except ValueError:
            return None, None

        cx *= W
        cy *= H
        w *= W
        h *= H

        rect = ((cx, cy), (w, h), 0.0)

        return class_id, rect

    else:
        raise ValueError("bbox_type 只支持 'obb' 或 'hbb'")


def rbox_to_mask(rbox, H, W):
    """
    旋转框转二值 mask。
    """
    mask = np.zeros((H, W), dtype=np.uint8)

    box_pts = cv2.boxPoints(rbox).astype(np.int32)

    box_pts[:, 0] = np.clip(box_pts[:, 0], 0, W - 1)
    box_pts[:, 1] = np.clip(box_pts[:, 1], 0, H - 1)

    cv2.fillPoly(mask, [box_pts], 1)

    return mask


# ============================================================
# HED 推理
# ============================================================

def hed_infer(model, img_bgr):
    """
    HED 边缘推理。

    输入:
        img_bgr: uint8, H x W x 3, BGR

    输出:
        edge: uint8, H x W
    """
    if img_bgr is None:
        return None

    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(
            f"HED 输入必须是 H x W x 3 BGR 图像，当前 shape={img_bgr.shape}"
        )

    h, w, _ = img_bgr.shape
    edge = np.zeros((h, w), np.float32)

    flips = [[False, False]]
    count = 0

    for s in HED_SCALES:
        for flip in flips:
            h1, w1 = int(s * h), int(s * w)

            if h1 <= 0 or w1 <= 0:
                continue

            img1 = cv2.resize(
                img_bgr,
                (w1, h1),
                interpolation=cv2.INTER_CUBIC
            )

            img1 = img1.astype(np.float32)
            img1 = img1 - np.array(
                (104.00699, 116.66877, 122.67891),
                dtype=np.float32
            )

            img1 = np.transpose(img1, [2, 0, 1])[None].astype(np.float32)
            img1 = tensor_flip(img1, flip)

            img1 = paddle.to_tensor(img1)

            with paddle.no_grad():
                e = model(img1)[0]
                e = F.sigmoid(e)

            e = tensor_flip(e, flip)
            e = e.numpy()[0, 0]

            e = cv2.resize(
                e,
                (w, h),
                interpolation=cv2.INTER_CUBIC
            )

            edge += e
            count += 1

    if count == 0:
        return np.zeros((h, w), dtype=np.uint8)

    edge /= count

    max_val = edge.max()
    if max_val <= 1e-6:
        return np.zeros((h, w), dtype=np.uint8)

    edge = edge / (max_val + 1e-6)

    return (edge * 255).astype(np.uint8)


# ============================================================
# 主流程
# ============================================================

def process_dataset(
        img_dir,
        label_dir,
        save_dir,
        pretrained_model,
        bbox_type="obb"
):
    os.makedirs(save_dir, exist_ok=True)

    save_full = os.path.join(save_dir, "edge_full")
    save_bbox = os.path.join(save_dir, "edge_bbox")

    os.makedirs(save_full, exist_ok=True)
    os.makedirs(save_bbox, exist_ok=True)

    model = HED()
    load_pretrained_model(model, pretrained_model)
    model.eval()

    img_paths = get_all_image_paths(img_dir)

    print(f"📸 共发现 {len(img_paths)} 张图像")

    if len(img_paths) == 0:
        print(f"未在目录中找到图像: {img_dir}")
        return

    success_count = 0
    skip_count = 0
    fail_count = 0
    no_label_count = 0

    for idx, img_path in enumerate(img_paths, start=1):
        name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, name + ".txt")

        full_save_path = os.path.join(save_full, name + ".png")
        bbox_save_path = os.path.join(save_bbox, name + ".png")

        print(f"\n[{idx}/{len(img_paths)}] 正在处理: {os.path.basename(img_path)}")

        # 断点续跑：两个结果都存在则跳过
        if SKIP_EXISTING and os.path.exists(full_save_path) and os.path.exists(bbox_save_path):
            print(f"已存在，跳过: {name}")
            skip_count += 1
            continue

        img = read_image_any_format(img_path)

        if img is None:
            print(f"跳过，图像读取失败: {img_path}")
            fail_count += 1
            continue

        H, W, _ = img.shape

        # === 1. 全图 Edge ===
        try:
            edge_full = hed_infer(model, img)
        except Exception as e:
            print(f"HED 推理失败: {img_path}")
            print(f"错误信息: {e}")
            fail_count += 1
            continue

        if edge_full is None:
            print(f"跳过，HED 推理结果为空: {img_path}")
            fail_count += 1
            continue

        ok_full = cv2.imwrite(full_save_path, edge_full)

        if not ok_full:
            print(f"edge_full 保存失败: {full_save_path}")
            fail_count += 1
            continue

        # === 2. bbox 内 Edge ===
        edge_bbox = np.zeros((H, W), dtype=np.uint8)

        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()

                if not line:
                    continue

                _, rbox = yolo_to_rbox(line, W, H, bbox_type)

                if rbox is None:
                    continue

                mask = rbox_to_mask(rbox, H, W)
                edge_bbox[mask > 0] = edge_full[mask > 0]

        else:
            print(f"未找到标签文件，仅保存全图边缘: {label_path}")
            no_label_count += 1

        ok_bbox = cv2.imwrite(bbox_save_path, edge_bbox)

        if not ok_bbox:
            print(f"edge_bbox 保存失败: {bbox_save_path}")
            fail_count += 1
            continue

        success_count += 1

        print(f"✅ {name} 处理完成")
        print(f"   edge_full: {full_save_path}")
        print(f"   edge_bbox: {bbox_save_path}")

    print("\n🎉 全部完成！")
    print("=" * 60)
    print(f"总图像数: {len(img_paths)}")
    print(f"成功处理: {success_count}")
    print(f"已存在跳过: {skip_count}")
    print(f"读取/推理/保存失败: {fail_count}")
    print(f"无标签文件: {no_label_count}")
    print("=" * 60)


# ============================================================
# 参数区
# ============================================================

if __name__ == "__main__":

    # === HED 模型 ===
    PRETRAINED_MODEL = r"D:\Desktop\research\RACEM\edge_detector\model_hed.pdparams"

    # === 数据路径 ===
    IMG_DIR = r"D:\Desktop\research\military_sence_detection\images"
    LABEL_DIR = r"D:\Desktop\research\military_sence_detection\labels_yolo_obb"
    SAVE_DIR = r"D:\Desktop\research\military_sence_detection\conditions\edge"

    # === 标签类型 ===
    BBOX_TYPE = "obb"   # "obb" 或 "hbb"

    process_dataset(
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        save_dir=SAVE_DIR,
        pretrained_model=PRETRAINED_MODEL,
        bbox_type=BBOX_TYPE
    )
