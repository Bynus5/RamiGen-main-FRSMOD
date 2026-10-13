import os
import json
import torch
from safetensors.torch import load_file as load_safetensors
from ssadit.config import get_args
from ssadit.modules.models import HUNYUAN_DIT_MODELS
from ssadit.inference import _to_tuple

def parse_lora_list(lora_ckpt: str):
    items = [x.strip() for x in lora_ckpt.split(",") if x.strip()]
    results = []
    for item in items:
        if ":" in item:
            path, scale = item.rsplit(":", 1)
            scale = float(scale)
        else:
            path = item
            scale = 1.0
        results.append((path, scale))
    return results

def load_peft_weights(lora_path):
    """
    加载 PEFT 格式的权重 (支持 safetensors 和 bin)
    返回: (state_dict, alpha, rank)
    """
    if not os.path.isdir(lora_path):
        raise ValueError(f"Path is not a PEFT directory: {lora_path}")

    # 1. 读取 config 获取 rank 和 alpha
    config_path = os.path.join(lora_path, "adapter_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    r = config.get("r", 8)
    lora_alpha = config.get("lora_alpha", r) # 如果没存 alpha，默认通常等于 r
    
    print(f"    -> Config detected: r={r}, alpha={lora_alpha}")

    # 2. 加载权重
    safetensors_path = os.path.join(lora_path, "adapter_model.safetensors")
    bin_path = os.path.join(lora_path, "adapter_model.bin")

    if os.path.exists(safetensors_path):
        print(f"    -> Loading safetensors: {safetensors_path}")
        lora_sd = load_safetensors(safetensors_path)
    elif os.path.exists(bin_path):
        print(f"    -> Loading bin: {bin_path}")
        lora_sd = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No weight file found in {lora_path}")
        
    return lora_sd, lora_alpha, r

def merge_lora_weights(base_model, lora_path, user_scale):
    print(f"[INFO] Merging LoRA from: {lora_path} (User Scale = {user_scale})")
    
    # 加载 PEFT 权重
    lora_sd, alpha, rank = load_peft_weights(lora_path)
    base_sd = base_model.state_dict()
    
    # 计算基础缩放因子: (alpha / r) * user_scale
    scaling = (alpha / rank) * user_scale
    print(f"    -> Final Scaling Factor: {scaling:.4f} (alpha={alpha} / rank={rank} * user={user_scale})")

    merged_keys = 0
    
    # 遍历 LoRA 权重
    for key in lora_sd.keys():
        if "lora_A" in key:
            # PEFT key 格式通常是: base_model.model.blocks.0...
            # 我们需要把它映射回 base model 的 key
            key_A = key
            key_B = key.replace("lora_A", "lora_B")
            
            if key_B not in lora_sd:
                continue

            # === Key 映射逻辑 ===
            # 去掉 PEFT 自动加的前缀 "base_model.model."
            base_key_name = key.replace("base_model.model.", "")
            # 去掉 ".lora_A.weight" 还原为线性层名字
            base_key_name = base_key_name.replace(".lora_A.weight", "")
            
            # HunyuanDiT 里的权重通常叫 .weight
            # 比如: blocks.0.attn.qkv.weight
            target_weight_key = base_key_name + ".weight"

            # 兼容性检查：如果 key 不在 base 里，可能不需要 "model." 前缀，或者其他情况
            if target_weight_key not in base_sd:
                # 尝试加上 "module." (如果 base 是 DDP 保存的)
                if "module." + target_weight_key in base_sd:
                    target_weight_key = "module." + target_weight_key
                else:
                    # 再次尝试：有时候 PEFT 保存的 key 很怪，视情况忽略
                    # print(f"[WARN] Skipped {key}, base key {target_weight_key} not found.")
                    continue

            # === 融合计算 ===
            # W_new = W_old + (B @ A) * scaling
            w_a = lora_sd[key_A].float()
            w_b = lora_sd[key_B].float()
            
            # 注意: Linear 层权重是 [out, in]，A是[r, in], B是[out, r]
            # delta = B @ A -> [out, in]
            delta = (w_b @ w_a) * scaling
            
            # 执行加法 (In-place)
            base_sd[target_weight_key].data += delta.to(base_sd[target_weight_key].device).type(base_sd[target_weight_key].dtype)
            merged_keys += 1

    print(f"[INFO] Successfully merged {merged_keys} modules.")

def main():
    args = get_args()

    # ===== 1. Build Base Model =====
    image_size = _to_tuple(args.image_size)
    latent_size = (image_size[0] // 8, image_size[1] // 8)

    print(f"[INFO] Building model: {args.model}")
    model = HUNYUAN_DIT_MODELS[args.model](
        args,
        input_size=latent_size,
        log_fn=print,
    )

    # 加载 Base 权重
    model_path = os.path.join(
        args.model_root, "t2i", "model", f"pytorch_model_{args.load_key}.pt"
    )
    print(f"[INFO] Loading base model from {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")
    if "module" in state_dict:
        state_dict = state_dict["module"]
    model.load_state_dict(state_dict)

    # ===== 2. Multi LoRA Merge =====
    lora_items = parse_lora_list(args.lora_ckpt)
    
    print(f"[INFO] Found {len(lora_items)} LoRA adapters to merge.")

    # 依次融合
    for idx, (lora_path, scale) in enumerate(lora_items):
        if not os.path.isdir(lora_path):
             # 容错：如果用户传的是 checkpoints/0010000.pt，但系统认为是文件
             # 但根据你的 ls，它是一个目录，所以这里应该通过 isdir 检查
             raise ValueError(f"LoRA path must be a PEFT directory: {lora_path}")
        
        print(f"\n--- Processing LoRA {idx+1}/{len(lora_items)} ---")
        merge_lora_weights(model, lora_path, scale)

    # ===== 3. Save =====
    print(f"\n[INFO] Saving merged model...")
    os.makedirs(os.path.dirname(args.output_merge_path), exist_ok=True)
    
    save_dict = model.state_dict()
    torch.save(save_dict, args.output_merge_path)
    
    print(f"[INFO] Done! Saved to {args.output_merge_path}")

if __name__ == "__main__":
    main()
