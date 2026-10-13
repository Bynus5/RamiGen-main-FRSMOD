import torch
from diffusers import HunyuanDiTPipeline

num_layers = 40
def load_hunyuan_dit_lora(transformer_state_dict, lora_state_dict, lora_scale):
    for i in range(num_layers):
        Wqkv = torch.matmul(lora_state_dict[f"blocks.{i}.attn1.Wqkv.lora_B.weight"], lora_state_dict[f"blocks.{i}.attn1.Wqkv.lora_A.weight"]) 
        q, k, v = torch.chunk(Wqkv, 3, dim=0)
        transformer_state_dict[f"blocks.{i}.attn1.to_q.weight"] += lora_scale * q
        transformer_state_dict[f"blocks.{i}.attn1.to_k.weight"] += lora_scale * k
        transformer_state_dict[f"blocks.{i}.attn1.to_v.weight"] += lora_scale * v

        out_proj = torch.matmul(lora_state_dict[f"blocks.{i}.attn1.out_proj.lora_B.weight"], lora_state_dict[f"blocks.{i}.attn1.out_proj.lora_A.weight"]) 
        transformer_state_dict[f"blocks.{i}.attn1.to_out.0.weight"] += lora_scale * out_proj

        q_proj = torch.matmul(lora_state_dict[f"blocks.{i}.attn2.q_proj.lora_B.weight"], lora_state_dict[f"blocks.{i}.attn2.q_proj.lora_A.weight"])
        transformer_state_dict[f"blocks.{i}.attn2.to_q.weight"] += lora_scale * q_proj

        kv_proj = torch.matmul(lora_state_dict[f"blocks.{i}.attn2.kv_proj.lora_B.weight"], lora_state_dict[f"blocks.{i}.attn2.kv_proj.lora_A.weight"])
        k, v = torch.chunk(kv_proj, 2, dim=0)
        transformer_state_dict[f"blocks.{i}.attn2.to_k.weight"] += lora_scale * k
        transformer_state_dict[f"blocks.{i}.attn2.to_v.weight"] += lora_scale * v

        out_proj = torch.matmul(lora_state_dict[f"blocks.{i}.attn2.out_proj.lora_B.weight"], lora_state_dict[f"blocks.{i}.attn2.out_proj.lora_A.weight"]) 
        transformer_state_dict[f"blocks.{i}.attn2.to_out.0.weight"] += lora_scale * out_proj
    
    q_proj = torch.matmul(lora_state_dict["pooler.q_proj.lora_B.weight"], lora_state_dict["pooler.q_proj.lora_A.weight"])
    transformer_state_dict["time_extra_emb.pooler.q_proj.weight"] += lora_scale * q_proj
    
    return transformer_state_dict

pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", torch_dtype=torch.float16)
pipe.to("cuda")

from safetensors import safe_open

lora_state_dict = {}
with safe_open("./ckpts/t2i/lora/jade/adapter_model.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        lora_state_dict[k[17:]] = f.get_tensor(k) # remove 'basemodel.model'

transformer_state_dict = pipe.transformer.state_dict()
transformer_state_dict = load_hunyuan_dit_lora(transformer_state_dict, lora_state_dict, lora_scale=1.0)
pipe.transformer.load_state_dict(transformer_state_dict)

# prompt = "玉石绘画风格，一只猫在追蝴蝶"
# image = pipe(
#     prompt, 
#     num_inference_steps=100,
#     guidance_scale=6.0, 
# ).images[0]
# image.save('img.png')