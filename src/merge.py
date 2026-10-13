import torch
import os
from ssadit.config import get_args
from ssadit.modules.models import HUNYUAN_DIT_MODELS
from ssadit.inference import _to_tuple

args = get_args()

image_size = _to_tuple(args.image_size)
latent_size = (image_size[0] // 8, image_size[1] // 8)

model = HUNYUAN_DIT_MODELS[args.model](
    args,
    input_size=latent_size,
    log_fn=print,
)

if hasattr(args, "base_ckpt") and args.base_ckpt is not None:
    model_path = args.base_ckpt
else:
    model_path = os.path.join(
        args.model_root, "t2i", "model", f"pytorch_model_{args.load_key}.pt"
    )

print(f"Loading model from {model_path}")
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
msg = model.load_state_dict(state_dict, strict=False)
print("missing_keys:", msg.missing_keys)
print("unexpected_keys:", msg.unexpected_keys)

print(f"Loading lora from {args.lora_ckpt}")
model.load_adapter(args.lora_ckpt)
model.merge_and_unload()

torch.save(model.state_dict(), args.output_merge_path)
print(f"Model saved to {args.output_merge_path}")
