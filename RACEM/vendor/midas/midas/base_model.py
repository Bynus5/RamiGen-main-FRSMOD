import torch


import torch


class BaseModel(torch.nn.Module):
    def load(self, path):
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        # 删除部分 timm / BEiT 版本差异导致的非核心 buffer
        remove_keys = []
        for k in parameters.keys():
            if "relative_position_index" in k:
                remove_keys.append(k)

        for k in remove_keys:
            print(f"Ignore unexpected key: {k}")
            parameters.pop(k)

        missing, unexpected = self.load_state_dict(parameters, strict=False)

        if len(missing) > 0:
            print("Missing keys:")
            for k in missing:
                print("  ", k)

        if len(unexpected) > 0:
            print("Unexpected keys:")
            for k in unexpected:
                print("  ", k)
