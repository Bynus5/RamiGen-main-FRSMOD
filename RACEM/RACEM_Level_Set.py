#!/usr/bin/env python3
"""Offline RACEM extraction: save selected condition maps one image at a time.

HED is in edge_detector/; SAM2/MiDaS are in vendor/; weights are in ../ckpts/racem/.
For online ControlNet training use --racem-mode batch_wise in its trainer.
Online extraction uses the trainer's DataLoader batch, not this command.
"""

import argparse
import json
import sys
from pathlib import Path


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", nargs="+", choices=("all", "edge", "depth", "mask"), default=["all"])
    parser.add_argument("--images", required=True, help="RGB image directory")
    parser.add_argument("--labels", help="YOLO labels; required for mask/bbox/clip")
    parser.add_argument("--output", required=True, help="Output conditions directory")
    parser.add_argument("--region", choices=("full", "bbox", "clip"), default="clip",
                        help="Edge/depth: full image, label-box region, or refined-mask region; "
                             "output size stays unchanged; mask always uses SAM2 + level-set")
    parser.add_argument("--bbox-type", choices=("obb", "hbb"), default="obb")
    parser.add_argument("--device", default=None, help="cpu or cuda:N; default uses available CUDA")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing feature images")
    parser.add_argument("--sam-weights", help="Override bundled SAM2.1-large weights")
    parser.add_argument("--edge-weights", help="Override Paddle HED model_hed.pdparams weights")
    parser.add_argument("--depth-weights", help="Override dpt_beit_large_512.pt")
    args = parser.parse_args(argv)

    import torch
    from RACEM import RACEMExtractor

    if args.labels is None and (args.region != "full" or "mask" in args.features or "all" in args.features):
        parser.error("--labels is required for mask/bbox/clip extraction")
    extractor = RACEMExtractor(
        args.features, label_dir=args.labels, region=args.region, bbox_type=args.bbox_type,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        sam_weights=args.sam_weights, edge_weights=args.edge_weights, depth_weights=args.depth_weights,
    )
    stats = extractor.preprocess(
        args.images, args.output,
        overwrite=args.overwrite,
    )
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
