"""Command-line interface for one breast."""

import argparse
import json

from . import MammoscreenPipeline, load_image


def main():
    parser = argparse.ArgumentParser(description="Crop and analyze one breast's CC/MLO images")
    parser.add_argument("--cc", help="CC grayscale image or .dcm file")
    parser.add_argument("--mlo", help="MLO grayscale image or .dcm file")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--crop-source", default="ianpan/mammo-crop")
    parser.add_argument("--screen-source", default="ianpan/mammoscreen")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if not (args.cc or args.mlo):
        parser.error("provide --cc and/or --mlo for a single breast")
    sample = {k: load_image(v) for k, v in {"cc": args.cc, "mlo": args.mlo}.items() if v}
    pipeline = MammoscreenPipeline.from_pretrained(
        args.device, args.crop_source, args.screen_source, local_files_only=args.local_files_only
    )
    result = pipeline(sample)
    print(json.dumps({k: v.tolist() for k, v in result.items()}))
