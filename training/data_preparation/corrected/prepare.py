"""Corrected preparation utilities; historical source remains in sibling folders."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Same alphabetical category ordering as the historical mask notebook.
LABELS = {
    "benign_calcification": 1,
    "benign_mass": 2,
    "malignant_calcification": 3,
    "malignant_mass": 4,
}


def crop_box(image, border=5, threshold=10):
    """Return pixel xywh in the ORIGINAL image, never normalized or rescaled twice."""
    if image.ndim != 2 or min(image.shape) <= 2 * border or border < 0:
        raise ValueError("Expected grayscale image larger than twice the border")
    interior = image[border:-border, border:-border] if border else image
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (interior > threshold).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        raise ValueError("No foreground component found")
    x, y, width, height, _ = stats[1 + stats[1:, cv2.CC_STAT_AREA].argmax()]
    return int(x + border), int(y + border), int(width), int(height)


def ellipse_mask(shape, lesions):
    """Draw each (left, top, width, height, label) independently.

    On overlapping pixels, later CSV lesions win; disjoint earlier lesions keep
    their category. Invalid/out-of-bounds boxes are rejected rather than clipped.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    for left, top, width, height, label in lesions:
        values = np.array([left, top, width, height], dtype=float)
        if not np.isfinite(values).all() or not 1 <= label <= 4:
            raise ValueError("Invalid lesion box or label")
        if (left < 0 or top < 0 or width <= 0 or height <= 0
                or left + width > shape[1] or top + height > shape[0]):
            raise ValueError("Lesion box is empty or outside image bounds")
        region = np.zeros(shape, dtype=np.uint8)
        cv2.ellipse(region, (int(left + width / 2), int(top + height / 2)),
                    (int(width / 2), int(height / 2)), 0, 0, 360, 1, -1)
        mask[region > 0] = label
    return mask


def _read(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes image directory: {relative}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    crop = commands.add_parser("crop-labels", help="Write pixel xywh labels; does not create folds")
    crop.add_argument("--images", type=Path, required=True)
    crop.add_argument("--output", type=Path, required=True)
    crop.add_argument("--border", type=int, default=5)
    masks = commands.add_parser("ddsm-masks", help="Write multi-category ellipse masks")
    masks.add_argument("--images", type=Path, required=True)
    masks.add_argument("--annotations", type=Path, required=True)
    masks.add_argument("--output", type=Path, required=True)
    masks.add_argument("--bbox-order", choices=["yx", "xy"], default="yx",
                       help="yx matches historical cleaned CSV notebook mapping: left=y1, top=x1")
    args = parser.parse_args(argv)
    if args.command == "crop-labels":
        if args.output.exists():
            raise FileExistsError(args.output)
        rows = []
        for path in sorted(args.images.rglob("*.png")):
            box = crop_box(_read(path), args.border)
            rows.append([path.relative_to(args.images).as_posix(), *box])
        if not rows:
            raise ValueError("No PNG images found")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "x", "y", "w", "h"])
            writer.writerows(rows)
    else:
        # Require a fresh directory, so partial completion cannot overwrite old masks.
        if args.output.exists():
            raise FileExistsError(args.output)
        grouped = defaultdict(list)
        with args.annotations.open(newline="") as f:
            for row in csv.DictReader(f):
                label = row["pathology"].lower().split("_")[0] + "_" + row["abnormality"].lower()
                if label not in LABELS:
                    raise ValueError(f"Unknown lesion category: {label}")
                a, b = args.bbox_order
                left, top = float(row[a + "1"]), float(row[b + "1"])
                width, height = float(row[a + "2"]) - left, float(row[b + "2"]) - top
                relative = Path(row["filepath"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Path escapes image directory: {row['filepath']}")
                if relative.suffix.lower() != ".png":
                    raise ValueError("Annotation image paths must end in .png")
                # Path removes redundant './' and separators before grouping, so
                # aliases cannot separately overwrite the same output mask.
                filename = relative.as_posix()
                _inside(args.images, filename)
                grouped[filename].append((left, top, width, height, LABELS[label]))
        if not grouped:
            raise ValueError("No annotation rows found")
        args.output.mkdir(parents=True)
        for filename, lesions in grouped.items():
            image = _read(_inside(args.images, filename))
            mask = ellipse_mask(image.shape, lesions)
            dest = _inside(args.output, filename)
            dest.parent.mkdir(parents=True, exist_ok=True)
            success, encoded = cv2.imencode(dest.suffix, mask)
            if not success:
                raise OSError(f"Could not write mask: {dest}")
            with dest.open("xb") as f:
                f.write(encoded.tobytes())
        with (args.output / "labels.json").open("x") as f:
            f.write(json.dumps(LABELS, indent=2) + "\n")


if __name__ == "__main__":
    main()
