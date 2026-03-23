import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def find_images(directory: Path, exts: Tuple[str, ...]) -> List[Path]:
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts])


def build_stem_map(files: List[Path]) -> Dict[str, Path]:
    return {f.stem: f for f in files}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def downsample_and_apply(
    images_dir: Path,
    masks_dir: Path,
    scale: int,
    output_images_dir: Path,
    output_masks_dir: Path | None,
    output_unmasked_dir: Path | None = None,
) -> None:
    image_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    mask_exts: Tuple[str, ...] = image_exts

    ensure_dir(output_images_dir)
    if output_masks_dir is not None:
        ensure_dir(output_masks_dir)

    image_files: List[Path] = find_images(images_dir, image_exts)
    mask_files: List[Path] = find_images(masks_dir, mask_exts)
    mask_map: Dict[str, Path] = build_stem_map(mask_files)

    total = len(image_files)
    processed = 0
    missing_masks: List[str] = []

    for img_path in image_files:
        stem = img_path.stem
        mask_path = mask_map.get(stem)
        if mask_path is None:
            missing_masks.append(stem)
            continue

        with Image.open(img_path) as img_src:
            # Convert to RGB to ensure 3 channels; drop alpha if present
            img_rgb = img_src.convert("RGB")
            width, height = img_rgb.size
            new_w = max(1, width // scale)
            new_h = max(1, height // scale)
            img_ds = img_rgb.resize((new_w, new_h), Image.LANCZOS)

        # Optionally save unmasked/downsampled copy (as PNG)
        if output_unmasked_dir is not None:
            ensure_dir(output_unmasked_dir)
            out_unmasked_path = (output_unmasked_dir / img_path.stem).with_suffix(".png")
            img_ds.save(out_unmasked_path)

        with Image.open(mask_path) as mask_src:
            # Convert mask to single channel
            mask_gray = mask_src.convert("L")
            # Downsample masks with nearest to preserve hard edges
            mask_ds = mask_gray.resize((new_w, new_h), Image.NEAREST)

        # Create RGBA and set alpha from mask
        rgba = img_ds.convert("RGBA")
        rgba.putalpha(mask_ds)

        out_img_path = (output_images_dir / img_path.stem).with_suffix(".png")
        rgba.save(out_img_path)

        if output_masks_dir is not None:
            out_mask_path = (output_masks_dir / mask_path.stem).with_suffix(".png")
            # Save downsampled mask as 8-bit grayscale PNG
            mask_ds.save(out_mask_path)

        processed += 1
        if processed % 50 == 0 or processed == total:
            print(f"Processed {processed}/{total}")

    if missing_masks:
        print(f"Warning: missing {len(missing_masks)} mask(s) for images: e.g., {missing_masks[:5]}")
    print(f"Done. Wrote masked images to: {output_images_dir}")
    if output_masks_dir is not None:
        print(f"Downsampled masks saved to: {output_masks_dir}")


def downsample_images_only(
    images_dir: Path,
    scale: int,
    output_images_dir: Path,
) -> None:
    image_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    ensure_dir(output_images_dir)

    image_files: List[Path] = find_images(images_dir, image_exts)
    total = len(image_files)
    processed = 0

    for img_path in image_files:
        with Image.open(img_path) as img_src:
            img_rgb = img_src.convert("RGB")
            width, height = img_rgb.size
            new_w = max(1, width // scale)
            new_h = max(1, height // scale)
            img_ds = img_rgb.resize((new_w, new_h), Image.LANCZOS)

        out_img_path = (output_images_dir / img_path.stem).with_suffix(".png")
        img_ds.save(out_img_path)

        processed += 1
        if processed % 50 == 0 or processed == total:
            print(f"Processed {processed}/{total}")

    print(f"Done. Wrote downsampled images to: {output_images_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downsample images and masks. Apply masks as alpha channel to PNGs.")
    parser.add_argument("--images_dir", type=Path, required=True, help="Directory containing input images")
    parser.add_argument("--masks_dir", type=Path, default=None, help="Directory containing input masks (required if mode=apply)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["apply", "copy", "both"],
        default="apply",
        help="apply: downsample and set mask as alpha; copy: just downsample images; both: save masked and unmasked",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=8,
        help="Downsample factor (integer). Output size = floor(input/scale)",
    )
    parser.add_argument(
        "--output_images_dir",
        type=Path,
        default=None,
        help="Directory to write masked, downsampled images (default: <images_dir>_ds<scale>_masked)",
    )
    parser.add_argument(
        "--output_masks_dir",
        type=Path,
        default=None,
        help="Optional directory to write downsampled masks (mode=apply only)",
    )
    parser.add_argument(
        "--output_unmasked_dir",
        type=Path,
        default=None,
        help="Optional directory to write downsampled unmasked images (mode=apply/both). Default: <images_dir>_ds<scale>_unmasked for mode=both",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    images_dir: Path = args.images_dir.resolve()
    masks_dir = Path(args.masks_dir).resolve() if args.masks_dir is not None else None
    scale: int = int(args.scale)
    mode: str = str(args.mode)

    if args.output_images_dir is None:
        # Default depends on mode
        default_suffix = "ds{scale}_masked" if mode in ("apply", "both") else f"ds{scale}"
        output_images_dir = images_dir.parent / f"{images_dir.name}_{default_suffix}"
        # Interpolate scale
        output_images_dir = Path(str(output_images_dir).format(scale=scale)).resolve()
    else:
        output_images_dir = Path(args.output_images_dir).resolve()

    output_masks_dir = Path(args.output_masks_dir).resolve() if args.output_masks_dir is not None else None
    output_unmasked_dir = Path(args.output_unmasked_dir).resolve() if args.output_unmasked_dir is not None else None

    if not images_dir.exists() or not images_dir.is_dir():
        raise SystemExit(f"images_dir not found or not a directory: {images_dir}")
    if scale < 1:
        raise SystemExit("scale must be >= 1")

    if mode == "apply":
        if masks_dir is None:
            raise SystemExit("--masks_dir is required when mode=apply")
        if not masks_dir.exists() or not masks_dir.is_dir():
            raise SystemExit(f"masks_dir not found or not a directory: {masks_dir}")
        downsample_and_apply(images_dir, masks_dir, scale, output_images_dir, output_masks_dir, output_unmasked_dir)
    elif mode == "copy":
        downsample_images_only(images_dir, scale, output_images_dir)
    elif mode == "both":
        if masks_dir is None:
            raise SystemExit("--masks_dir is required when mode=both")
        if not masks_dir.exists() or not masks_dir.is_dir():
            raise SystemExit(f"masks_dir not found or not a directory: {masks_dir}")

        # Default unmasked dir if not provided
        if output_unmasked_dir is None:
            output_unmasked_dir = images_dir.parent / f"{images_dir.name}_ds{scale}_unmasked"
        output_unmasked_dir = Path(output_unmasked_dir).resolve()

        downsample_and_apply(
            images_dir,
            masks_dir,
            scale,
            output_images_dir,
            output_masks_dir,
            output_unmasked_dir,
        )
    else:
        raise SystemExit(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()


