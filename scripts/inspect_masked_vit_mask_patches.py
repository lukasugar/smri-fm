#!/usr/bin/env python
"""Inspect how mean mask calculation changes MaskedViT patch token counts.

Default mode avoids running transformer attention and reports the exact number
of patch tokens that would enter the encoder. Use --run-forward on a GPU node if
you also want to verify the actual forward_embedding output shapes.

Each input path must be a .pt file. The script loads the file with torch.load
and uses the first element of the loaded object as the image tensor.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smri_mae.model_mae import MaskedViT, trim_patch_mask  # noqa: E402
from smri_mae.modules import patchify3d  # noqa: E402


class PatchFlowSummary(NamedTuple):
    total_patches: int
    patch_grid: tuple[int, int, int]
    voxels_per_patch: int
    observed_voxels: list[int]
    visible_patches_before_trim: list[int]
    fully_masked_patches_before_trim: list[int]
    tokens_entering_transformer: int
    dropped_visible_patches_by_batch_trim: list[int]


def to_3_tuple(values: int | list[int] | tuple[int, ...], name: str) -> tuple[int, int, int]:
    if isinstance(values, int):
        return (values, values, values)
    values = tuple(int(value) for value in values)
    if len(values) == 1:
        return (values[0], values[0], values[0])
    if len(values) == 3:
        return values
    raise ValueError(f"{name} must contain either 1 or 3 integers, got {values}")


def summarize_patch_flow(
    images: torch.Tensor,
    *,
    patch_size: tuple[int, int, int],
    mask: torch.Tensor | None,
) -> PatchFlowSummary:
    if images.ndim != 5:
        raise ValueError(f"images must have shape [B, C, T, H, W], got {tuple(images.shape)}")

    batch_size, channels, time, height, width = images.shape
    p_t, p_h, p_w = patch_size
    if time % p_t or height % p_h or width % p_w:
        raise ValueError(
            f"image spatial shape {(time, height, width)} must be divisible by patch_size "
            f"{patch_size}"
        )

    patch_grid = (time // p_t, height // p_h, width // p_w)
    total_patches = math.prod(patch_grid)
    voxels_per_patch = channels * math.prod(patch_size)

    if mask is None:
        full_voxels = channels * time * height * width
        return PatchFlowSummary(
            total_patches=total_patches,
            patch_grid=patch_grid,
            voxels_per_patch=voxels_per_patch,
            observed_voxels=[full_voxels] * batch_size,
            visible_patches_before_trim=[total_patches] * batch_size,
            fully_masked_patches_before_trim=[0] * batch_size,
            tokens_entering_transformer=total_patches,
            dropped_visible_patches_by_batch_trim=[0] * batch_size,
        )

    mask = mask.to(dtype=torch.bool).expand_as(images)
    mask_patches = patchify3d(mask, patch_size)
    patch_num_obs = mask_patches.sum(dim=-1)
    patch_mask = patch_num_obs > 0
    visible_counts = patch_mask.sum(dim=1)
    trimmed_patch_mask, mask_ids = trim_patch_mask(patch_mask.clone(), mask_ratio=0.0)
    trimmed_counts = trimmed_patch_mask.sum(dim=1)

    if mask_ids.ndim != 2:
        raise RuntimeError(f"expected trim_patch_mask ids to be 2D, got {tuple(mask_ids.shape)}")

    return PatchFlowSummary(
        total_patches=total_patches,
        patch_grid=patch_grid,
        voxels_per_patch=voxels_per_patch,
        observed_voxels=mask.flatten(1).sum(dim=1).cpu().tolist(),
        visible_patches_before_trim=visible_counts.cpu().tolist(),
        fully_masked_patches_before_trim=(total_patches - visible_counts).cpu().tolist(),
        tokens_entering_transformer=int(mask_ids.shape[1]),
        dropped_visible_patches_by_batch_trim=(visible_counts - trimmed_counts).cpu().tolist(),
    )


def calculate_mean_mask(images: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, images.ndim))
    return images > images.mean(dim=dims, keepdim=True)


def load_pt_image(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=False)
    try:
        image = data[0]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"{path} must load to an indexable object with image at index 0") from error

    image = torch.as_tensor(image, dtype=torch.float32)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(
            f"{path}[0] must be a 3D [T,H,W] or 4D [C,T,H,W] image tensor, "
            f"got shape {tuple(image.shape)}"
        )
    return image


def pad_center_crop(tensor: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    spatial_shape = tensor.shape[-3:]
    pad_args: list[int] = []
    for current, target in zip(reversed(spatial_shape), reversed(size), strict=True):
        pad_total = max(0, target - current)
        pad_before = (pad_total + 1) // 2
        pad_after = pad_total - pad_before
        pad_args.extend([pad_before, pad_after])

    padded = F.pad(tensor, pad_args, mode="constant", value=0.0)
    crop_slices: list[slice] = []
    for current, target in zip(padded.shape[-3:], size, strict=True):
        start = max(0, (current - target) // 2)
        crop_slices.append(slice(start, start + target))
    return padded[(Ellipsis, *crop_slices)]


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def model_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("model") or {})


def make_batch(
    paths: list[Path],
    *,
    img_size: tuple[int, int, int],
    pad_or_crop: bool,
) -> tuple[torch.Tensor, list[tuple[int, ...]]]:
    images: list[torch.Tensor] = []
    original_shapes: list[tuple[int, ...]] = []
    for path in paths:
        image = load_pt_image(path)
        original_shapes.append(tuple(image.shape))
        if tuple(image.shape[-3:]) != img_size:
            if not pad_or_crop:
                raise ValueError(
                    f"{path} has spatial shape {tuple(image.shape[-3:])}, expected {img_size}. "
                    "Pass --pad-center-crop to adapt images before batching."
                )
            image = pad_center_crop(image, img_size)
        images.append(image)
    return torch.stack(images, dim=0), original_shapes


def print_summary(label: str, summary: PatchFlowSummary) -> None:
    print(f"\n{label}")
    print(f"  patch_grid: {summary.patch_grid}")
    print(f"  total_patches_per_image: {summary.total_patches}")
    print(f"  voxels_per_patch: {summary.voxels_per_patch}")
    print(f"  observed_voxels_per_image: {summary.observed_voxels}")
    print(f"  visible_patches_before_batch_trim: {summary.visible_patches_before_trim}")
    print(f"  fully_masked_patches_before_batch_trim: {summary.fully_masked_patches_before_trim}")
    print(f"  tokens_entering_transformer_per_image: {summary.tokens_entering_transformer}")
    print(f"  dropped_visible_patches_by_batch_trim: {summary.dropped_visible_patches_by_batch_trim}")


def run_forward(
    images: torch.Tensor,
    mean_mask: torch.Tensor,
    *,
    img_size: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    in_chans: int,
    model_kwargs: dict[str, Any],
    checkpoint_path: Path | None,
    device: str,
) -> None:
    from evaluation.backbones import load_smri_mae_checkpoint

    resolved_device = device
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MaskedViT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        **model_kwargs,
    ).to(resolved_device)
    if checkpoint_path is not None:
        load_smri_mae_checkpoint(model, checkpoint_path)
    model.eval()

    images = images.to(resolved_device)
    mean_mask = mean_mask.to(resolved_device)
    with torch.inference_mode():
        no_mask_cls, no_mask_reg, no_mask_patch = model.forward_embedding(images, mask=None)
        mean_cls, mean_reg, mean_patch = model.forward_embedding(images, mask=mean_mask)

    def shape_or_none(tensor: torch.Tensor | None) -> tuple[int, ...] | None:
        return None if tensor is None else tuple(tensor.shape)

    print("\nforward_embedding output shapes")
    print(
        "  no_mask: "
        f"cls={shape_or_none(no_mask_cls)}, reg={shape_or_none(no_mask_reg)}, "
        f"patch={shape_or_none(no_mask_patch)}"
    )
    print(
        "  mean_mask: "
        f"cls={shape_or_none(mean_cls)}, reg={shape_or_none(mean_reg)}, "
        f"patch={shape_or_none(mean_patch)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs=4, type=Path, help="Four input .pt files")
    parser.add_argument("--config", type=Path, help="Optional evaluation YAML config")
    parser.add_argument("--img-size", nargs="+", type=int, help="Model image size, one int or T H W")
    parser.add_argument("--patch-size", nargs="+", type=int, help="Patch size, one int or T H W")
    parser.add_argument("--in-chans", type=int, help="Input channels")
    parser.add_argument(
        "--pad-center-crop",
        action="store_true",
        help="Pad/crop each input to --img-size before batching",
    )
    parser.add_argument(
        "--run-forward",
        action="store_true",
        help="Also run MaskedViT.forward_embedding for no-mask and mean-mask batches",
    )
    parser.add_argument("--checkpoint-path", type=Path, help="Optional MAE checkpoint for --run-forward")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--json-out", type=Path, help="Optional path for JSON summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    cfg_model = model_config(config)

    img_size = to_3_tuple(args.img_size or cfg_model.get("img_size", [208, 240, 208]), "img_size")
    patch_size = to_3_tuple(args.patch_size or cfg_model.get("patch_size", 8), "patch_size")
    in_chans = int(args.in_chans or cfg_model.get("in_chans", 1))
    checkpoint_path = args.checkpoint_path or cfg_model.get("checkpoint_path")
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    run_model_kwargs = dict(cfg_model.get("model_kwargs") or {})

    images, original_shapes = make_batch(
        args.images,
        img_size=img_size,
        pad_or_crop=args.pad_center_crop,
    )
    if images.shape[1] != in_chans:
        raise ValueError(f"batch has {images.shape[1]} channel(s), but in_chans={in_chans}")

    mean_mask = calculate_mean_mask(images)
    no_mask_summary = summarize_patch_flow(images, patch_size=patch_size, mask=None)
    mean_mask_summary = summarize_patch_flow(images, patch_size=patch_size, mask=mean_mask)
    image_means = images.mean(dim=(1, 2, 3, 4)).cpu().tolist()

    print(f"input_paths: {[str(path) for path in args.images]}")
    print(f"original_shapes_[C,T,H,W]: {original_shapes}")
    print(f"batched_shape_[B,C,T,H,W]: {tuple(images.shape)}")
    print(f"img_size: {img_size}")
    print(f"patch_size: {patch_size}")
    print(f"mean_mask_rule: mask = image > image.mean(dim=(C,T,H,W), keepdim=True)")
    print(f"image_means: {image_means}")
    print_summary("no mask", no_mask_summary)
    print_summary("mean mask", mean_mask_summary)
    print(
        "\ninterpretation: masked voxels are zeroed before patch embedding, but a patch is kept "
        "if any voxel in that patch is visible. Fully masked patches are removed. In a batch, "
        "all samples are trimmed to the smallest visible-patch count."
    )

    if args.json_out is not None:
        payload = {
            "input_paths": [str(path) for path in args.images],
            "original_shapes": original_shapes,
            "batched_shape": tuple(images.shape),
            "img_size": img_size,
            "patch_size": patch_size,
            "image_means": image_means,
            "no_mask": no_mask_summary._asdict(),
            "mean_mask": mean_mask_summary._asdict(),
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.run_forward:
        run_forward(
            images,
            mean_mask,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            model_kwargs=run_model_kwargs,
            checkpoint_path=checkpoint_path,
            device=args.device,
        )


if __name__ == "__main__":
    main()
