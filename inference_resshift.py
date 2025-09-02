#!/usr/bin/env python
# -*- coding:utf-8 -*-
# MRI Super-Resolution Inference using ResShiftSampler
# - Robust .nii/.nii.gz handling
# - Optional PNG single-image path (e.g., 64x64)
# - Percentile / z-score / min-max normalisation options (MRI)
# - Single-slice or whole-volume inference
# - NIfTI output with corrected voxel sizes for in-plane SR
# - Checkpoint overrides for model and autoencoder
# - Uses --bs for tiled inference throughput

import argparse
from pathlib import Path
from typing import Tuple, Optional, List

import nibabel as nib
import numpy as np
import torch
from PIL import Image

from omegaconf import OmegaConf
from sampler import ResShiftSampler
from basicsr.utils.download_util import load_file_from_url

# Only using v3 training version by default
_STEP = {"v3": 4}
_LINK = {
    "v3": "https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s4_v3.pth",
    "vqgan": "https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth",
}

# ------------------------------
# Helpers
# ------------------------------
def is_nii_like(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")


def is_png_like(p: Path) -> bool:
    return p.suffix.lower() == ".png"


def safe_stem(p: Path) -> str:
    n = p.name
    if n.lower().endswith(".nii.gz"):
        return n[:-7]
    if n.lower().endswith(".nii"):
        return n[:-4]
    return p.stem


def norm_img(arr: np.ndarray, mode: str = "percentile", p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """Normalise a 2D slice to [0,1] with robust options."""
    arr = np.asarray(arr, dtype=np.float32)
    if mode == "percentile":
        lo, hi = np.percentile(arr, (p_low, p_high))
        if hi <= lo:
            lo, hi = float(arr.min()), float(arr.max())
        arr = np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    elif mode == "zscore":
        mu, sigma = float(arr.mean()), float(arr.std())
        if sigma < 1e-8:
            sigma = 1.0
        arr = (arr - mu) / (sigma + 1e-8)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    else:  # "minmax"
        mn, mx = float(arr.min()), float(arr.max())
        if mx <= mn:
            return np.zeros_like(arr, dtype=np.float32)
        arr = (arr - mn) / (mx - mn + 1e-8)
    return arr.astype(np.float32)


def extract_slice(vol: np.ndarray, axis: int, idx: Optional[int]) -> Tuple[np.ndarray, int]:
    """Return a 2D slice and used index along the given axis."""
    assert vol.ndim == 3, f"Expected 3D volume, got {vol.shape}"
    if idx is None:
        idx = vol.shape[axis] // 2
    slicer = [slice(None)] * 3
    slicer[axis] = idx
    sl = vol[tuple(slicer)]
    # Ensure 2D (H,W) view; for axial (axis=2) this is typically already (H,W)
    if axis == 0:
        sl = np.ascontiguousarray(sl.T)
    elif axis == 1:
        sl = np.ascontiguousarray(sl)  # already (H,W) under common conventions
    else:
        sl = np.ascontiguousarray(sl)  # axis==2: axial
    return sl, idx


def move_axis_last(vol: np.ndarray, axis: int) -> np.ndarray:
    return np.moveaxis(vol, axis, -1)


def move_axis_back(vol_moved: np.ndarray, axis: int) -> np.ndarray:
    return np.moveaxis(vol_moved, -1, axis)


def new_zooms_after_inplane_sr(zooms: Tuple[float, float, float], scale: int, axis: int) -> Tuple[float, float, float]:
    """For in-plane SR along the two axes orthogonal to 'axis', voxel sizes shrink by factor 'scale'."""
    z = list(zooms[:3])
    if axis == 2:      # axial -> scale X and Y
        z[0] /= scale
        z[1] /= scale
    elif axis == 1:    # coronal -> scale X and Z
        z[0] /= scale
        z[2] /= scale
    elif axis == 0:    # sagittal -> scale Y and Z
        z[1] /= scale
        z[2] /= scale
    return tuple(float(v) for v in z)


def to_tensor_1ch(img2d: np.ndarray) -> torch.Tensor:
    """(H,W) -> [1,1,H,W] float32"""
    return torch.from_numpy(img2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)


def read_png_as_float_grayscale(path: Path) -> np.ndarray:
    """Read a PNG and return float32 in [0,1] grayscale."""
    with Image.open(path) as im:
        im = im.convert("L")
        arr = np.array(im, dtype=np.float32) / 255.0
    return arr


# ------------------------------
# Core SR runner
# ------------------------------
def run_sr_on_slice(
    model: ResShiftSampler,
    img2d_norm: np.ndarray,
    save_dir: Path,
    file_name: str,
    force_rgb: bool = False,
) -> np.ndarray:
    """
    Run SR on a single 2D slice/PNG.
    Returns: (Hs, Ws) float32 in [0,1]
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    tensor = to_tensor_1ch(img2d_norm).float()

    # If the checkpoint expects 3-channel input, optionally repeat the single channel
    if force_rgb and tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)

    with torch.inference_mode():
        out = model.inference_tensor(
            image_tensor=tensor,
            save_dir=save_dir,
            file_name=file_name,
            noise_repeat=False,
        )

    # Try to interpret sampler's return
    if out is not None and isinstance(out, torch.Tensor):
        out = out.detach().cpu().float()
        if out.ndim == 4:
            out = out[0, 0 if out.shape[1] == 1 else slice(None)]
        if out.ndim == 3:
            # If RGB returned, convert to luminance-like average for consistent grayscale path
            out = out.mean(0)
        out_np = out.numpy()
    else:
        # Fallback: read the saved PNG (grayscale)
        with Image.open(save_dir / file_name) as im:
            im = im.convert("L")
            out_np = np.array(im, dtype=np.float32) / 255.0

    return np.clip(out_np, 0.0, 1.0).astype(np.float32)


# ------------------------------
# Configs & model
# ------------------------------
def get_configs(args) -> OmegaConf:
    ckpt_dir = Path("./weights")
    ckpt_dir.mkdir(exist_ok=True)

    config_path = "./configs/realsr_swinunet_realesrgan256_journal.yaml"
    configs = OmegaConf.load(config_path)

    # Resolve checkpoints: use overrides if provided, else download defaults
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = ckpt_dir / f"resshift_realsrx4_s{_STEP[args.version]}_{args.version}.pth"
        if not ckpt_path.exists():
            load_file_from_url(_LINK[args.version], model_dir=ckpt_dir, file_name=ckpt_path.name)

    if args.ae:
        vq_path = Path(args.ae)
    else:
        vq_path = ckpt_dir / "autoencoder_vq_f4.pth"
        if not vq_path.exists():
            load_file_from_url(_LINK["vqgan"], model_dir=ckpt_dir, file_name=vq_path.name)

    configs.model.ckpt_path = str(ckpt_path)
    configs.autoencoder.ckpt_path = str(vq_path)
    configs.diffusion.params.sf = args.scale  # ensure sampler scale matches CLI

    Path(args.out_path).mkdir(parents=True, exist_ok=True)
    return configs


def build_model(configs, args) -> ResShiftSampler:
    return ResShiftSampler(
        configs,
        sf=args.scale,
        chop_size=args.chop_size,
        chop_stride=args.chop_stride,
        chop_bs=args.bs,
        use_amp=not args.no_amp,
        seed=args.seed,
        padding_offset=configs.model.params.get("lq_size", 64),
    )


# ------------------------------
# I/O flows
# ------------------------------
def process_single_slice(
    model: ResShiftSampler,
    nii: nib.Nifti1Image,
    input_path: Path,
    args,
) -> None:
    vol = nii.get_fdata()
    sl2d, used_idx = extract_slice(vol, axis=args.axis, idx=args.slice_idx)
    sl2d = norm_img(sl2d, mode=args.norm, p_low=args.p_low, p_high=args.p_high)

    out_png_name = f"{safe_stem(input_path)}_axis{args.axis}_slice{used_idx}_SR.png"
    sr2d = run_sr_on_slice(model, sl2d, Path(args.out_path), out_png_name, force_rgb=args.repeat3)

    print(f"[OK] Saved SR slice preview -> {Path(args.out_path) / out_png_name}")

    if args.save_nii:
        # Build single-slice SR NIfTI
        Hs, Ws = sr2d.shape
        sr_vol = None
        if args.axis == 2:   # axial
            sr_vol = np.zeros((Hs, Ws, 1), dtype=np.float32)
            sr_vol[:, :, 0] = sr2d
        elif args.axis == 1: # coronal
            sr_vol = np.zeros((Hs, 1, Ws), dtype=np.float32)
            sr_vol[:, 0, :] = sr2d
        else:                # sagittal
            sr_vol = np.zeros((1, Hs, Ws), dtype=np.float32)
            sr_vol[0, :, :] = sr2d

        hdr = nii.header.copy()
        zooms = hdr.get_zooms()[:3]
        new_z = new_zooms_after_inplane_sr(zooms, args.scale, args.axis)
        try:
            hdr.set_zooms(new_z)
        except Exception:
            pass

        out_nii = Path(args.out_path) / f"{safe_stem(input_path)}_axis{args.axis}_slice{used_idx}_SRx{args.scale}.nii.gz"
        nib.save(nib.Nifti1Image(sr_vol, nii.affine, hdr), str(out_nii))
        print(f"[OK] Saved SR NIfTI (single slice) -> {out_nii}")


def process_whole_volume(
    model: ResShiftSampler,
    nii: nib.Nifti1Image,
    input_path: Path,
    args,
) -> None:
    vol = nii.get_fdata()
    assert vol.ndim == 3, f"Expected 3D volume, got {vol.shape}"

    vol_m = move_axis_last(vol, args.axis)
    D = vol_m.shape[-1]

    sr_slices: List[np.ndarray] = []
    out_dir = Path(args.out_path)
    stem = safe_stem(input_path)

    for k in range(D):
        sl = vol_m[..., k]
        sl = norm_img(sl, mode=args.norm, p_low=args.p_low, p_high=args.p_high)
        png_name = f"{stem}_axis{args.axis}_slice{k:04d}_SR.png"
        sr2d = run_sr_on_slice(
            model, sl, out_dir,
            png_name if args.save_png else f"__tmp_{png_name}",
            force_rgb=args.repeat3
        )
        sr_slices.append(sr2d)
        if not args.save_png:
            tmp_path = out_dir / f"__tmp_{png_name}"
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
        if (k + 1) % 10 == 0 or k == D - 1:
            print(f"[{k+1}/{D}] slices processed")

    sr_stack = np.stack(sr_slices, axis=-1)  # (Hs, Ws, D)
    sr_vol = move_axis_back(sr_stack, args.axis).astype(np.float32)

    hdr = nii.header.copy()
    zooms = hdr.get_zooms()[:3]
    new_z = new_zooms_after_inplane_sr(zooms, args.scale, args.axis)
    try:
        hdr.set_zooms(new_z)
    except Exception:
        pass

    out_nii = out_dir / f"{stem}_SRx{args.scale}.nii.gz"
    nib.save(nib.Nifti1Image(sr_vol, nii.affine, hdr), str(out_nii))
    print(f"[OK] Saved SR NIfTI (volume) -> {out_nii}")


def process_png(
    model: ResShiftSampler,
    input_path: Path,
    args,
) -> None:
    # Load single PNG as grayscale [0,1]; no additional normalisation
    arr = read_png_as_float_grayscale(input_path)  # (H,W), e.g., 64x64
    out_png_name = f"{safe_stem(input_path)}_SR.png"

    _ = run_sr_on_slice(
        model,
        arr,
        Path(args.out_path),
        out_png_name,
        force_rgb=args.repeat3
    )

    print(f"[OK] Saved SR PNG -> {Path(args.out_path) / out_png_name}")


# ------------------------------
# CLI
# ------------------------------
def get_parser():
    p = argparse.ArgumentParser(description="MRI SR inference with ResShiftSampler")
    p.add_argument("-i", "--in_path", type=str, required=True,
                   help="Path to MRI .nii/.nii.gz (volume) or a single .png (e.g., 64x64)")
    p.add_argument("-o", "--out_path", type=str, default="./results", help="Output directory")

    p.add_argument("--scale", type=int, default=4, help="Super-resolution scale (in-plane)")
    p.add_argument("--bs", type=int, default=1, help="Tile batch size for inference")
    p.add_argument("--version", type=str, default="v3", choices=["v3"], help="Model version (default v3)")

    # Model/AE overrides (use these to point at MRI-trained checkpoints if you have them)
    p.add_argument("--ckpt", type=str, default=None, help="Path to ResShift checkpoint (.pth) to override default")
    p.add_argument("--ae", type=str, default=None, help="Path to autoencoder checkpoint (.pth) to override default")

    # Normalisation options (MRI only)
    p.add_argument("--norm", type=str, default="percentile", choices=["percentile", "zscore", "minmax"],
                   help="Slice normalisation mode (NIfTI only)")
    p.add_argument("--p_low", type=float, default=1.0, help="Low percentile (for percentile norm)")
    p.add_argument("--p_high", type=float, default=99.0, help="High percentile (for percentile norm)")

    # Slicing options (MRI only)
    p.add_argument("--axis", type=int, default=2, choices=[0, 1, 2], help="Axis to slice along (0=sagittal,1=coronal,2=axial)")
    p.add_argument("--slice_idx", type=int, default=None, help="Slice index (default: middle slice)")

    # Output modes (MRI)
    p.add_argument("--all_slices", action="store_true", help="Process the entire volume and save NIfTI")
    p.add_argument("--save_nii", action="store_true", help="Also save NIfTI for single-slice mode")
    p.add_argument("--save_png", action="store_true", help="Keep per-slice PNGs during whole-volume processing")

    # Sampler/tiling controls
    p.add_argument("--chop_size", type=int, default=256, help="Tile size for inference")
    p.add_argument("--chop_stride", type=int, default=192, help="Tile stride/overlap for inference")
    p.add_argument("--no_amp", action="store_true", help="Disable AMP (use full float32)")
    p.add_argument("--seed", type=int, default=123, help="Random seed for sampler")

    # Channel repeat (use if your checkpoint expects RGB)
    p.add_argument("--repeat3", action="store_true",
                   help="Repeat single-channel inputs to 3 channels before inference (for RGB-trained checkpoints).")

    return p.parse_args()


def main():
    args = get_parser()

    input_path = Path(args.in_path)

    configs = get_configs(args)
    model = build_model(configs, args)

    if is_nii_like(input_path):
        nii = nib.load(str(input_path))
        if args.all_slices:
            process_whole_volume(model, nii, input_path, args)
        else:
            process_single_slice(model, nii, input_path, args)
    elif is_png_like(input_path):
        process_png(model, input_path, args)
    else:
        raise ValueError(
            f"Unsupported input for: {input_path.name} "
            f"(expected .nii/.nii.gz for volumes or .png for a single image)"
        )


if __name__ == "__main__":
    main()
