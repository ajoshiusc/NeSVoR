#!/usr/bin/env python3
"""
Main entry point for running the SVR option of NeSVoR from the main repo.
Usage example:
    python run_svr.py --input /path/to/stacks --output /path/to/output.nii.gz --resolution 0.8 --intensity 700 --n_iter 3 --device cpu
"""
import argparse
import torch
from nesvor.svr.pipeline import slice_to_volume_reconstruction
## from nesvor.image.image import load_slices  # unused
from nesvor.image.image_utils import save_nii_volume


def main():
    parser = argparse.ArgumentParser(description="Run NeSVoR SVR (Slice-to-Volume Registration)")
    parser.add_argument('--input', type=str, required=True, help='Path to input slices folder (expects NIfTI slices)')
    parser.add_argument('--output', type=str, required=True, help='Path to save output volume (NIfTI .nii.gz)')
    parser.add_argument('--resolution', type=float, default=0.8, help='Output volume resolution')
    parser.add_argument('--intensity', type=float, default=700, help='Output intensity mean')
    parser.add_argument('--n_iter', type=int, default=3, help='Number of outer iterations')
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default=default_device, help='Device to run on (cpu or cuda)')
    args = parser.parse_args()

    print(f"Loading slices from {args.input} ...")
    import os
    from nesvor.image.image_utils import load_nii_volume, affine2transformation
    from nesvor.image.image import Slice
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA device specified but not available.')
    stack_files = [f for f in os.listdir(args.input) if (f.endswith("nii") or f.endswith("nii.gz")) and not f.startswith("mask-")]
    stack_files = sorted(stack_files)
    slices = []
    for f in stack_files:
        stack_data, resolutions, affine = load_nii_volume(os.path.join(args.input, f))
        stack_tensor = torch.tensor(stack_data, dtype=torch.float32).to(device)
        mask_path = os.path.join(args.input, "mask-" + f)
        if os.path.exists(mask_path):
            mask_data, _, _ = load_nii_volume(mask_path)
            mask_tensor = torch.tensor(mask_data, dtype=torch.bool).to(device)
        else:
            mask_tensor = torch.ones_like(stack_tensor, dtype=torch.bool).to(device)
        # Split stack into individual slices along axis 0
        num_slices = stack_tensor.shape[0]
        for idx in range(num_slices):
            slice_tensor = stack_tensor[idx].to(device)
            mask_slice_tensor = mask_tensor[idx].to(device) if mask_tensor.shape[0] == num_slices else mask_tensor.to(device)
            # Guarantee shape (1, H, W)
            if slice_tensor.ndim == 2:
                slice_tensor = slice_tensor.unsqueeze(0)
            while slice_tensor.ndim > 3:
                slice_tensor = slice_tensor.squeeze()
            if mask_slice_tensor.ndim == 2:
                mask_slice_tensor = mask_slice_tensor.unsqueeze(0)
            while mask_slice_tensor.ndim > 3:
                mask_slice_tensor = mask_slice_tensor.squeeze()
            # Affine transformation for each slice
            slice_tensor, mask_slice_tensor, transformation = affine2transformation(
                slice_tensor, mask_slice_tensor, resolutions, affine
            )
            slices.append(
                Slice(
                    image=slice_tensor,
                    mask=mask_slice_tensor,
                    transformation=transformation,
                    resolution_x=resolutions[0],
                    resolution_y=resolutions[1],
                    resolution_z=resolutions[2],
                )
            )
    # Ensure all slices and their attributes are on the correct device
    for s in slices:
        s.image = s.image.to(device)
        s.mask = s.mask.to(device)
        if hasattr(s, 'transformation') and hasattr(s.transformation, 'matrix'):
            if hasattr(s.transformation, 'to_device'):
                s.transformation = s.transformation.to_device(device)
    print("Running SVR...")
    volume, _, _ = slice_to_volume_reconstruction(
        slices,
        output_resolution=args.resolution,
        output_intensity_mean=args.intensity,
        n_iter=args.n_iter,
        device=device
    )
    print(f"Saving output volume to {args.output} ...")
    save_nii_volume(args.output, volume.image, None)  # None for affine if not available
    print("Done.")

if __name__ == "__main__":
    main()
