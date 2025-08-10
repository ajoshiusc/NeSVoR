#!/usr/bin/env python3
"""
Frontend script for running the SVR option of NeSVoR using the official CLI logic.
Usage example:
    python run_nesvor_svr.py --input /path/to/stacks --output /path/to/output.nii.gz [other options]
"""
import argparse
import torch
from nesvor.cli.commands import SVR

def main():
    parser = argparse.ArgumentParser(description="Frontend for NeSVoR SVR (Slice-to-Volume Registration)")
    # Input arguments
    parser.add_argument('--input', type=str, required=False, help='(DEPRECATED) Path to input slices folder (expects NIfTI stacks)')
    parser.add_argument('--input-stacks', type=str, nargs='+', required=False, help='Paths to input stack files (expects NIfTI stacks, e.g. stack-1.nii.gz stack-2.nii.gz ...)')
    parser.add_argument('--input-slices', type=str, required=False, help='Folder of the input slices (motion corrected)')
    parser.add_argument('--stack-masks', type=str, nargs='+', required=False, help='Paths to masks of input stacks')
    parser.add_argument('--thicknesses', type=float, nargs='+', required=False, help='Slice thickness of each input stack')
    parser.add_argument('--volume-mask', type=str, required=False, help='Path to a 3D mask applied to each input stack')
    parser.add_argument('--stacks-intersection', action='store_true', help='Only consider region defined by intersection of input stacks')
    parser.add_argument('--background-threshold', type=float, default=0.0, help='Background threshold for stack masking')
    parser.add_argument('--otsu-thresholding', action='store_true', help='Apply Otsu thresholding to each input stack')
    # Output arguments
    parser.add_argument('--output', type=str, required=True, help='Path to save output volume (NIfTI .nii.gz)')
    parser.add_argument('--output-volume', type=str, required=False, help='Path to save output volume (NIfTI .nii.gz)')
    parser.add_argument('--output-slices', type=str, required=False, help='Folder to save motion corrected slices')
    parser.add_argument('--simulated-slices', type=str, required=False, help='Folder to save simulated slices from reconstructed volume')
    parser.add_argument('--output-model', type=str, required=False, help='Path to save output model (.pt)')
    parser.add_argument('--output-stack-masks', type=str, nargs='+', required=False, help='Path(s) to output masks')
    parser.add_argument('--output-corrected-stacks', type=str, nargs='+', required=False, help='Path(s) to output corrected stacks')
    parser.add_argument('--output-json', type=str, required=False, help='Path to save inputs/results as JSON')
    # Output sampling arguments
    parser.add_argument('--resolution', type=float, default=0.8, help='Output volume resolution')
    parser.add_argument('--intensity', type=float, default=700, help='Output intensity mean')
    parser.add_argument('--output-resolution', type=float, default=0.8, help='Isotropic resolution of the reconstructed volume')
    parser.add_argument('--output-intensity-mean', type=float, default=700.0, help='Mean intensity of the output volume')
    parser.add_argument('--inference-batch-size', type=int, help='Batch size for inference')
    parser.add_argument('--n-inference-samples', type=int, help='Number of samples for PSF during inference')
    parser.add_argument('--output-psf-factor', type=float, default=1.0, help='PSF factor for output volume')
    parser.add_argument('--sample-orientation', type=str, help='Path to nii file for volume reorientation')
    parser.add_argument('--sample-mask', type=str, help='3D mask for sampling INR')
    # SVR/registration arguments
    parser.add_argument('--registration', type=str, default='svort', choices=['svort', 'svort-only', 'svort-stack', 'stack', 'none'], help='Registration method')
    parser.add_argument('--svort-version', type=str, default='v2', choices=['v1', 'v2'], help='Version of SVoRT model')
    parser.add_argument('--scanner-space', action='store_true', help='Perform registration in scanner space')
    # Segmentation
    parser.add_argument('--segmentation', action='store_true', help='Enable brain segmentation preprocessing')
    parser.add_argument('--batch-size-seg', type=int, default=16, help='Batch size for segmentation')
    parser.add_argument('--no-augmentation-seg', action='store_true', help='Disable inference data augmentation in segmentation')
    parser.add_argument('--dilation-radius-seg', type=float, default=1.0, help='Dilation radius for segmentation mask (mm)')
    parser.add_argument('--threshold-small-seg', type=float, default=0.1, help='Threshold for removing small segmentation masks')
    # Bias field correction
    parser.add_argument('--bias-field-correction', action='store_true', help='Enable bias field correction (N4 algorithm)')
    parser.add_argument('--n-proc-n4', type=int, default=8, help='Number of workers for N4 algorithm')
    parser.add_argument('--shrink-factor-n4', type=int, default=2, help='Shrink factor for N4')
    parser.add_argument('--tol-n4', type=float, default=0.001, help='Convergence threshold for N4')
    parser.add_argument('--spline-order-n4', type=int, default=3, help='Order of B-spline for N4')
    parser.add_argument('--noise-n4', type=float, default=0.01, help='Noise estimate for N4')
    parser.add_argument('--n-iter-n4', type=int, default=50, help='Max iterations for N4')
    parser.add_argument('--n-levels-n4', type=int, default=4, help='Max levels for N4')
    parser.add_argument('--n-control-points-n4', type=int, default=4, help='Control points for N4')
    parser.add_argument('--n-bins-n4', type=int, default=200, help='Bins in log intensity histogram for N4')
    # Assessment
    parser.add_argument('--metric', type=str, choices=['ncc', 'matrix-rank', 'volume', 'iqa2d', 'iqa3d', 'none'], default='none', help='Metric for stack assessment')
    parser.add_argument('--filter-method', type=str, choices=['top', 'bottom', 'threshold', 'percentage', 'none'], default='none', help='Method to remove low-quality stacks')
    parser.add_argument('--cutoff', type=float, help='Cutoff value for filtering')
    parser.add_argument('--batch-size-assess', type=int, default=8, help='Batch size for IQA network')
    parser.add_argument('--no-augmentation-assess', action='store_true', help='Disable augmentation in IQA network')
    # Outlier removal
    parser.add_argument('--no-slice-robust-statistics', action='store_true', help='Disable slice-level robust statistics for outlier removal')
    parser.add_argument('--no-pixel-robust-statistics', action='store_true', help='Disable pixel-level robust statistics for outlier removal')
    parser.add_argument('--no-local-exclusion', action='store_true', help='Disable pixel-level exclusion based on SSIM')
    parser.add_argument('--no-global-exclusion', action='store_true', help='Disable slice-level exclusion based on NCC')
    parser.add_argument('--global-ncc-threshold', type=float, default=0.1, help='Threshold for global exclusion')
    parser.add_argument('--local-ssim-threshold', type=float, default=0.1, help='Threshold for local exclusion')
    # Optimization
    parser.add_argument('--with-background', action='store_true', help='Reconstruct the background in the volume')
    parser.add_argument('--n_iter', type=int, default=3, help='Number of outer iterations')
    parser.add_argument('--n_iter_rec', type=int, nargs='+', default=[7, 7, 21], help='Number of inner iterations per outer iteration')
    parser.add_argument('--psf', type=str, default='gaussian', choices=['gaussian', 'sinc'], help='PSF type (gaussian or sinc)')
    # Regularization
    parser.add_argument('--delta', type=float, default=0.2, help='Parameter for edge-preserving regularization')
    # Miscellaneous
    default_device = 0 if torch.cuda.is_available() else -1
    parser.add_argument('--device', type=int, default=default_device, help='Device ID (nonnegative for GPU, negative for CPU)')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2], help='Verbosity level')
    parser.add_argument('--output-log', type=str, help='Path to output log file')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    # Fix parameter names to match CLI expectations (replace underscores with hyphens)
    args.n_iter = getattr(args, 'n_iter', 3)
    args.n_iter_rec = getattr(args, 'n_iter_rec', [7, 7, 21])
    args.global_ncc_threshold = getattr(args, 'global_ncc_threshold', 0.5)
    args.local_ssim_threshold = getattr(args, 'local_ssim_threshold', 0.4)
    
    # Convert device ID to torch device
    if args.device >= 0:
        args.device = torch.device(f'cuda:{args.device}')
    else:
        args.device = torch.device('cpu')
    
    # Prepare args for SVR command
    args.output_volume = args.output
    # Handle output sampling arguments
    if not hasattr(args, 'output_resolution'):
        args.output_resolution = getattr(args, 'resolution', 0.8)
    if not hasattr(args, 'output_intensity_mean'):
        args.output_intensity_mean = getattr(args, 'intensity', 700)
    
    # Prefer --input-stacks if provided, else fallback to --input
    if args.input_stacks:
        pass
    elif args.input:
        args.input_stacks = args.input
    else:
        raise ValueError("Either --input-stacks or --input must be provided")
    
    # SVR expects args as Namespace
    svr_cmd = SVR(args)
    svr_cmd.exec()

if __name__ == "__main__":
    main()
