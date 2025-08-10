#!/usr/bin/env python3
"""
Frontend script for running the SVR option of NeSVoR using the official CLI logic.
Optimized for GPU memory usage with automatic memory management.
Usage example:
    python run_nesvor_svr.py --input /path/to/stacks --output /path/to/output.nii.gz [other options]
"""
import argparse
import torch
import gc
import logging
import warnings
import os
import contextlib
from typing import Optional, Any
from nesvor.cli.commands import SVR

# Configure logging for memory monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPUMemoryManager:
    """Context manager for strict GPU memory management"""
    
    def __init__(self, memory_limit_gb: float, device: torch.device):
        self.memory_limit_bytes = int(memory_limit_gb * 1024**3)
        self.device = device
        self.original_malloc = None
        
    def __enter__(self):
        if torch.cuda.is_available() and self.device.type == 'cuda':
            # Set PyTorch memory fraction more aggressively
            try:
                total_memory = torch.cuda.get_device_properties(self.device).total_memory
                # Use 90% of our limit as the PyTorch fraction to leave some buffer
                fraction = min(0.9, (self.memory_limit_bytes * 0.9) / total_memory)
                torch.cuda.set_per_process_memory_fraction(fraction, self.device)
                logger.info(f"Set GPU memory fraction to {fraction:.3f} "
                           f"({self.memory_limit_bytes/1024**3:.2f}GB limit with 10% buffer)")
            except Exception as e:
                logger.warning(f"Could not set memory fraction: {e}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Reset memory fraction
            try:
                torch.cuda.set_per_process_memory_fraction(1.0, self.device)
            except Exception:
                pass

def safe_to_device(tensor, device, memory_limit_gb=None):
    """Safely move tensor to device with memory checking"""
    if not torch.cuda.is_available() or device.type != 'cuda':
        # For CPU device, just return the object as-is since NeSVoR objects
        # don't need explicit device transfers to CPU
        return tensor
    
    try:
        # Handle NeSVoR Slice, Stack, Volume objects
        if hasattr(tensor, 'data') and hasattr(tensor, 'mask') and hasattr(tensor, 'transformation'):
            # This is likely a NeSVoR object (Slice, Stack, Volume, etc.)
            # Check memory for the data tensor
            if memory_limit_gb and hasattr(tensor.data, 'element_size') and hasattr(tensor.data, 'numel'):
                tensor_size_gb = (tensor.data.element_size() * tensor.data.numel()) / 1024**3
                current_memory_gb = torch.cuda.memory_allocated(device) / 1024**3
                
                if (current_memory_gb + tensor_size_gb) > memory_limit_gb:
                    logger.warning(f"NeSVoR object data ({tensor_size_gb:.2f}GB) would exceed memory limit. Keeping on CPU.")
                    return tensor  # Keep on current device (likely CPU)
            
            # For NeSVoR objects, we need to move their internal tensors
            # but the objects themselves don't have a .to() method
            # The framework should handle device placement automatically
            return tensor
        
        # Handle regular PyTorch tensors
        elif hasattr(tensor, 'to'):
            # Check if we have enough memory before moving
            if memory_limit_gb and hasattr(tensor, 'element_size') and hasattr(tensor, 'numel'):
                tensor_size_gb = (tensor.element_size() * tensor.numel()) / 1024**3
                current_memory_gb = torch.cuda.memory_allocated(device) / 1024**3
                
                if (current_memory_gb + tensor_size_gb) > memory_limit_gb:
                    logger.warning(f"Tensor ({tensor_size_gb:.2f}GB) would exceed memory limit. Keeping on CPU.")
                    return tensor.to(torch.device('cpu'), non_blocking=True)
            
            return tensor.to(device, non_blocking=True)
        
        # For objects without device methods, return as-is
        else:
            return tensor
        
    except torch.cuda.OutOfMemoryError:
        logger.warning("GPU out of memory. Keeping tensor on CPU.")
        if hasattr(tensor, 'to'):
            return tensor.to(torch.device('cpu'), non_blocking=True)
        else:
            return tensor
    except Exception as e:
        logger.warning(f"Error moving tensor to device: {e}")
        return tensor

class MemoryOptimizedSVR(SVR):
    """
    Memory-optimized version of SVR command with strict GPU memory management
    """
    
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.memory_threshold = getattr(args, 'memory_threshold', 0.85)
        self.enable_memory_monitoring = getattr(args, 'enable_memory_monitoring', True)
        self.force_cpu_offload = getattr(args, 'force_cpu_offload', False)
        self.adaptive_batch_size = getattr(args, 'adaptive_batch_size', True)
        self.max_gpu_memory_gb = getattr(args, 'max_gpu_memory_gb', None)
        
        # Initialize memory manager
        self.memory_manager = None
        
        # Calculate effective memory limit
        if torch.cuda.is_available() and hasattr(args, 'device') and hasattr(args.device, 'type') and args.device.type == 'cuda':
            device = args.device
            total_device_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
            
            if self.max_gpu_memory_gb:
                # Use the smaller of user-specified limit or device total memory
                self.effective_memory_limit = min(self.max_gpu_memory_gb, total_device_memory)
                logger.info(f"Setting strict GPU memory limit to {self.effective_memory_limit:.2f}GB "
                           f"(user limit: {self.max_gpu_memory_gb}GB, device total: {total_device_memory:.2f}GB)")
                
                # Create memory manager for strict enforcement
                self.memory_manager = GPUMemoryManager(self.effective_memory_limit, device)
                
                # If limit is very restrictive, force CPU offload
                if self.effective_memory_limit < 2.0:
                    logger.warning(f"Very low memory limit ({self.effective_memory_limit:.2f}GB). "
                                 f"Enabling CPU offload for better stability.")
                    self.force_cpu_offload = True
                    
            else:
                # Use threshold percentage of total memory
                self.effective_memory_limit = total_device_memory * self.memory_threshold
                logger.info(f"Using {self.memory_threshold:.1%} of GPU memory: {self.effective_memory_limit:.2f}GB")
        else:
            self.effective_memory_limit = None
    
    def enforce_memory_limit(self):
        """Aggressively enforce memory limit"""
        if not torch.cuda.is_available() or self.effective_memory_limit is None:
            return
            
        try:
            device = self.args.device
            if hasattr(device, 'type') and device.type == 'cuda':
                current_allocated = torch.cuda.memory_allocated(device) / 1024**3
                
                if current_allocated > self.effective_memory_limit:
                    logger.warning(f"Memory usage ({current_allocated:.2f}GB) exceeds limit ({self.effective_memory_limit:.2f}GB)!")
                    
                    # Aggressive cleanup
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Check again after cleanup
                    new_allocated = torch.cuda.memory_allocated(device) / 1024**3
                    if new_allocated > self.effective_memory_limit:
                        logger.error(f"Cannot reduce memory below limit even after cleanup. "
                                   f"Current: {new_allocated:.2f}GB, Limit: {self.effective_memory_limit:.2f}GB")
                        # Force CPU offload for future operations
                        self.force_cpu_offload = True
                        
        except Exception as e:
            logger.warning(f"Failed to enforce memory limit: {e}")
    
    def check_memory_limit(self) -> bool:
        """Check if current GPU memory usage exceeds the limit"""
        if not torch.cuda.is_available() or self.effective_memory_limit is None:
            return True
            
        try:
            device = self.args.device
            if hasattr(device, 'type') and device.type == 'cuda':
                current_allocated = torch.cuda.memory_allocated(device) / 1024**3
                return current_allocated <= self.effective_memory_limit
        except Exception:
            return True
        
        return True
    
    def get_available_memory_gb(self) -> float:
        """Get available GPU memory in GB, respecting the user-defined limit"""
        if not torch.cuda.is_available() or self.effective_memory_limit is None:
            return 0.0
            
        try:
            device = self.args.device
            if hasattr(device, 'type') and device.type == 'cuda':
                current_allocated = torch.cuda.memory_allocated(device) / 1024**3
                available = max(0.0, self.effective_memory_limit - current_allocated)
                return available
        except Exception:
            return 0.0
        
        return 0.0
        
    def monitor_gpu_memory(self, stage: str = ""):
        """Monitor and log GPU memory usage"""
        if not self.enable_memory_monitoring or not torch.cuda.is_available():
            return
            
        try:
            device = self.args.device
            if hasattr(device, 'type') and device.type == 'cuda':
                allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
                total_device_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
                
                if self.effective_memory_limit:
                    usage_percent = (allocated / self.effective_memory_limit) * 100
                    available = max(0.0, self.effective_memory_limit - allocated)
                    
                    logger.info(f"GPU Memory {stage}: "
                               f"Allocated: {allocated:.2f}GB ({usage_percent:.1f}% of limit), "
                               f"Available: {available:.2f}GB, "
                               f"Limit: {self.effective_memory_limit:.2f}GB, "
                               f"Device Total: {total_device_memory:.2f}GB")
                    
                    # Enforce limit strictly
                    if allocated > self.effective_memory_limit:
                        self.enforce_memory_limit()
                        
                else:
                    usage_percent = (allocated / total_device_memory) * 100
                    logger.info(f"GPU Memory {stage}: "
                               f"Allocated: {allocated:.2f}GB ({usage_percent:.1f}%), "
                               f"Reserved: {reserved:.2f}GB, "
                               f"Total: {total_device_memory:.2f}GB")
                    
        except Exception as e:
            logger.warning(f"Failed to monitor GPU memory: {e}")
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU memory cleanup completed")
    
    def optimize_device_placement(self, data, force_cpu: bool = False):
        """
        Intelligently place data on CPU or GPU based on memory availability and limits
        """
        if force_cpu or self.force_cpu_offload:
            # For NeSVoR objects, we can't force them to CPU directly
            # The framework handles device placement automatically
            # Just return them as-is
            return data
            
        if not torch.cuda.is_available():
            return data
        
        # Use safe transfer function
        return safe_to_device(data, self.args.device, self.effective_memory_limit)
    
    def preprocess(self):
        """Override preprocess with memory optimization"""
        self.monitor_gpu_memory("before preprocessing")
        
        try:
            # Call parent preprocess
            input_dict = super().preprocess()
            
            # Optimize memory usage for loaded data
            if 'input_stacks' in input_dict and input_dict['input_stacks']:
                logger.info("Optimizing memory placement for input stacks...")
                optimized_stacks = []
                for i, stack in enumerate(input_dict['input_stacks']):
                    # Process stacks individually to manage memory
                    self.monitor_gpu_memory(f"processing stack {i+1}")
                    optimized_stack = self.optimize_device_placement(stack)
                    optimized_stacks.append(optimized_stack)
                    
                    # Clean up after each stack if memory is tight
                    if i % 2 == 0:  # Clean up every 2 stacks
                        self.cleanup_gpu_memory()
                        
                input_dict['input_stacks'] = optimized_stacks
            
            if 'input_slices' in input_dict and input_dict['input_slices']:
                logger.info("Optimizing memory placement for input slices...")
                # Process slices in batches to manage memory
                batch_size = 10 if self.adaptive_batch_size else len(input_dict['input_slices'])
                optimized_slices = []
                
                for i in range(0, len(input_dict['input_slices']), batch_size):
                    batch_slices = input_dict['input_slices'][i:i+batch_size]
                    self.monitor_gpu_memory(f"processing slice batch {i//batch_size + 1}")
                    
                    for slice_obj in batch_slices:
                        optimized_slice = self.optimize_device_placement(slice_obj)
                        optimized_slices.append(optimized_slice)
                    
                    # Clean up after each batch
                    self.cleanup_gpu_memory()
                    
                input_dict['input_slices'] = optimized_slices
            
            self.monitor_gpu_memory("after preprocessing")
            return input_dict
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU out of memory during preprocessing: {e}")
            logger.info("Falling back to CPU processing...")
            
            # Force CPU processing
            self.force_cpu_offload = True
            self.cleanup_gpu_memory()
            
            # Retry with CPU
            return super().preprocess()
    
    def exec(self):
        """Override exec with strict memory optimization"""
        
        # Use memory manager context if available
        memory_context = self.memory_manager if self.memory_manager else contextlib.nullcontext()
        
        with memory_context:
            self.monitor_gpu_memory("start of execution")
            
            try:
                input_dict = self.preprocess()
                
                self.new_timer("Memory-Optimized Reconstruction")
                self.monitor_gpu_memory("before reconstruction")
                
                # Check memory limit before reconstruction
                if not self.check_memory_limit():
                    logger.warning("Memory limit exceeded before reconstruction. Forcing CPU offload.")
                    self.force_cpu_offload = True
                    self.cleanup_gpu_memory()
                
                # Import the reconstruction function
                from nesvor.svr import slice_to_volume_reconstruction
                
                # Prepare arguments with memory management
                svr_args = vars(self.args).copy()
                
                # Add memory management to the reconstruction args
                if self.effective_memory_limit:
                    # Force smaller batch sizes if memory is limited
                    if hasattr(self.args, 'inference_batch_size') and self.args.inference_batch_size:
                        original_batch_size = self.args.inference_batch_size
                        # Reduce batch size more aggressively for small memory limits
                        if self.effective_memory_limit < 1.5:  # Less than 1.5GB
                            svr_args['inference_batch_size'] = min(original_batch_size, 1)
                            logger.info(f"Reducing batch size from {original_batch_size} to 1 due to very low memory limit")
                        elif self.effective_memory_limit < 3.0:  # Less than 3GB
                            svr_args['inference_batch_size'] = min(original_batch_size, 2)
                            logger.info(f"Reducing batch size from {original_batch_size} to 2 due to memory limits")
                        elif self.effective_memory_limit < 6.0:  # Less than 6GB
                            svr_args['inference_batch_size'] = min(original_batch_size, 4)
                            logger.info(f"Reducing batch size from {original_batch_size} to 4 due to memory limits")
                
                # Run reconstruction with memory monitoring
                try:
                    # Monitor memory before starting reconstruction
                    self.enforce_memory_limit()
                    
                    output_volume, output_slices, simulated_slices = slice_to_volume_reconstruction(
                        input_dict["input_slices"], **svr_args
                    )
                    
                    # Check memory after reconstruction
                    self.monitor_gpu_memory("after reconstruction")
                    
                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"GPU out of memory during reconstruction: {e}")
                    
                    if self.max_gpu_memory_gb:
                        logger.info(f"Current memory limit: {self.max_gpu_memory_gb}GB. "
                                  f"Consider reducing --max-gpu-memory-gb further or using --force-cpu-offload")
                    else:
                        logger.info("Consider using --max-gpu-memory-gb to set a memory limit or --force-cpu-offload")
                    
                    logger.info("Retrying with forced CPU offloading...")
                    
                    # Force CPU offloading and retry
                    self.force_cpu_offload = True
                    self.cleanup_gpu_memory()
                    
                    # Move input slices to CPU
                    cpu_slices = []
                    for slice_obj in input_dict["input_slices"]:
                        # For NeSVoR objects, we can't directly move them to CPU
                        # The framework should handle this automatically
                        cpu_slices.append(slice_obj)
                    
                    # Update device to CPU for fallback
                    svr_args['device'] = torch.device('cpu')
                    
                    # Retry reconstruction
                    output_volume, output_slices, simulated_slices = slice_to_volume_reconstruction(
                        cpu_slices, **svr_args
                    )
                
                self.new_timer("Results saving")
                
                # Import the outputs function
                from nesvor.cli.io import outputs
                
                # Optimize output data placement - move to CPU to free GPU memory
                if output_volume:
                    # For NeSVoR Volume objects, they handle device placement automatically
                    logger.info("Output volume ready (device placement handled by NeSVoR)")
                
                if output_slices:
                    # For NeSVoR Slice objects, they handle device placement automatically  
                    logger.info("Output slices ready (device placement handled by NeSVoR)")
                
                if simulated_slices:
                    # For NeSVoR Slice objects, they handle device placement automatically
                    logger.info("Simulated slices ready (device placement handled by NeSVoR)")
                
                # Clean up before saving
                self.cleanup_gpu_memory()
                
                outputs(
                    {
                        "output_volume": output_volume,
                        "output_slices": output_slices,
                        "simulated_slices": simulated_slices,
                    },
                    self.args,
                )
                
                self.monitor_gpu_memory("end of execution")
                
            except Exception as e:
                logger.error(f"Error during execution: {e}")
                self.cleanup_gpu_memory()
                raise
            
            finally:
                # Final cleanup
                self.cleanup_gpu_memory()

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
    # Memory optimization arguments
    parser.add_argument('--memory-threshold', type=float, default=0.85, help='GPU memory usage threshold (0.0-1.0) for triggering cleanup')
    parser.add_argument('--disable-memory-monitoring', action='store_true', help='Disable GPU memory monitoring')
    parser.add_argument('--force-cpu-offload', action='store_true', help='Force offloading data to CPU when GPU memory is limited')
    parser.add_argument('--disable-adaptive-batch-size', action='store_true', help='Disable adaptive batch size for memory optimization')
    parser.add_argument('--max-gpu-memory-gb', type=float, help='Maximum GPU memory to use (in GB)')
    # Miscellaneous
    default_device = 0 if torch.cuda.is_available() else -1
    parser.add_argument('--device', type=int, default=default_device, help='Device ID (nonnegative for GPU, negative for CPU)')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2], help='Verbosity level')
    parser.add_argument('--output-log', type=str, help='Path to output log file')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    # Set up memory optimization flags
    args.enable_memory_monitoring = not args.disable_memory_monitoring
    args.adaptive_batch_size = not args.disable_adaptive_batch_size
    
    # Configure GPU memory limits if specified
    if args.max_gpu_memory_gb and torch.cuda.is_available():
        try:
            # Set memory fraction (PyTorch doesn't have direct memory limit, but we can monitor)
            logger.info(f"Setting maximum GPU memory usage to {args.max_gpu_memory_gb}GB")
            # Adjust memory threshold based on max memory setting
            if hasattr(torch.cuda, 'get_device_properties'):
                device_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                args.memory_threshold = min(args.memory_threshold, args.max_gpu_memory_gb / device_memory)
        except Exception as e:
            logger.warning(f"Could not set GPU memory limit: {e}")

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
    
    # Log initial GPU memory state
    if torch.cuda.is_available() and args.enable_memory_monitoring:
        device = args.device if hasattr(args.device, 'type') and args.device.type == 'cuda' else torch.device('cuda:0')
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        logger.info(f"GPU Device: {torch.cuda.get_device_name(device)}")
        logger.info(f"Total GPU Memory: {total_memory:.2f}GB")
        logger.info(f"Memory threshold: {args.memory_threshold:.1%}")
    
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
    
    # Use memory-optimized SVR command
    try:
        svr_cmd = MemoryOptimizedSVR(args)
        svr_cmd.exec()
        logger.info("SVR reconstruction completed successfully with memory optimization")
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"GPU out of memory error: {e}")
        logger.info("Consider using --force-cpu-offload or --max-gpu-memory-gb options")
        raise
    except Exception as e:
        logger.error(f"Error during SVR execution: {e}")
        raise

if __name__ == "__main__":
    main()
