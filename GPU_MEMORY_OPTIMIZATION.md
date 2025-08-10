# GPU Memory Optimization for NeSVoR SVR

The `run_nesvor_svr.py` script now includes advanced GPU memory management features to optimize memory usage and prevent out-of-memory errors.

## New Memory Optimization Arguments

### Memory Limiting
- `--max-gpu-memory-gb LIMIT`: Set maximum GPU memory usage in GB
  - Example: `--max-gpu-memory-gb 4.0` limits GPU usage to 4GB
  - Automatically keeps memory usage below this limit
  - Offloads data to CPU when necessary

- `--memory-threshold THRESHOLD`: Set memory usage threshold (0.0-1.0)
  - Default: 0.85 (85% of available memory)
  - Example: `--memory-threshold 0.7` triggers cleanup at 70% usage

### Memory Monitoring
- `--disable-memory-monitoring`: Disable GPU memory monitoring and logging
  - By default, memory usage is logged at each processing stage
  - Shows allocated memory, available memory, and usage percentages

### Memory Management
- `--force-cpu-offload`: Force data to be kept on CPU when GPU memory is limited
  - Useful for very large datasets or limited GPU memory
  - Automatically enabled if GPU memory runs out

- `--disable-adaptive-batch-size`: Disable automatic batch size reduction
  - By default, batch sizes are reduced when memory is limited
  - Helps prevent out-of-memory errors during processing

## Usage Examples

### Basic Memory Limiting
```bash
# Limit GPU memory usage to 4GB
python run_nesvor_svr.py \
    --input-stacks stack1.nii.gz stack2.nii.gz \
    --output output.nii.gz \
    --max-gpu-memory-gb 4.0
```

### Conservative Memory Usage
```bash
# Use only 60% of GPU memory with monitoring
python run_nesvor_svr.py \
    --input-stacks stack1.nii.gz stack2.nii.gz \
    --output output.nii.gz \
    --memory-threshold 0.6 \
    --max-gpu-memory-gb 3.0
```

### Force CPU Processing for Large Datasets
```bash
# Force CPU offload for very large datasets
python run_nesvor_svr.py \
    --input-stacks large_stack1.nii.gz large_stack2.nii.gz \
    --output output.nii.gz \
    --force-cpu-offload
```

### Minimal Memory Footprint
```bash
# Minimize GPU memory usage
python run_nesvor_svr.py \
    --input-stacks stack1.nii.gz stack2.nii.gz \
    --output output.nii.gz \
    --max-gpu-memory-gb 2.0 \
    --memory-threshold 0.5 \
    --force-cpu-offload
```

## How It Works

### Memory Monitoring
- Continuously monitors GPU memory usage at each processing stage
- Logs detailed memory statistics including:
  - Currently allocated memory
  - Available memory within the user-defined limit
  - Percentage usage relative to the limit
  - Total device memory

### Intelligent Data Placement
- Automatically decides whether to place data on GPU or CPU
- Considers:
  - Current memory usage
  - Size of new data
  - User-defined memory limits
  - Available memory buffer

### Automatic Fallback
- If GPU memory is exceeded, automatically:
  1. Cleans up GPU memory
  2. Moves data to CPU
  3. Retries processing with CPU offload
  4. Continues with hybrid CPU/GPU processing

### Memory Cleanup
- Automatic garbage collection and cache clearing
- Triggered when memory usage exceeds thresholds
- Final cleanup at the end of processing

## Memory Usage Tips

### For 4GB GPUs or Less
```bash
--max-gpu-memory-gb 3.0 --memory-threshold 0.7
```

### For 6-8GB GPUs
```bash
--max-gpu-memory-gb 6.0 --memory-threshold 0.8
```

### For 12GB+ GPUs
```bash
--max-gpu-memory-gb 10.0 --memory-threshold 0.85
```

### When Processing Very Large Datasets
```bash
--force-cpu-offload --disable-adaptive-batch-size
```

## Troubleshooting

### Out of Memory Errors
1. Reduce `--max-gpu-memory-gb` value
2. Lower `--memory-threshold` (e.g., 0.6 or 0.5)
3. Use `--force-cpu-offload`
4. Process fewer stacks at once

### Slow Processing
1. Increase `--max-gpu-memory-gb` if you have available GPU memory
2. Increase `--memory-threshold` (e.g., 0.9)
3. Avoid `--force-cpu-offload` unless necessary

### Memory Monitoring
- Use `--disable-memory-monitoring` to reduce log output
- Monitor the log output to understand memory usage patterns
- Adjust limits based on observed memory usage

## Performance Impact

- **Memory Monitoring**: Minimal overhead (~1-2% performance impact)
- **Automatic Data Placement**: Slight overhead for memory checking
- **CPU Offloading**: Slower than pure GPU processing, but prevents crashes
- **Adaptive Batch Sizing**: May reduce throughput but improves stability

The memory optimization features provide a good balance between performance and stability, ensuring that SVR reconstruction can complete successfully even with limited GPU memory.
