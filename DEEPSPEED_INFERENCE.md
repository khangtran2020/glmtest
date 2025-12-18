# DeepSpeed Inference with Tensor Parallelism

This document explains how to use DeepSpeed inference with tensor parallelism to solve OOM issues and speed up inference for test and testgen modes.

## Overview

DeepSpeed inference with tensor parallelism distributes the model across multiple GPUs, reducing memory usage per GPU and enabling faster inference through parallel computation.

### Benefits:
- **Reduced Memory Usage**: Model weights are split across GPUs
- **Faster Inference**: Parallel computation across multiple GPUs
- **Better Throughput**: Process more samples without OOM errors

## Installation

Ensure DeepSpeed is installed:
```bash
pip install deepspeed>=0.15.4
```

## Configuration

The DeepSpeed inference configuration is located at `configs/deepspeed_inference.json`:

```json
{
  "tensor_parallel": {
    "tp_size": 2
  },
  "dtype": "bf16",
  "replace_with_kernel_inject": true,
  "enable_cuda_graph": false,
  "triangular_masking": true,
  "return_tuple": true,
  "training_mp_size": 1,
  "max_out_tokens": 512,
  "min_out_tokens": 1
}
```

### Key Configuration Options:

- **tp_size**: Number of GPUs for tensor parallelism (default: 2)
- **dtype**: Data type for inference ("bf16" or "fp16")
- **replace_with_kernel_inject**: Use DeepSpeed optimized kernels
- **enable_cuda_graph**: Enable CUDA graphs for performance (may not work with all models)

## Usage

### Test Mode

Run inference with DeepSpeed tensor parallelism:

```bash
python main.py \
  --mode test \
  --model_weight_path ./results/model_weights.pt \
  --use_deepspeed_inference \
  --tensor_parallel_size 2 \
  --deepspeed_inference_config configs/deepspeed_inference.json \
  --batch_size 4 \
  --max_seq_length 8192 \
  --dtype bf16
```

### TestGen Mode

Run test case generation with DeepSpeed:

```bash
python main.py \
  --mode testgen \
  --model_weight_path ./results/model_weights.pt \
  --use_deepspeed_inference \
  --tensor_parallel_size 2 \
  --deepspeed_inference_config configs/deepspeed_inference.json \
  --do_generate \
  --batch_size 4 \
  --max_seq_length 8192 \
  --dtype bf16
```

### Command-Line Arguments

- `--use_deepspeed_inference`: Enable DeepSpeed inference (flag)
- `--tensor_parallel_size`: Number of GPUs for tensor parallelism (default: 2)
- `--deepspeed_inference_config`: Path to DeepSpeed config file (default: `configs/deepspeed_inference.json`)
- `--max_seq_length`: Max sequence length for input truncation (default: 8192)

## Important Notes

### 1. Max Sequence Length
The code now properly handles max sequence length to prevent "sequence longer than maximum" errors. Always set `--max_seq_length` to your model's context window size or less (e.g., 8192).

### 2. Number of GPUs
- The `--tensor_parallel_size` must match the number of available GPUs
- For 2 GPUs: `--tensor_parallel_size 2`
- For 4 GPUs: `--tensor_parallel_size 4`
- The model will be split evenly across all specified GPUs

### 3. Memory Savings
Example memory usage with a 7B parameter model:
- **Without DeepSpeed**: ~28GB per GPU (won't fit on most GPUs)
- **With tp_size=2**: ~14GB per GPU
- **With tp_size=4**: ~7GB per GPU

### 4. Batch Size
With tensor parallelism, you can often increase batch size:
- Without DeepSpeed: `--batch_size 1` (to avoid OOM)
- With DeepSpeed: `--batch_size 4` or higher

## Troubleshooting

### OOM Errors Still Occurring

1. **Increase tensor_parallel_size**: Use more GPUs
   ```bash
   --tensor_parallel_size 4
   ```

2. **Reduce batch size**:
   ```bash
   --batch_size 1
   ```

3. **Reduce max_seq_length**:
   ```bash
   --max_seq_length 4096
   ```

### DeepSpeed Initialization Fails

If DeepSpeed initialization fails, the code will automatically fall back to standard inference. Check:
1. All GPUs are available: `nvidia-smi`
2. DeepSpeed is properly installed: `pip show deepspeed`
3. Config file exists and is valid JSON

### Slower Than Expected

1. **Enable CUDA graphs** (if compatible with your model):
   ```json
   "enable_cuda_graph": true
   ```

2. **Adjust batch size**: Sometimes larger batch sizes improve throughput
   ```bash
   --batch_size 8
   ```

3. **Check GPU utilization**: Use `nvidia-smi` to ensure all GPUs are being used

## Performance Tips

1. **Use bf16 when possible**: Better numerical stability than fp16
   ```bash
   --dtype bf16
   ```

2. **Optimize batch size**: Experiment with different batch sizes
   ```bash
   --batch_size 4  # or 8, 16, etc.
   ```

3. **Profile your runs**: Monitor GPU memory and utilization
   ```bash
   watch -n 1 nvidia-smi
   ```

## Example Commands

### Single GPU (No DeepSpeed)
```bash
python main.py --mode test \
  --model_weight_path ./results/checkpoint.pt \
  --batch_size 1 \
  --max_seq_length 4096
```

### 2 GPUs with DeepSpeed
```bash
python main.py --mode test \
  --model_weight_path ./results/checkpoint.pt \
  --use_deepspeed_inference \
  --tensor_parallel_size 2 \
  --batch_size 4 \
  --max_seq_length 8192
```

### 4 GPUs with DeepSpeed
```bash
python main.py --mode testgen \
  --model_weight_path ./results/checkpoint.pt \
  --use_deepspeed_inference \
  --tensor_parallel_size 4 \
  --batch_size 8 \
  --max_seq_length 8192 \
  --do_generate
```

## Additional Resources

- [DeepSpeed Inference Documentation](https://www.deepspeed.ai/inference/)
- [Tensor Parallelism Guide](https://www.deepspeed.ai/tutorials/inference-tutorial/)
- [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed)
