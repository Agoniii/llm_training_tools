This is a tool to estimate the memory of different parallel settings without running.

## Features
- Support TP/PP/CP/SP
- Support BF16/FP8/FP8_init
- Support optimizer dtype and gradient dtype are set to FP32/FP16
- Support selective checkpointing or use flash attention.
- Support dense model only, moe model and multimodal is under development.

## How to use
Pls refer to `test_mem_llama7B.sh`

The training command is very similar to the LLM training. This 

And we add several params:
 - `--data-parallel-size`: Degree of data parallelism. As we just use one process to estimate the memory, so we need to manually specify data parallel size. And the total world_size = tensor_model_parallel_size * pipeline_model_parallel_size * data_parallel_size * context_parallel_size.
 - `--optimizer-dtype`: Dtype of optimizer states. Support BF16 and FP32. Default is FP32.
 - `--gradient-dtype`: Dtype of gradient. Support BF16 and FP32. Default is FP32.
 - `--fp8-init`: Use FP8 for model init, it can reduce memory usage.

Note:
- Need to set `--vocab-size`