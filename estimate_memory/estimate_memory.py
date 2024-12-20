# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Computes theoretical memory footprint for model training."""

import math

from megatron.training import get_args
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import load_args_from_checkpoint
from megatron.training.global_vars import set_global_variables
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding


NUM_BYTES_IN_MEGABYTE = 1024 * 1024


def compute_weight_and_optimizer_memory(args, verbose=False):
    if not args.group_query_attention:
        args.num_query_groups = args.num_attention_heads
    kv_hidden_size = args.hidden_size / args.num_attention_heads * args.num_query_groups
    
    num_parameters_word_embedding = args.hidden_size * args.padded_vocab_size / args.tensor_model_parallel_size
    if verbose:
        print(
            f"Number of parameters in word_embedding in billions: {num_parameters_word_embedding}, args.padded_vocab_size: {args.padded_vocab_size}, args.hidden_size: {args.hidden_size}, kv_hidden_size {kv_hidden_size}"
        )
    
    if args.position_embedding_type == "rope":
        num_parameters_position_embedding = 0
    else:
        num_parameters_position_embedding = args.hidden_size * args.seq_length
    
    # attention: 
    # layernorm: 2h
    num_parameters_attention = 2 * args.hidden_size
    # QKV weight: 3h*h/tp, bias: 3h/tp
    # output linear weight: h*h/tp, bias: h
    num_parameters_attention += ((args.hidden_size + 2 * kv_hidden_size) * args.hidden_size + args.hidden_size * args.hidden_size) / args.tensor_model_parallel_size
    if args.add_bias_linear:
        num_parameters_attention += (args.hidden_size + 2 * kv_hidden_size) / args.tensor_model_parallel_size + args.hidden_size
    elif args.add_qkv_bias:
        num_parameters_attention += (args.hidden_size + 2 * kv_hidden_size) / args.tensor_model_parallel_size
    
    # MLP: 
    # layernorm: 2h
    num_parameters_mlp = 2 * args.hidden_size
    # mlp1 weight: h*ffn/tp, bias: ffn/tp
    # mlp2 weight: ffn*h/tp, bias: h
    if args.swiglu:
        num_parameters_mlp += args.hidden_size * args.ffn_hidden_size * 3 / args.tensor_model_parallel_size
        if args.add_bias_linear:
            num_parameters_mlp += args.ffn_hidden_size * 2 / args.tensor_model_parallel_size + args.hidden_size
    else:
        num_parameters_mlp += args.hidden_size * args.ffn_hidden_size * 2 / args.tensor_model_parallel_size
        if args.add_bias_linear:
            num_parameters_mlp += args.ffn_hidden_size / args.tensor_model_parallel_size + args.hidden_size
    
    num_parameters_in_single_layer = num_parameters_attention + num_parameters_mlp
    if verbose:
        print(
            f"Number of parameters in attention: {num_parameters_attention}"
        )
        print(
            f"Number of parameters in MLP: {num_parameters_mlp}"
        )
        print(
            f"Number of parameters in single layer: {num_parameters_in_single_layer}"
        )
    num_parameters_in_total_layers = num_parameters_in_single_layer * args.num_layers / args.pipeline_model_parallel_size
    
    num_parameters_output_layernorm = 2 * args.hidden_size
    num_parameters_output_embedding = num_parameters_word_embedding
    
    if args.pipeline_model_parallel_size == 1:
        num_parameters_total = (
            num_parameters_word_embedding
            + num_parameters_position_embedding
            + num_parameters_in_total_layers
            + num_parameters_output_layernorm
        )
        if args.untie_embeddings_and_output_weights:
            num_parameters_total += num_parameters_output_embedding
        if verbose:
            print(
                f"Number of parameters in total layers in billions: {num_parameters_total / 10**9:.2f}"
            )
    
    else:
        num_parameters_first_stage = (
            num_parameters_word_embedding
            + num_parameters_position_embedding
            + num_parameters_in_total_layers
        )
        num_parameters_last_stage = (
            num_parameters_in_total_layers
            + num_parameters_output_layernorm
            + num_parameters_output_embedding
        )
        num_parameters_other_stage = num_parameters_in_total_layers
        if verbose:
            print(
                f"Number of parameters in first stage in billions: {num_parameters_first_stage}"
            )
            print(
                f"Number of parameters in last stage in billions: {num_parameters_last_stage / 10**9:.2f}"
            )
            print(
                f"Number of parameters in other stages in billions: {num_parameters_other_stage / 10**9:.2f}"
            )
        num_parameters_total = num_parameters_first_stage
    
    if args.optimizer_dtype == "fp32":
        opt_size = 4
    elif args.optimizer_dtype == "bf16" or args.optimizer_dtype == "fp16":
        opt_size = 2

    if args.gradient_dtype == "fp32":
        grad_size = 4
    elif args.gradient_dtype == "bf16" or args.gradient_dtype == "fp16":
        grad_size = 2

    if args.fp16 or args.bf16 and args.fp8 is not None and not args.fp8_init:
        num_bytes_per_parameter = (
            4 + grad_size + (4 + opt_size * 2)
            if not args.use_distributed_optimizer
            else 4 + grad_size + ((4 + opt_size * 2) / args.data_parallel_size / args.context_parallel_size)
        )
    elif args.fp16 or args.bf16:
        num_bytes_per_parameter = (
            2 + grad_size + (4 + opt_size * 2)
            if not args.use_distributed_optimizer
            else 2 + grad_size + ((4 + opt_size * 2) / args.data_parallel_size / args.context_parallel_size)
        )
    else:
        raise ValueError("only support bf16/fp16/fp8 training")

    weight_and_optimizer_memory = (
        num_parameters_total * num_bytes_per_parameter
    )

    return weight_and_optimizer_memory
    

def compute_activation_memory(args, num_microbatches, verbose=False, debug=False):
    # Using formula in Table 2 of https://arxiv.org/pdf/2205.05198.pdf.
    # We are trying to compute the maximum activation footprint, so all calculations in this function
    # are for the first pipeline stage.
    if not args.group_query_attention:
        args.num_query_groups = args.num_attention_heads
    kv_hidden_size = args.hidden_size / args.num_attention_heads * args.num_query_groups

    if args.fp16 or args.bf16 and args.fp8 is None:
        activation_dtype = 2
    elif args.fp16 or args.bf16 and args.fp8 is not None:
        activation_dtype = 1

    # Memory footprint from transformer layer (self-attention and MLP).
    # Attention
    # LN 2bsq
    activation_mem_attn_ln = args.seq_length * args.micro_batch_size * args.hidden_size * 2
    if not args.sequence_parallel:
        activation_mem_attn_ln *= args.tensor_model_parallel_size
    # qkv 2bsh/1bsh
    activation_mem_attn_qkv = args.seq_length * args.micro_batch_size * args.hidden_size * activation_dtype
    if not args.sequence_parallel:
        activation_mem_attn_qkv *= args.tensor_model_parallel_size
    # attention q 2bsh
    activation_mem_attn_q = args.seq_length * args.micro_batch_size * args.hidden_size * 2
    # attention k and v 4bsh
    activation_mem_attn_kv = args.seq_length * args.micro_batch_size * kv_hidden_size * 2 * 2
    # attention proj 2bsh/1bsh
    activation_mem_attn_proj = args.seq_length * args.micro_batch_size * args.hidden_size * activation_dtype
    # dropout bsh
    if args.attention_dropout > 0:
        activation_mem_attn_dropout = args.seq_length * args.micro_batch_size * args.hidden_size
        if not args.sequence_parallel:
            activation_mem_attn_dropout *= args.tensor_model_parallel_size
    else:
        activation_mem_attn_dropout = 0
    # bf16: 2+2+2+4+2+1=13bsh
    # fp8: 2+1+2+4+1+1=11bsh
    activation_memory_attn = (
        activation_mem_attn_ln
        + activation_mem_attn_qkv
        + activation_mem_attn_q
        + activation_mem_attn_kv
        + activation_mem_attn_proj
        + activation_mem_attn_dropout
    )
    if debug:
        print("activation_mem_attn_ln", activation_mem_attn_ln / args.tensor_model_parallel_size)
        print("activation_mem_attn_qkv", activation_mem_attn_qkv / args.tensor_model_parallel_size)
        print("activation_mem_attn_q", activation_mem_attn_q / args.tensor_model_parallel_size)
        print("activation_mem_attn_kv", activation_mem_attn_kv / args.tensor_model_parallel_size)
        print("activation_mem_attn_proj", activation_mem_attn_proj / args.tensor_model_parallel_size)
        print("activation_mem_attn_dropout", activation_mem_attn_dropout / args.tensor_model_parallel_size)
    
    # MLP
    # LN 2bsh
    activation_mem_mlp_ln = args.seq_length * args.micro_batch_size * args.hidden_size * 2
    if not args.sequence_parallel:
        activation_mem_mlp_ln *= args.tensor_model_parallel_size
    # FC1 2bsh/1bsh
    activation_mem_mlp_fc1 = args.seq_length * args.micro_batch_size * args.hidden_size * activation_dtype
    if not args.sequence_parallel:
        activation_mem_mlp_fc1 *= args.tensor_model_parallel_size
    # Act 8bsh
    if args.swiglu:
        activation_mem_mlp_act = args.seq_length * args.micro_batch_size * args.ffn_hidden_size * 2 * 2
    else:
        activation_mem_mlp_act = args.seq_length * args.micro_batch_size * args.ffn_hidden_size * 2
    # FC2 8bsh/4bsh
    activation_mem_mlp_fc2 = args.seq_length * args.micro_batch_size * args.ffn_hidden_size * activation_dtype
    # dropout bsh
    if args.hidden_dropout > 0:
        activation_mem_mlp_dropout = args.seq_length * args.micro_batch_size * args.hidden_size
        if not args.sequence_parallel:
            activation_mem_mlp_dropout *= args.tensor_model_parallel_size
    else:
        activation_mem_mlp_dropout = 0
    # bf16: 2+2+8+8+1=21
    # fp8: 2+1+8+4+1=16
    activation_memory_mlp = (
        activation_mem_mlp_ln
        + activation_mem_mlp_fc1
        + activation_mem_mlp_act
        + activation_mem_mlp_fc2
        + activation_mem_mlp_dropout
    )
    if debug:
        print("activation_mem_mlp_ln", activation_mem_mlp_ln / args.tensor_model_parallel_size)
        print("activation_mem_mlp_fc1", activation_mem_mlp_fc1 / args.tensor_model_parallel_size)
        print("activation_mem_mlp_act", activation_mem_mlp_act / args.tensor_model_parallel_size)
        print("activation_mem_mlp_fc2", activation_mem_mlp_fc2 / args.tensor_model_parallel_size)
        print("activation_mem_mlp_dropout", activation_mem_mlp_dropout / args.tensor_model_parallel_size)

    activation_memory = activation_memory_attn + activation_memory_mlp

    if verbose:
        print(
            f"Activation memory footprint per transformer layer: "
            f"{activation_memory / NUM_BYTES_IN_MEGABYTE / args.tensor_model_parallel_size:.1f} MB"
        )
    activation_memory *= args.num_layers

    # Now add activation memory required for input embeddings, last LayerNorm and output layer.

    # Input to embedding (pp_size microbatches in flight).
    activation_memory += (
        8 * args.seq_length * args.micro_batch_size * args.pipeline_model_parallel_size
    )
    # Dropout in embedding layer (pp_size microbatches in flight).
    activation_memory += (
        args.seq_length
        * args.micro_batch_size
        * args.hidden_size
        * args.pipeline_model_parallel_size
    )

    # Multiply by interleaved PP memory factor.
    if args.virtual_pipeline_model_parallel_size is not None:
        interleaved_schedule_memory_penalty = 1 + (
            (args.pipeline_model_parallel_size - 1)
            / (args.pipeline_model_parallel_size * args.virtual_pipeline_model_parallel_size)
        )
        in_flight_microbatches = math.ceil(
            interleaved_schedule_memory_penalty * args.pipeline_model_parallel_size
        )
        if verbose:
            print(
                f"Memory penalty from interleaved schedule: {interleaved_schedule_memory_penalty:.2f}"
            )
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")
        activation_memory *= interleaved_schedule_memory_penalty

    # If using non-interleaved schedule, number of microbatches in pipeline can be less than pp_size,
    # so discount accordingly.
    if args.virtual_pipeline_model_parallel_size is None and args.pipeline_model_parallel_size > 1:
        if num_microbatches is not None:
            activation_memory *= min(1, num_microbatches / args.pipeline_model_parallel_size)
            in_flight_microbatches = min(num_microbatches, args.pipeline_model_parallel_size)
        else:
            in_flight_microbatches = args.pipeline_model_parallel_size
        if verbose:
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")

    if args.pipeline_model_parallel_size == 1:
        # Inputs to output layer
        activation_memory += args.seq_length * args.micro_batch_size * args.hidden_size * 2
        # CE loss(bf16+fp32*2)
        activation_memory += args.seq_length * args.micro_batch_size * args.padded_vocab_size * (2 + 4 + 4)

    # sendrecv memory
    if args.pipeline_model_parallel_size > 1:
        activation_memory += args.seq_length * args.micro_batch_size * args.padded_vocab_size * 2

    # Activation memory is partitioned by TP size due to tensor and sequence model parallelism.
    return activation_memory / args.tensor_model_parallel_size / args.context_parallel_size


def report_theoretical_memory(args, num_microbatches=None, verbose=False):
    # Formulae here assume sequence parallelism and selective activation recomputation.
    if args.recompute_granularity != 'selective' or args.use_flash_attn:
        print("Theoretical memory estimate only supported with recompute_granularity=selective or use_falsh_attn")
        return

    weight_and_optimizer_memory = (
        compute_weight_and_optimizer_memory(args, verbose=verbose) / NUM_BYTES_IN_MEGABYTE
    )
    activation_memory = (
        compute_activation_memory(args, num_microbatches=num_microbatches, verbose=verbose)
        / NUM_BYTES_IN_MEGABYTE
    )
    total_memory = weight_and_optimizer_memory + activation_memory

    print(
        f"tensor_parallel_size: {args.tensor_model_parallel_size}, pipeline_parallel_size: {args.pipeline_model_parallel_size}, data_parallel_size: {args.data_parallel_size}, context_parallel_size: {args.context_parallel_size}\n"
        f"Theoretical memory footprints: weight and optimizer={weight_and_optimizer_memory/1024:.2f} GB, "
        f"activation={activation_memory/1024:.2f} GB, "
        f"total={total_memory/1024:.2f} GB\n"
    )


def add_network_size_args(parser):
    group = parser.add_argument_group(title='data parallel size')
    group.add_argument('--data-parallel-size', type=int, default=1,
                       help='Degree of data parallelism.')
    group.add_argument('--optimizer-dtype', type=str, default="fp32",
                       help='Dtype of optimizer states.')
    group.add_argument('--gradient-dtype', type=str, default="fp32",
                       help='Dtype of gradient.')
    group.add_argument('--fp8-init', action='store_true', default=False)
    return parser


if __name__ == "__main__":
    # Parse arguments

    args = parse_args(extra_args_provider=add_network_size_args, ignore_unknown_args=False)
    args_defaults = {}

    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
        assert args.load is not None, "--use-checkpoints-args requires --load argument"
        load_args_from_checkpoint(args)

    args.world_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.data_parallel_size * args.context_parallel_size

    validate_args(args, args_defaults)
    if args.optimizer_dtype not in ["fp32", "bf16", "fp16"]:
        raise ValueError("unsupported optimizer_dtype ", args.optimizer_dtype)
    if args.gradient_dtype not in ["fp32", "bf16", "fp16"]:
        raise ValueError("unsupported gradient_dtype ", args.gradient_dtype)

    # set global args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(args, build_tokenizer=False)
    args.padded_vocab_size = _vocab_size_with_padding(orig_vocab_size=args.vocab_size, args=args)
    report_theoretical_memory(args, verbose=False)
