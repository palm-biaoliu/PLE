import argparse
import os
from datetime import timedelta

import jsonlines
import torch
from torch import distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer

from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.utils import get_processor, get_strategy, get_tokenizer


# ==================== 新增：计算 Log-Probs 的函数 ====================
def batch_log_probs(args):
    """
    计算数据集里每个回答在给定模型下的总 log-probability。
    """
    strategy = get_strategy(args)
    strategy.setup_distributed(timeout=timedelta(minutes=180))

    # 加载 Actor 模型（Policy 或 Ref）
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.attn_implementation,
        bf16=args.bf16,
    )

    # 配置 Tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)

    # 准备模型
    model = strategy.prepare(model)
    model.eval()

    # 加载待评估的数据集（通常是包含 input 和 output 的生成数据）
    dataset = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
    )
    dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    # 使用 SFTDataset 来处理 prompt + output 的拼接
    sft_dataset = SFTDataset(
        dataset, tokenizer, args.max_len, strategy, pretrain_mode=False, input_template=args.input_template
    )
    dataloader = strategy.setup_dataloader(
        sft_dataset, args.micro_batch_size, True, False, sft_dataset.collate_fn, drop_last=False
    )
    
    pbar = tqdm(
        dataloader,
        desc=f"Calculating Log-Probs ({args.label})",
        disable=not strategy.is_rank_0(),
    )

    dist.barrier()

    output_dataset = []
    with torch.no_grad():
        for input_ids, attention_masks, _, info in pbar:
            input_ids = input_ids.squeeze(1).to(torch.cuda.current_device())
            attention_masks = attention_masks.squeeze(1).to(torch.cuda.current_device())
            
            # 前向传播获取 logits
            # OpenRLHF 的 Actor 模型在 return_logprobs=True 时会返回经过 log_softmax 的值
            output = model(input_ids, attention_mask=attention_masks, return_output=True)
            logits = output.logits

            # 计算 log_probs: (batch, seq_len, vocab_size)
            log_probs = F.log_softmax(logits, dim=-1)

            # 获取对应的 label (input_ids 右移一位)
            labels = input_ids[:, 1:].contiguous()
            log_probs = log_probs[:, :-1, :]

            # 提取每个 token 的 log_prob
            per_token_logps = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)

            # 掩码逻辑：只计算 Response 部分，排除 Prompt 和 Padding
            # SFTDataset 的 info 里包含每个 sample 的 prompt 长度
            # 注意：OpenRLHF 的 SFTDataset 可能通过不同的 key 传递长度，这里需要对应
            for i in range(input_ids.size(0)):
                # 寻找 prompt 结束的位置（通常是第一个非 mask 的位置 + prompt 长度，或者基于 label 掩码）
                # 这里我们利用 attention_mask 和 info 中的 prompt 长度进行过滤
                # 假设 SFTDataset 在 info 里存了 prompt 的 token 数
                prompt_len = info["prompt_len"][i] if "prompt_len" in info else 0
                
                mask = attention_masks[i, 1:].clone()
                mask[:prompt_len-1] = False # 掩盖掉 prompt 部分
                
                # 求和得到该回答的总 log_prob
                total_logp = (per_token_logps[i] * mask).sum().item()
                
                # 组装数据，添加 logp_{label} 字段
                res_obj = {
                    "input": info["input"][i],
                    "output": info["output"][i],
                }
                # 如果输入原本就有 reward 等字段，保留它们
                for key in info:
                    if key not in ["input", "output", "prompt_len"]:
                        res_obj[key] = info[key][i]
                
                # 动态添加 label (如 logp_policy 或 logp_ref)
                res_obj[f"logp_{args.label}"] = total_logp
                output_dataset.append(res_obj)

    # 结果收集与保存逻辑 (参考 batch_rm_inference)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with jsonlines.open(args.output_path + str(strategy.get_rank()), mode="w") as writer:
        writer.write_all(output_dataset)

    dist.barrier()

    if strategy.is_rank_0():
        all_data = []
        world_size = dist.get_world_size()
        for rank in range(world_size):
            file = args.output_path + str(rank)
            with jsonlines.open(file, mode="r") as reader:
                for obj in reader:
                    all_data.append(obj)
            os.remove(file)

        with jsonlines.open(args.output_path, mode="w") as writer:
            writer.write_all(all_data)
# =====================================================================

def batch_generate_vllm(args):
    from vllm import LLM, SamplingParams

    # configure strategy
    class Empty:
        pass

    dummy_strategy = Empty()
    dummy_strategy.print = print
    dummy_strategy.is_rank_0 = lambda: True
    dummy_strategy.args = args

    # configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain, trust_remote_code=True)

    # configure model
    llm = LLM(
        model=args.pretrain,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        seed=args.seed,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
    )

    # Create a sampling params object.
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        skip_special_tokens=False,
        truncate_prompt_tokens=args.prompt_max_len,
        include_stop_str_in_output=True,
    )

    prompts_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        dummy_strategy,
        args.seed,
        max_count=args.max_samples,
    )
    if args.iter is None:
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    else:
        # for iterative generation
        start_idx = args.iter * args.rollout_batch_size
        end_idx = start_idx + args.rollout_batch_size
        prompts_data = prompts_data.select(range(start_idx, min(end_idx, len(prompts_data))))

    prompts_dataset = PromptDataset(prompts_data, tokenizer, dummy_strategy, input_template=args.input_template)
    prompts = []
    for _, prompt, _ in prompts_dataset:
        prompts.append(prompt)

    # Conditional SFT inference
    if args.enable_csft:
        for i in range(len(prompts)):
            prompts[i] += args.csft_prompt.strip() + " "

    # best of n
    N = args.best_of_n
    output_dataset = []

    outputs = llm.generate(prompts * N, sampling_params)
    for output in outputs:
        prompt = output.prompt
        output = output.outputs[0].text
        output_dataset.append({"input": prompt, "output": output})

    with jsonlines.open(args.output_path, mode="w") as writer:
        writer.write_all(output_dataset)


def batch_generate(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed(timeout=timedelta(minutes=720))

    # configure model
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    # prepare models
    model = strategy.prepare(model)
    model.eval()

    # tokenizer
    def tokenize_fn(texts):
        batch = tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=args.prompt_max_len,
            padding=True,
            truncation=True,
        )
        return {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}

    prompts_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
    )
    if args.iter is None:
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    else:
        # for iterative generation
        start_idx = args.iter * args.rollout_batch_size
        end_idx = start_idx + args.rollout_batch_size
        prompts_data = prompts_data.select(range(start_idx, min(end_idx, len(prompts_data))))

    prompts_dataset = PromptDataset(prompts_data, tokenizer, strategy, input_template=args.input_template)
    prompts_dataloader = strategy.setup_dataloader(
        prompts_dataset, args.micro_batch_size, True, False, drop_last=False
    )
    pbar = tqdm(
        prompts_dataloader,
        desc="Generating",
        disable=not strategy.is_rank_0(),
    )

    dist.barrier()
    N = args.best_of_n
    output_dataset = []

    for _, prompts, _ in pbar:
        # Conditional SFT inference
        if args.enable_csft:
            for i in range(len(prompts)):
                prompts[i] += args.csft_prompt.strip() + " "

        inputs = tokenize_fn(prompts)
        for _ in range(N):
            outputs = model.model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
                do_sample=not args.greedy_sampling,
                top_p=args.top_p,
                early_stopping=False,
                num_beams=1,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for prompt, output in zip(prompts, outputs):
                output = output[len(prompt) :]
                output_dataset.append({"input": prompt, "output": output})

        dist.barrier()

    with jsonlines.open(args.output_path + str(strategy.get_rank()), mode="w") as writer:
        writer.write_all(output_dataset)

    # wait unitl all processes generate done
    dist.barrier()

    # concate multiple output files in rank 0
    if strategy.is_rank_0():
        output_dataset = []
        world_size = dist.get_world_size()
        files = [args.output_path + str(rank) for rank in range(world_size)]
        for file in files:
            with jsonlines.open(file, mode="r") as reader:
                for obj in reader:
                    output_dataset.append(obj)
            os.remove(file)

        with jsonlines.open(args.output_path, mode="w") as writer:
            writer.write_all(output_dataset)


def batch_rm_inference(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed(timeout=timedelta(minutes=180))

    # configure model
    # load huggingface model/config
    model = get_llm_for_sequence_regression(
        args.pretrain,
        "reward",
        normalize_reward=True,
        attn_implementation=args.attn_implementation,
        bf16=args.bf16,
        value_head_prefix=args.value_head_prefix,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    # prepare models
    model = strategy.prepare(model)
    model.eval()

    dataset = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
    )
    dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = SFTDataset(
        dataset, tokenizer, args.max_len, strategy, pretrain_mode=False, input_template=args.input_template
    )
    dataloader = strategy.setup_dataloader(
        dataset, args.micro_batch_size, True, False, dataset.collate_fn, drop_last=False
    )
    pbar = tqdm(
        dataloader,
        desc="Rewarding",
        disable=not strategy.is_rank_0(),
    )

    dist.barrier()

    output_dataset = []
    with torch.no_grad():
        for input_ids, attention_masks, _, info in pbar:
            input_ids = input_ids.squeeze(1).to(torch.cuda.current_device())
            attention_masks = attention_masks.squeeze(1).to(torch.cuda.current_device())
            rewards = model(input_ids, attention_masks)
            for item, reward in zip(info, rewards):
                output_dataset.append({"input": item["input"], "output": item["output"], "reward": reward.item()})
            # for prompt, output, reward in zip(info["input"], info["output"], rewards):
            #     output_dataset.append({"input": prompt, "output": output, "reward": reward.item()})

            dist.barrier()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with jsonlines.open(args.output_path + str(strategy.get_rank()), mode="w") as writer:
        writer.write_all(output_dataset)

    # wait unitl all processes generate done
    dist.barrier()

    # concate multiple output files in rank 0
    if strategy.is_rank_0():
        output_dataset = []
        world_size = dist.get_world_size()
        files = [args.output_path + str(rank) for rank in range(world_size)]
        for file in files:
            with jsonlines.open(file, mode="r") as reader:
                for obj in reader:
                    output_dataset.append(obj)
            # os.remove(file)

        rewards = torch.tensor([obj["reward"] for obj in output_dataset])
        print(f"Reward mean: {rewards.mean().item()}, std: {rewards.std().item()}")

        if args.post_processor and args.post_processor != "null":
            strategy.print(f"Use Processor {args.post_processor}, Reward Norm {args.normalize_reward}")
            processor = get_processor(args.post_processor)
            output_dataset = processor(args, output_dataset)

        with jsonlines.open(args.output_path, mode="w") as writer:
            writer.write_all(output_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_task", type=str, default=None, help="Set to generate_vllm, generate (HF generate) or rm"
    )
    parser.add_argument("--zero_stage", type=int, default=0, help="DeepSpeed ZeRO Stage")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed cli")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16 for deepspeed")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation (e.g., eager, flash_attention_2, flash_attention_3, kernels-community/vllm-flash-attn3)",
    )
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--full_determinism",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )

    # Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF pretrain model name or path")
    parser.add_argument(
        "--value_head_prefix", type=str, default="value_head", help="value_head prefix for Reward Model"
    )

    # Custom dataset
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default=None)
    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default="output", help="JSON dataset key")
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="HF tokenizer apply_chat_template"
    )
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--output_path", type=str, default=None, help="Output JSON data path")

    # For generation
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for prompt")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens in generation")
    parser.add_argument("--greedy_sampling", action="store_true", default=False, help="Use Greedy sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p for Sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for Sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--best_of_n", type=int, default=1, help="Number of responses to generate per prompt")
    parser.add_argument(
        "--post_processor",
        type=str,
        default=None,
        help="set to rs (Rejection Sampling), csft (Conditional SFT), iter_dpo (Iterative DPO) or None",
    )
    # For vllm
    parser.add_argument("--tp_size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)

    parser.add_argument(
        "--top_reward_diff_percentile",
        type=float,
        default=1.0,
        help="Percentile for selecting top reward differences.",
    )

    parser.add_argument(
        "--sigmoid_temperature",
        type=float,
        default=1.0,
        help="Temperature for sigmoid weighting.",
    )

    # For Iterative generation and Rejection Sampling
    parser.add_argument(
        "--iter",
        type=int,
        default=None,
        help="Used to slice the datasets in range iter * rollout_batch_size: (iter + 1) * rollout_batch_size",
    )
    parser.add_argument("--rollout_batch_size", type=int, default=2048, help="Number of samples to generate")

    # ==================== START OF MODIFICATION ====================
    # 为迭代式加权DPO添加的参数
    parser.add_argument(
        "--selection_percentage",
        type=float,
        default=1.0, # 默认为1.0，即所有数据权重都为1，保持原始行为
        help="Percentage of top reward-margin data to use with full loss weight (e.g., 0.1 for 10%)."
    )
    parser.add_argument(
        "--low_loss_weight",
        type=float,
        default=0.1, # 剩余数据的权重
        help="Loss weight for the remaining data (not in the top percentage)."
    )
    # ===================== END OF MODIFICATION =====================

    # ==================== 新增参数 ====================
    parser.add_argument(
        "--label",
        type=str,
        default="policy",
        help="Label for log-probs (e.g., 'policy' or 'ref'). Will save as logp_{label}"
    )
    # =================================================



    # For Conditional SFT
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normalization")
    parser.add_argument("--reward_template", type=str, default=None)
    parser.add_argument("--enable_csft", action="store_true", default=False)
    parser.add_argument("--csft_prompt", type=str, default="<rm_score>: 5.00", help="Conditional SFT prompt")

    parser.add_argument(
    "--low_priority_weight",
    type=float,
    default=0.1,  # 默认的低优先级权重值
    help="用于 weighted_reward_diff_processor, 为非 top 百分位的数据设置损失权重。",
)


    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()

    # ==================== 修改分发逻辑 ====================
    if args.eval_task == "generate":
        batch_generate(args)
    elif args.eval_task == "generate_vllm":
        batch_generate_vllm(args)
    elif args.eval_task == "rm":
        batch_rm_inference(args)
    elif args.eval_task == "log_probs": # 新增分发
        batch_log_probs(args)
    else:
        print("Invalid eval_task.")
    # =====================================================


    # if args.eval_task and args.eval_task == "generate":
    #     batch_generate(args)
    # if args.eval_task and args.eval_task == "generate_vllm":
    #     batch_generate_vllm(args)
    # elif args.eval_task and args.eval_task == "rm":
    #     batch_rm_inference(args)
    # else:
    #     print("Invalid or missing '--eval_task' argument. Please specify either 'generate' or 'rm'.")

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()
