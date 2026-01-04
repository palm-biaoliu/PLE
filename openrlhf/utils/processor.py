import torch
from tqdm import tqdm
import numpy as np
import math
def reward_normalization(objs):
    rewards = [float(obj["reward"]) for obj in objs]
    rewards = torch.tensor(rewards, dtype=torch.float64)
    rewards = (rewards - rewards.mean()) / rewards.std()
    for i, obj in enumerate(objs):
        obj["reward"] = rewards[i].item()


# Conditional SFT
# See https://arxiv.org/abs/2308.12050
DEFAULT_REWARD_PROMPT = "{input} <rm_score>: {reward} "


def conditional_sft_processor(args, objs):
    if "reward_template" not in args or args.reward_template is None:
        reward_template = DEFAULT_REWARD_PROMPT
    else:
        reward_template = args.reward_template
    assert "{input}" in reward_template
    assert "{reward}" in reward_template

    if args.normalize_reward:
        reward_normalization(objs)

    for obj in tqdm(objs, desc="Conditional SFT process..."):
        input = obj["input"]
        reward = "{:.2f}".format(float(obj["reward"]))
        input = reward_template.replace("{reward}", reward).replace("{input}", input)
        obj["input"] = input

    return objs


# Rejection Sampling
# See https://arxiv.org/abs/2307.09288
def rejection_sampling_processor(args, objs):
    out = {}
    for obj in tqdm(objs, desc="Rejection Sampling process...."):
        input = obj["input"]
        output = obj["output"]
        reward = float(obj["reward"])

        if input not in out:
            out[input] = {"output": output, "reward": reward}
        elif reward > out[input]["reward"]:
            out[input]["reward"] = reward
            out[input]["output"] = output

    return [{"input": k, "output": v["output"], "reward": v["reward"]} for k, v in out.items()]


# Iterative DPO
# See https://github.com/RLHFlow/Online-RLHF/blob/main/run_loop.sh
def iterative_dpo_processor(args, objs):
    out = {}
    for obj in tqdm(objs, desc="Iterative DPO process...."):
        input = obj["input"]
        output = obj["output"]
        reward = float(obj["reward"])

        if input not in out:
            out[input] = {
                "output": output,
                "chosen": output,
                "chosen_reward": reward,
                "rejected": output,
                "rejected_reward": reward,
            }
        elif reward > out[input]["chosen_reward"]:
            out[input]["chosen_reward"] = reward
            out[input]["chosen"] = output
        elif reward < out[input]["rejected_reward"]:
            out[input]["rejected_reward"] = reward
            out[input]["rejected"] = output

    return [
        {
            "prompt": k,
            "chosen": v["chosen"],
            "chosen_reward": v["chosen_reward"],
            "rejected": v["rejected"],
            "rejected_reward": v["rejected_reward"],
        }
        for k, v in out.items()
    ]

def reward_diff_weighting_processor(args, output_dataset):
    """
    根据奖励差值的百分位对数据进行加权，而不是过滤。
    - 奖励差值在前 N% 的数据，权重为 1.0。
    - 剩余的数据，权重为 args.low_loss_weight。
    """
    print(f"Applying reward difference weighting processor...")
    
    # 1. 将数据按 prompt 分组
    prompts = {}
    for item in output_dataset:
        prompt = item["input"]
        if prompt not in prompts:
            prompts[prompt] = []
        prompts[prompt].append({"output": item["output"], "reward": item["reward"]})
    
    # 2. 为每个 prompt 内的回答创建 (chosen, rejected) 对，并计算权重
    paired_dataset = []
    reward_margins = [] # 用于存储所有的奖励差值

    for prompt, responses in prompts.items():
        if len(responses) < 2:
            continue
        
        # 对回答按奖励降序排序
        responses.sort(key=lambda x: x["reward"], reverse=True)
        
        # 创建所有可能的偏好对 (best vs rest)
        chosen = responses[0]
        for i in range(1, len(responses)):
            rejected = responses[i]
            margin = chosen["reward"] - rejected["reward"]
            reward_margins.append(margin)
            paired_dataset.append({
                "prompt": prompt,
                "chosen": chosen["output"],
                "rejected": rejected["output"],
                "chosen_reward": chosen["reward"],
                "rejected_reward": rejected["reward"],
                "margin": margin,
                # 先临时设置一个占位符权重
                "loss_weight": 1.0,
            })

    if not paired_dataset:
        return []

    # 3. 根据所有差值的分布来确定权重
    # 将 margin 和 paired_dataset 里的数据一一对应
    sorted_indices = torch.tensor([m for m in reward_margins]).argsort(descending=True)
    
    num_samples = len(paired_dataset)
    top_k_index = int(num_samples * args.top_reward_diff_percentile)

    print(f"Total pairs: {num_samples}. Top {args.top_reward_diff_percentile * 100:.1f}% ({top_k_index} samples) will have weight 1.0.")
    print(f"Remaining {num_samples - top_k_index} samples will have weight {args.low_loss_weight}.")
    
    # 4. 创建最终的带权重的数据集
    final_dataset = []
    weights = [1.0] * top_k_index + [args.low_loss_weight] * (num_samples - top_k_index)

    # 按排序后的索引重新构建数据集，并赋予正确的权重
    for i, original_idx in enumerate(sorted_indices):
        item = paired_dataset[original_idx]
        item["loss_weight"] = weights[i]
        final_dataset.append(item)
        
    return final_dataset

def weighted_reward_diff_processor(args, objs):


    # 确保相关参数存在
    if not hasattr(args, "top_reward_diff_percentile") or not hasattr(args, "sigmoid_temperature"):
        raise ValueError("缺少 --top_reward_diff_percentile 或 --sigmoid_temperature 参数")

    # 1. 找出所有偏好对并计算差值
    all_pairs = []
    out = {}

    # 按 prompt 分组并找到最佳/最差（复用之前的逻辑）
    for obj in tqdm(objs, desc="Finding best/worst pairs for all prompts..."):
        input = obj["input"]
        output = obj["output"]
        reward = float(obj["reward"])
        if input not in out:
            out[input] = {
                "chosen": output,
                "chosen_reward": reward,
                "rejected": output,
                "rejected_reward": reward,
            }
        elif reward > out[input]["chosen_reward"]:
            out[input]["chosen_reward"] = reward
            out[input]["chosen"] = output
        elif reward < out[input]["rejected_reward"]:
            out[input]["rejected_reward"] = reward
            out[input]["rejected"] = output

    for k, v in out.items():
        # 只保留 chosen_reward > rejected_reward 时才构成有效对（或者你可以保留反例但权重极低）
        if v["chosen_reward"] > v["rejected_reward"]:
            diff = v["chosen_reward"] - v["rejected_reward"]
            all_pairs.append(
                {
                    "prompt": k,
                    "chosen": v["chosen"],
                    "rejected": v["rejected"],
                    "reward_diff": diff,
                }
            )

    # 2. 根据奖励差值进行全局降序排序
    all_pairs.sort(key=lambda x: x["reward_diff"], reverse=True)

    # 3. 计算分割点
    percentile = args.top_reward_diff_percentile
    cutoff_index = int(len(all_pairs) * percentile)

    # 计算剩余数据的比例系数（例如前10%是1，剩下90%就是0.9）
    remaining_ratio = 1.0

    # 获取温度参数
    T = args.sigmoid_temperature

    final_dataset = []
    for i, pair in enumerate(all_pairs):
        diff = pair["reward_diff"]

        if i < cutoff_index:
            # 前 X% 的数据，权重固定为 1.0
            weight = 1.0
        else:
            # 剩余数据：Sigmoid 动态权重 * 比例系数
            # Sigmoid = 1 / (1 + exp(-diff / T))
            # 注意: diff 应该是正数，所以 sigmoid 值在 (0.5, 1.0) 之间
            # 奖励差越大，权重越高；差越小，权重越接近 0.5 * ratio
            sigmoid_val = 1.0 / (1.0 + np.exp(-diff / T))

            weight = sigmoid_val * remaining_ratio

        pair["weight"] = float(weight)  # 确保是 float
        del pair["reward_diff"]  # 移除临时的键
        final_dataset.append(pair)

    print(f"Total pairs: {len(final_dataset)}.")
    print(f"Top {cutoff_index} pairs: weight = 1.0")
    print(
        f"Remaining {len(final_dataset) - cutoff_index} pairs: Weight = Sigmoid(diff/{T}) * {remaining_ratio:.2f}"
    )

    return final_dataset


def bees_intersection_processor(args, objs):
    """
    逻辑修复版：
    1. 先按 Prompt 将数据分组。
    2. 在每个组内找出奖励最高的作为 Chosen，最低的作为 Rejected。
    3. 计算所有对子的 m_ex 和 m_im。
    4. 全局计算 Top 30% 阈值并分配权重。
    """
    import numpy as np
    from tqdm import tqdm

    print(f"Applying BeeS Intersection Processor (Top {args.top_reward_diff_percentile*100}%)")

    # --- 步骤 1: 按 Prompt 分组并配对 ---
    # 我们需要先将平铺的 objs 转为以 prompt 为 key 的字典
    prompt_groups = {}
    for obj in objs:
        prompt = obj["input"]
        if prompt not in prompt_groups:
            prompt_groups[prompt] = []
        prompt_groups[prompt].append(obj)

    paired_objs = []
    
    for prompt, responses in tqdm(prompt_groups.items(), desc="Pairing responses"):
        if len(responses) < 2:
            continue
        
        # 按照外部 RM 分数排序，取最好和最坏
        responses.sort(key=lambda x: x["reward"], reverse=True)
        chosen = responses[0]
        rejected = responses[-1]

        # 检查是否包含隐式 Log-probs (这些字段应该是由前置推理步骤加入的)
        # 字段名需与你 batch_inference.py 中保存的一致
        try:
            m_ex = chosen["reward"] - rejected["reward"]
            m_im = (chosen["chosen_logp_policy"] - chosen["chosen_logp_ref"]) - \
                   (rejected["rejected_logp_policy"] - rejected["rejected_logp_ref"])
            
            paired_objs.append({
                "prompt": prompt,
                "chosen": chosen["output"],
                "rejected": rejected["output"],
                "chosen_reward": chosen["reward"],
                "rejected_reward": rejected["reward"],
                "m_ex": m_ex,
                "m_im": m_im,
            })
        except KeyError as e:
            # 如果缺少隐式分字段，报错提示
            raise KeyError(f"数据中缺少字段 {e}。请确保在执行此 Processor 前已运行 Log-probs 推理步骤。")

    if not paired_objs:
        print("Warning: No valid pairs found!")
        return []

    # --- 步骤 2: 全局计算阈值 ---
    m_ex_list = [p["m_ex"] for p in paired_objs]
    m_im_list = [p["m_im"] for p in paired_objs]

    # q = 1 - 0.3 = 0.7 (即排名在前 30% 的分界线)
    q = 1.0 - args.top_reward_diff_percentile
    threshold_ex = np.quantile(m_ex_list, q)
    threshold_im = np.quantile(m_im_list, q)

    # --- 步骤 3: 分配权重 ---
    final_dataset = []
    counts = {1.0: 0, 0.5: 0, 0.0: 0}

    for p in paired_objs:
        is_ex_top = p["m_ex"] >= threshold_ex
        is_im_top = p["m_im"] >= threshold_im

        if is_ex_top and is_im_top:
            weight = 1.0
        elif is_ex_top or is_im_top:
            weight = 0.5
        else:
            weight = 0.0
        
        counts[weight] += 1
        
        # 构建最终 DPO 训练格式
        final_dataset.append({
            "prompt": p["prompt"],
            "chosen": p["chosen"],
            "rejected": p["rejected"],
            "weight": float(weight), # 确保 RewardDataset 能读到这个 key
            # 也可以保留这些用于分析，但 DPO 训练不需要
            "chosen_reward": p["chosen_reward"],
            "rejected_reward": p["rejected_reward"]
        })
    
    print(f"BeeS Stats -> Total Pairs: {len(final_dataset)}")
    print(f"Weight Distribution -> 1.0: {counts[1.0]}, 0.5: {counts[0.5]}, 0.0: {counts[0.0]}")
    
    return final_dataset


PROCESSORS = {
    "rs": rejection_sampling_processor,
    "csft": conditional_sft_processor,
    "iter_dpo": iterative_dpo_processor,
    "weighted_reward_diff": weighted_reward_diff_processor,
    "bees_intersection": bees_intersection_processor
}


def get_processor(name):
    if name in PROCESSORS:
        return PROCESSORS[name]
    else:
        raise ValueError(f"Processor {name} does not exist.")
