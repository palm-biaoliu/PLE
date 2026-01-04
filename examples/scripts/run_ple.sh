#!/bin/bash
set -x

# ====== 通用环境 ======
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=warn
export NCCL_IB_DISABLE=1

# ✅ 显式设置 master 地址，避免跨网卡问题
export MASTER_ADDR=127.0.0.1
checkSuccess() {
   if [[ $? != 0 ]]; then
      echo "FAILED $1"
      exit 1
   fi
}

# ====== 端口分配策略（重要）======
# ✅ 给每次迭代的 RM / DPO 分配不同端口，避免前一个阶段未完全退出导致 EADDRINUSE
MASTER_PORT_BASE=29600   # 可改为你喜欢的起始空闲端口

# ====== 路径/参数 ======
BASE_CHECKPOINT_PATH=./checkpoint/iterative-dpo-weight
GENERATE_OUTPUT=${BASE_CHECKPOINT_PATH}/generate.jsonl
RM_OUTPUT=${BASE_CHECKPOINT_PATH}/rm.jsonl
PERCENTILE_SCHEDULE=(0.1 0.2 0.3 0.4 0.5)
TRAINING_ITERS=${#PERCENTILE_SCHEDULE[@]}
ROLLOUT_BATCH_SIZE=10240
INITIAL_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
POLICY_MODEL_PATH=$INITIAL_MODEL_PATH
REF_MODEL_PATH=$INITIAL_MODEL_PATH
iter=0
while (( iter < TRAINING_ITERS )); do
   echo "========================================================"
   echo " Iteration: $((iter + 1)) / $TRAINING_ITERS"
   echo "========================================================"
   current_percentile=${PERCENTILE_SCHEDULE[$iter]}
   # ==================== MODIFICATION: Update log message ====================
   echo "Current Weighting Strategy: Training on Top ${current_percentile}% with full weight, rest with low weight."
   # ========================================================================
   CURRENT_MODEL_OUTPUT_PATH=${BASE_CHECKPOINT_PATH}/iter_${iter}
   mkdir -p "$CURRENT_MODEL_OUTPUT_PATH"
   if (( iter > 0 )); then
      POLICY_MODEL_PATH=${BASE_CHECKPOINT_PATH}/iter_$((iter - 1))
   fi
   # ✅ 为本次迭代计算独立端口
   PORT_RM=$((MASTER_PORT_BASE + iter*2))       # RM 阶段端口
   PORT_DPO=$((MASTER_PORT_BASE + iter*2 + 1))  # DPO 阶段端口
   # --- Step 1: Generate（vLLM 推理，不需要 init_process_group）---
   read -r -d '' generate_commands <<EOF
openrlhf.cli.batch_inference \
   --eval_task generate_vllm \
   --pretrain $POLICY_MODEL_PATH \
   --max_new_tokens 768 \
   --prompt_max_len 1536 \
   --dataset openbmb/UltraFeedback \
   --input_key context_messages \
   --apply_chat_template \
   --temperature 0.9 \
   --top_p 0.95 \
   --tp_size 4 \
   --best_of_n 8 \
   --enable_prefix_caching \
   --max_num_seqs 64 \
   --iter $iter \
   --rollout_batch_size $ROLLOUT_BATCH_SIZE \
   --output_path $GENERATE_OUTPUT
EOF
   echo "--------- GENERATING... ---------"
   echo "$generate_commands"
   # 这里不是分布式进程组，保持 python -m 即可
   python -m $generate_commands
   checkSuccess "GENERATE"

   # --- Step 2: RM 评估 + 加权（分布式，需要端口）---
   read -r -d '' get_rewards_commands <<EOF
openrlhf.cli.batch_inference \
   --eval_task rm \
   --pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --bf16 \
   --max_len 4096 \
   --dataset $GENERATE_OUTPUT \
   --dataset_probs 1.0 \
   --zero_stage 2 \
   --post_processor weighted_reward_diff \
   --top_reward_diff_percentile $current_percentile \
   --sigmoid_temperature 10 \
   --micro_batch_size 1 \
   --output_path $RM_OUTPUT
EOF
   # ==================== MODIFICATION: Update log message ====================
   echo "--------- REWARDING & WEIGHTING... ---------"
   # ========================================================================
   echo "$get_rewards_commands"
   # ✅ 指定本阶段专用端口
   deepspeed --include localhost:0,1,2,3 --master_port $PORT_RM --module $get_rewards_commands
   checkSuccess "RM"

   # --- Step 3: DPO 训练（分布式，需要端口）---
   # (DPO 训练命令无需任何修改)
   read -r -d '' dpo_commands <<EOF
openrlhf.cli.train_dpo \
   --max_len 4096 \
   --dataset $RM_OUTPUT \
   --dataset_probs 1.0 \
   --prompt_key prompt \
   --train_batch_size 128 \
   --micro_train_batch_size 2 \
   --pretrain $POLICY_MODEL_PATH \
   --ref_pretrain $REF_MODEL_PATH \
   --save_path $CURRENT_MODEL_OUTPUT_PATH \
   --zero_stage 3 \
   --max_epochs 1 \
   --bf16 \
   --learning_rate 3e-6 \
   --beta 0.1 \
   --gradient_checkpointing
EOF
   echo "--------- DPO TRAINING... ---------"
   echo "$dpo_commands"
   # ✅ 指定与 RM 不同的端口，杜绝冲突
   deepspeed --include localhost:0,1,2,3 --master_port $PORT_DPO --module $dpo_commands
   checkSuccess "DPO"

   iter=$((iter + 1))
done

# ==================== MODIFICATION: Update final message ====================
echo "Curriculum DPO (Reward Difference Weighting) training finished successfully!"
# ========================================================================