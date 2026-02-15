#!/bin/bash
# Stage2: 使用 Stage1 生成的 ICD 序列进行 C-Eval 推理和评估

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
python "$PROJECT_ROOT/test_icd_sequences.py" \
  --task ceval \
  --model llada \
  --sampler text_sim_qwen_mmr \
  --scorer infoscore \
  --construct_order no_order \
  --beam_size 5 \
  --few_shot 4 \
  --candidate_num 64 \
  --sample_num 200 \
  --mc_num 1 \
  --coarse_k 200 \
  --mmr_lambda 0.1 \
  --icd_rank 0 \
  --device cuda:1
