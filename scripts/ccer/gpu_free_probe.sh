#!/usr/bin/env bash
# 快速查看各卡空闲显存，辅助选 CUDA_VISIBLE_DEVICES
nvidia-smi --query-gpu=index,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits \
  | awk -F', ' '{printf "GPU %s: free %d MiB / %d MiB (util %s%%)\n", $1, $2, $3, $4}'

echo ""
echo "Qwen3-32B bf16 约需 62GiB 单卡，或多卡 auto 切分 + CPU spill。"
echo "示例: export CUDA_VISIBLE_DEVICES=0,1,4"
