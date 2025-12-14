from modelscope import snapshot_download
from datasets import load_dataset
import torch
import vllm
import os
print(torch.__version__, torch.cuda.is_available())   # 期望 2.5.1 or 2.9 + True
print(vllm.__version__)  

# 下载模型权重
# model_dir = snapshot_download("Qwen/Qwen2-VL-7B-Instruct")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#下载数据集
ds = load_dataset("cais/mmlu", "all")
