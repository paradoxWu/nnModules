from datasets import load_dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
import evaluate
import torch
import os, time
from tqdm import tqdm

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def forward(model, processor, batch):
    # MMLU 格式：choice A B C D → 0 1 2 3
    questions = batch["question"]
    choices = batch["choices"]  # list[list[str]]
    refs = batch["answer"]  # list[int]
    preds = []
    for q, opts in zip(questions, choices):
        prompt = f"Question: {q}\nOptions:\nA: {opts[0]}\nB: {opts[1]}\nC: {opts[2]}\nD: {opts[3]}\nAnswer:"
        inputs = processor(text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        pred = (
            processor.decode(
                out[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            .strip()
            .upper()
        )
        # 字母 → 数字
        preds.append({"A": 0, "B": 1, "C": 2, "D": 3}.get(pred, -1))
    return preds


def main(use_babc=False, batch_size=8):
    acc = evaluate.load("accuracy")

    # 加载 MMLU（只取 1 个学科调试，全量把 split='test' 即可）
    # data_path = "~/.cache/huggingface/datasets/cais___mmlu"
    mmlu = load_dataset("cais/mmlu", "all", split="dev")

    model_path = "/home/wuyuanhao/model/"
    model_name = (
        "Qwen/Qwen2-VL-7B-Instruct"  # "Qwen/Qwen2-7B-Instruct-Int4"  # 官方已量化
    )
    model_id = model_path + model_name
    # if use bitsandbytes to quant
    quant_config = None
    if use_babc:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # 精度最高
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )  # 再省 0.3-0.5 GB

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        max_memory={0: "10GiB", "cpu": "4GiB"},
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # 批量推理（batch=8 显存峰值 < 7 GB）
    preds = []

    for i in tqdm(range(0, len(mmlu), batch_size), desc="MMLU batch"):
        preds.extend(forward(model, processor, mmlu[i : i + batch_size]))

    acc.add_batch(predictions=preds, references=mmlu["answer"])
    print("MMLU accuracy (100 samples):", acc.compute())


def vllm_infer():
    from vllm import LLM, SamplingParams

    model_path = "/home/wuyuanhao/model/"
    model_name = (
        "Qwen/Qwen2-VL-7B-Instruct"  # "Qwen/Qwen2-7B-Instruct-Int4"  # 官方已量化
    )
    model_id = model_path + model_name
    model = LLM(
        model=model_id,
        quantization="bitsandbytes",
        load_format="bitsandbytes",      # ← 显式指定
        gpu_memory_utilization=0.6,
        kv_cache_dtype="fp8_e5m2",
        dtype="bfloat16",
        max_model_len=512,
        trust_remote_code=True,
    )
    params = SamplingParams(max_tokens=1, temperature=0)

    mmlu = load_dataset("cais/mmlu", "all", split="test")
    prompts = [
        f"Question: {q}\nA: {c[0]}\nB: {c[1]}\nC: {c[2]}\nD: {c[3]}\nAnswer:"
        for q, c in zip(mmlu["question"], mmlu["choices"])
    ]

    t0 = time.time()
    outputs = model.generate(prompts, params)  # 内部自动 batch=32+
    t1 = time.time()

    preds = [out.outputs[0].text.strip().upper()[0] for out in outputs]
    preds = [{"A": 0, "B": 1, "C": 2, "D": 3}.get(p, -1) for p in preds]
    acc = sum(p == r for p, r in zip(preds, mmlu["answer"])) / len(preds)

    print(f"Accuracy: {acc:.4f}")
    print(f"Total time: {t1 - t0:.1f}s  |  Speed: {len(prompts) / (t1 - t0):.1f} sam/s")


if __name__ == "__main__":
    use_babc = True
    main(use_babc=use_babc,batch_size = 16)
    # vllm_infer()
