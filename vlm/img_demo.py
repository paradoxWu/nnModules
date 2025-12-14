from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch,os

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1" 
# 定义量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# 加载模型和处理器
model_name = "/home/wuyuanhao/model/Qwen/Qwen2-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    max_memory={0: "10GiB", "cpu": "4GiB"}
)
processor = AutoProcessor.from_pretrained(model_name)

# 进行预测
image = "https://raw.githubusercontent.com/Qwen-Model/assets/main/qwen-vl-demo.png"
text = "Hello, how are you?"
inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs)
print(processor.decode(outputs[0], skip_special_tokens=True))