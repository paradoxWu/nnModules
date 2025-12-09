# demo_profiler.py
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

device = "cuda" if torch.cuda.is_available() else "cpu"
B, D, H = 64, 512, 256
model = nn.Sequential(
    nn.Linear(D, H),
    nn.ReLU(),
    nn.Linear(H, H),
    nn.ReLU(),
    nn.Linear(H, 10)
).to(device)
x = torch.randn(B, D, device=device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# 1) 准备 warm-up，避免 CUDA malloc 干扰
for _ in range(5):
    y = model(x)
    loss = y.sum()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

# 2) 正式记录
with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # 记录 tensor 分配
        profile_memory=True,
        # 保存 timeline
        with_stack=True,
        with_flops=True,
        # 跳过前 3 个 step，只记第 4~6 step
        schedule=torch.profiler.schedule(wait=3, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),  # 供 TensorBoard 打开
) as prof:
    for step in range(10):
        with record_function("forward"):
            y = model(x)
        with record_function("loss"):
            loss = y.sum()
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            opt.step()
            opt.zero_grad(set_to_none=True)

        prof.step()  # 通知 profiler 当前迭代结束

# 3) 控制台打印「按 CUDA 时间排序」的汇总表
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# 4) 导出 Chrome trace（用 chrome://tracing 打开）
prof.export_chrome_trace("trace.json")