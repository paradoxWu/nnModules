# -*- coding: utf-8 -*-
import time, csv
import numpy as np
import pandas as pd, matplotlib.pyplot as plt
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Noto Sans CJK JP'
mpl.rcParams['axes.unicode_minus'] = False   # 解决负号乱码

def search_font():
    import matplotlib.font_manager as fm
    for f in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
        if 'Noto' in f:
            print(f)          # 路径
            print(fm.FontProperties(fname=f).get_name())  # 家族名

def record_gpu_status(record_sec):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)  # 卡 0
    with open("gpu_mem.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "used_MB", "free_MB", "total_MB"])
        # unit:s
        
        for _ in range(record_sec):                # 记录 2 min
            info = nvmlDeviceGetMemoryInfo(handle)
            writer.writerow([time.time(), info.used//1024**2, info.free//1024**2, info.total//1024**2])
            f.flush() 
            time.sleep(5)

def draw_plt():
    df = pd.read_csv("gpu_mem.csv")
    df["timestamp"] -= df["timestamp"].iloc[0]   # 相对时间
    max_usd_mem = np.max(df["used_MB"])
    min_usd_mem = np.min(df["used_MB"])
    print(f"vlm max_usd_mem={max_usd_mem-min_usd_mem}")
    plt.plot(df["timestamp"], df["used_MB"])
    plt.xlabel("Time (s)")
    plt.ylabel("Used Memory (MB)")
    plt.title("RTX 3060 12 GB 显存变化")
    plt.grid()
    plt.savefig("gpu_mem.jpg")


if __name__ == "__main__":
    record_sec = 60 // 5  * 20 
    record_gpu_status(record_sec)
    draw_plt()
    # search_font()
