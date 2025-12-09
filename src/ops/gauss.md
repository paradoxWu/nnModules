下面给出一份 **“高斯模糊 = 热力图平滑”** 的 **完整速查包**（2025-12 仍通用）：

- 原理一句话  
- 二维离散公式  
- separable 实现（O(n²)→O(n)）  
- **CUDA 版** & **OpenCV 一行版**  
- **热力图实战**（0-1 浮点 → 白-红调色）

---

### 1. 一句话原理

> **“每个像素 = 邻域加权平均，权重 = 二维高斯分布”**  
> **σ 越大 → 权重越分散 → 图像越糊**。

---

### 2. 二维离散高斯核公式

$$G(x,y)=\frac{1}{2\pi\sigma^2}\exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)$$

**核大小** 取 **2⌈3σ⌉+1**（经验 3σ 截断）。

---

### 3. 可分离卷积（CPU 参考实现）

```cpp
#include <vector>
#include <cmath>

// 生成 1D 高斯核
std::vector<float> gauss1D(float sigma, int& radius_out) {
    radius_out = static_cast<int>(std::ceil(3.0f * sigma));
    int size = 2 * radius_out + 1;
    std::vector<float> kernel(size);
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        float x = i - radius_out;
        kernel[i] = std::exp(-0.5f * x * x / (sigma * sigma));
        sum += kernel[i];
    }
    // 归一化
    for (auto& v : kernel) v /= sum;
    return kernel;
}

// 1D 卷积（in-place）
void conv1D(const float* in, float* out, int len,
            const float* kernel, int radius) {
    int ksize = 2 * radius + 1;
    for (int i = 0; i < len; ++i) {
        float sum = 0.0f;
        for (int k = 0; k < ksize; ++k) {
            int src = i - radius + k;
            src = std::max(0, std::min(src, len - 1)); // 边缘复制
            sum += in[src] * kernel[k];
        }
        out[i] = sum;
    }
}

// 二维可分离高斯模糊
void gaussianBlur2D(const float* in, float* out, int h, int w, float sigma) {
    int radius;
    auto kernel = gauss1D(sigma, radius);
    std::vector<float> tmp(h * w);
    // 先横向
    for (int y = 0; y < h; ++y)
        conv1D(in + y * w, tmp.data() + y * w, w, kernel.data(), radius);
    // 再纵向
    for (int x = 0; x < w; ++x) {
        std::vector<float> col(h), col_out(h);
        for (int y = 0; y < h; ++y) col[y] = tmp[y * w + x];
        conv1D(col.data(), col_out.data(), h, kernel.data(), radius);
        for (int y = 0; y < h; ++y) out[y * w + x] = col_out[y];
    }
}
```

---

### 4. 热力图实战（0-1 → 白-红）

```python
import cv2
import numpy as np

# 0. 生成模拟热力图 (H, W, 1)
h, w = 180, 320
heat = np.random.rand(h, w).astype(np.float32)

# 1. 高斯模糊
blur = cv2.GaussianBlur(heat, (0, 0), sigmaX=5, sigmaY=5)

# 2. 伪彩色 (白→红)
color = cv2.applyColorMap((blur * 255).astype(np.uint8), cv2.COLORMAP_JET)

cv2.imwrite("heat_gauss.png", color)
```

---

### 5. CUDA 一行（PyTorch）

```python
import torch
heat = torch.rand(1, 1, 180, 320, device='cuda')
blur = torch.nn.functional.gaussian_blur(heat, kernel_size=(21, 21), sigma=(5., 5.))
```

---

### 6. 一句话口诀

> **“高斯核 = 二维钟形权重；σ 控制模糊半径；**  
> **先横后纵 = O(n²)→O(n)；**  
> **热力图 = 高斯模糊 + 伪彩色映射。”**