- register_buffer和直接赋值一个变量常数有什么区别

  1. **是否被模型状态收录**
     * `register_buffer(name, tensor)` → **计入 `model.state_dict()`** → 保存/加载时会随模型一起序列化。
     * 直接赋值 →  **普通 Python 属性** ，**不计入 state_dict** → 保存模型时 **丢失** 。
  2. **设备移动（`cuda/cpu`）**
     * `buffer`：当调用 `model.cuda()` 或 `model.to(device)` 时 **自动跟随移动** 。
     * 直接赋值：留在**原设备**不动 → 需要**手工 `.to(device)`** 才能对齐。
  3. **优化器过滤**
     * `buffer`：**不会**被 `optimizer` 收录（默认 `requires_grad=False`）。
     * 直接赋值：若你误写 `self.foo = nn.Parameter(tensor)` → **会被当做可训练参数** → 增大参数量、可能报「未使用参数」warning。

- 什么时候用 register_buffer？
  * **mask / 常量 / running 统计量** （mean, std）
  * **不可训练、但要随模型保存/移动** 的张量
  * **防止被 optimizer 收录** （如缺失掩码、位置编码表）
