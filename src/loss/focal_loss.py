import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss 实现
        Args:
            alpha: 类别权重，解决类别不平衡，shape=[num_classes]，如二分类可设[0.25, 0.75]
            gamma: 调制因子，gamma越大，易分样本权重越低，通常取2
            reduction: 损失聚合方式，可选 'none'/'mean'/'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        # 若alpha是列表，转为tensor
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def forward(self, inputs, targets):
        """
        前向计算
        Args:
            inputs: 模型输出，shape=[batch_size, num_classes]，可以是logits（未经过softmax）
            targets: 真实标签，shape=[batch_size]，是类别索引（非one-hot）
        Returns:
            计算好的focal loss
        """
        # 1. 将logits转为概率分布
        if inputs.dim() == 2:
            # 多分类：logits → softmax 概率
            probs = F.softmax(inputs, dim=1)
        else:
            # 二分类：logits → sigmoid 概率（需保证inputs shape=[batch_size]）
            probs = torch.sigmoid(inputs)
            # 适配二分类的概率格式，shape=[batch_size, 2]
            probs = torch.cat([1 - probs, probs], dim=1)

        # 2. 提取每个样本对应真实类别的概率
        # 生成one-hot标签，shape=[batch_size, num_classes]
        num_classes = probs.size(1)
        one_hot = F.one_hot(targets, num_classes=num_classes).float()
        # 真实类别概率 p_t，shape=[batch_size]
        p_t = torch.sum(probs * one_hot, dim=1)

        # 3. 计算调制因子 (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma

        # 4. 计算带alpha权重的交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        if self.alpha is not None:
            # 将alpha放到对应设备
            alpha = self.alpha.to(inputs.device)
            # 提取真实类别的alpha权重
            alpha_t = torch.sum(alpha * one_hot, dim=1)
            ce_loss = alpha_t * ce_loss

        # 5. 计算最终focal loss
        focal_loss = modulating_factor * ce_loss

        # 6. 损失聚合
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

if __name__ == "__main__":
    # 初始化focal loss，二分类，alpha解决类别不平衡，gamma=2
    focal_loss = FocalLoss(alpha=[0.25, 0.75], gamma=2, reduction='mean')

    # 模拟输入：batch_size=4，二分类logits
    inputs = torch.randn(4, 2)
    # 模拟真实标签：类别0和1
    targets = torch.tensor([0, 1, 0, 1])

    # 计算损失
    loss = focal_loss(inputs, targets)
    print(loss)