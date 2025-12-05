import torch
import torch.nn as nn


class gMLPBlock(nn.Module):
    def __init__(self, feature_size, sq_len, hidden_size=128):
        super().__init__()
        self.norm = nn.LayerNorm(feature_size)
        self.proj = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, feature_size),
        )
        # self.gate = nn.Conv1d(sq_len, sq_len, 1)
        # 等价于
        self.gate = nn.Linear(sq_len, sq_len)

    def forward(self, agent):
        x = agent
        x = self.norm(x)
        x = self.proj(x)
        # x = x.mean(dim=-1).squeeze()
        gate = self.gate(x)
        return agent + x * gate


class MaskgMLPBlock(gMLPBlock):
    def __init__(self, feature, sq_len, hidden_size=128):
        super().__init__(feature, sq_len, hidden_size)
        self.register_buffer("mask_val", torch.tensor(-1e4))

    def forward(self, agent, mask):
        x = agent
        x = self.norm(x)
        x = x * mask
        x = self.proj(x)
        gate_in = x.mean(dim=-1)
        mask = mask.squeeze(-1)
        gate_in = gate_in.masked_fill(mask == 0, self.mask_val)
        gate = self.gate(gate_in)
        gate = torch.sigmoid(gate)
        gate = gate * mask
        x = x * gate.unsqueeze(-1)
        return agent + x


class gMLPEncoder(nn.Module):
    def __init__(self, feature, sq_len, hidden_size=128, num_layers=3):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MaskgMLPBlock(feature, sq_len, hidden_size) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(feature)

    def forward(self, x, mask):
        for blk in self.blocks:
            x = blk(x, mask)
        return self.norm(x)

class Expand_gMLP(nn.Module):
    def __init__(self, feature, sq_len, out_d, hidden_size=128, num_layers=3):
        super().__init__()
        self.expand = nn.Linear(feature, hidden_size)
        self.gmlp = gMLPEncoder(hidden_size, sq_len, hidden_size, num_layers=num_layers)
        self.compress = nn.Linear(hidden_size, out_d)
        # self.pool = nn.AdaptiveAvgPool1d(1)  # 沿 N 维平均，再 Linear

    def forward(self, x, mask):
        x = self.expand(x)
        x = self.gmlp(x, mask)
        x = self.compress(x)
        x = x.mean(dim=2)
        # x = self.pool(x).squeeze(-1)          # (B,T,d_out)
        return x 

if __name__ == "__main__":
    mock_input_0 = torch.randn(128, 31, 39, 9)
    mock_input_1 = torch.randint(0, 2, (128, 31, 39, 1))
    model_time = gMLPEncoder(9, 39)
    output0 = model_time(mock_input_0, mock_input_1)
    print(output0.shape)

    mock_input_0 = mock_input_0.permute(0, 2, 1, 3)
    mock_input_1 = mock_input_1.permute(0, 2, 1, 3)
    model_space = gMLPEncoder(9, 31)
    output1 = model_space(mock_input_0, mock_input_1)
    print(output1.shape)
