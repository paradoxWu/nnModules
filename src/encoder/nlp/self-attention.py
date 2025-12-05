import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.register_buffer("mask_val", torch.tensor(-1e9))

    def forward(self, x, mask=None):
        # x: [B, T, d]
        B, T, _ = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, self.mask_val)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return out

class MultiHeadSelfAttention(nn.Module):
   def __init__(self, embed_size, heads):
       super(MultiHeadSelfAttention, self).__init__()
       assert embed_size % heads == 0
       self.heads = heads
       self.head_dim = embed_size // heads
       self.query = nn.Linear(embed_size, embed_size)
       self.key = nn.Linear(embed_size, embed_size)
       self.value = nn.Linear(embed_size, embed_size)
       self.fc_out = nn.Linear(embed_size, embed_size)
       self.register_buffer("mask_val", torch.tensor(-1e9))
   def forward(self, x,mask):
       N, seq_len, _ = x.shape
       Q = self.query(x).view(N, seq_len, self.heads, self.head_dim).transpose(1, 2)
       K = self.key(x).view(N, seq_len, self.heads, self.head_dim).transpose(1, 2)
       V = self.value(x).view(N, seq_len, self.heads, self.head_dim).transpose(1, 2)
       scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
       if mask is not None:
           mask = mask.unsqueeze(-1).repeat(1,1,self.heads,1)
           scores = scores.masked_fill(mask == 0, self.mask_val)
       attn_weights = F.softmax(scores, dim=-1)
       out = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(N, seq_len, -1)
       return self.fc_out(out)


if __name__ == "__main__":
    feature_num = 9
    batch_size = 2
    time_steps = 3
    x = torch.randn(batch_size, time_steps, feature_num)
    mask = torch.randint(0, 2, (batch_size, time_steps, 1))
    model = SelfAttention(9)
    output0 = model(x, mask)
    print(output0.shape)
    model = MultiHeadSelfAttention(9,3)
    output0 = model(x, mask)
    print(output0.shape)
