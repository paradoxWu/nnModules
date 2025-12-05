import numpy as np
import torch
import torch.nn as nn
'''
归一化,将特征的数值转化为均值0,方差1的数据
-特征分布稳定，减少梯度消失问题（数据在激活函数饱和区）
'''

def LayerNorm(x,gamma,beta,eps=1e-5):
    '''
    learnable parameters: gamma,beta
    gamma控制方差，beta控制均值
    '''
    mean = x.mean(-1,keepdims=True)
    var = x.var(-1,unbiased=False,keepdims=True)
    return gamma*(x-mean)/np.sqrt(var+eps)+beta

def BatchNorm(x,gamma,beta,eps=1e-5):
    mean = x.mean(dim=(0,1),keepdims=True)
    var = x.std(dim=(0,1),unbiased=False,keepdims=True)
    return gamma*(x-mean)/(var+eps)+beta

if __name__ == "__main__":
    feature_num = 4
    batch_size = 2
    time_steps = 3  
    x = torch.randn(batch_size, time_steps,feature_num)

    print(LayerNorm(x,1,0))
    # layernorm: feature is the last position
    model = nn.LayerNorm(feature_num,elementwise_affine=False)
    print(model(x))

    print(BatchNorm(x,1,0))
    # bathnorm: feature is the 2nd position
    model = nn.BatchNorm1d(feature_num,affine=False)
    y = x.permute(0,2,1) #(B,C,L)
    res = model(y).permute(0,2,1)
    print(res)