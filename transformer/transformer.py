import torch
import torch.nn as nn
import math


def scaled_dot_product_attention(q, k, v, mask=None):
    # 计算注意力分数
    d_k = k.size(-1)
    attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 应用mask
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
    
    # 应用softmax得到注意力权重
    attention_weights = torch.softmax(attention_scores, dim=-1)
    
    # 计算输出
    output = torch.matmul(attention_weights, v)
    
    return output, attention_weights


class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, head_dim)
        self.k_proj = nn.Linear(embed_dim, head_dim)
        self.v_proj = nn.Linear(embed_dim, head_dim)

    def forward(self, x):
        # 1. 将 x 传入线性层得到 q, k, v
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. 调用 scaled_dot_product_attention
        output, attention_weights = scaled_dot_product_attention(q, k, v)
        
        # 3. 返回输出结果
        return output

# --- 验证部分 ---
# 创建实例和输入数据，并检查输出形状
if __name__ == "__main__":
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    
    # 创建一个随机输入张量 x，形状为 (batch_size=2, seq_len=4, embed_dim=8)
    batch_size, seq_len, embed_dim = 2, 4, 8
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # 创建 SimpleSelfAttention 实例，设置 head_dim=8
    head_dim = 8
    self_attention = SimpleSelfAttention(embed_dim, head_dim)
    
    # 将张量输入到 SimpleSelfAttention 模块中
    output = self_attention(x)
    
    # 打印输出张量的形状，并验证其是否与输入张量的形状一致
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    
    # 验证输出形状是否与输入形状一致
    assert output.shape == x.shape, f"输出形状 {output.shape} 与输入形状 {x.shape} 不一致"
    print("验证成功！")