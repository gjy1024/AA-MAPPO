import torch
import torch.nn as nn
from algorithms.utils.util import init, get_clones

class MLPBase(nn.Module):
    """
    Attention-Augmented MLP (AA-MAPPO 核心网络层)
    在标准的MAPPO全连接层中，无缝植入多头注意力机制，用于特征提取。
    """
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()
        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        obs_dim = obs_shape[0] * self._stacked_frames

        # 特征归一化层
        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        # 权重初始化方法
        init_method =[nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        # 1. 初始特征映射层
        self.fc1 = nn.Sequential(
            init_(nn.Linear(obs_dim, self.hidden_size)),
            nn.ReLU() if self._use_ReLU else nn.Tanh()
        )

        # ================== 核心创新：多头注意力层 ==================
        # embed_dim 必须等于 hidden_size，batch_first=True 适配强化学习的Tensor维度
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=4, batch_first=True)
        # 引入残差连接后的归一化，保证训练极度稳定
        self.attn_norm = nn.LayerNorm(self.hidden_size)
        # ============================================================

        # 2. 后续的深度前馈网络
        self.mlp = nn.Sequential()
        for i in range(self._layer_N - 1):
            self.mlp.add_module(f"fc_{i}", init_(nn.Linear(self.hidden_size, self.hidden_size)))
            self.mlp.add_module(f"act_{i}", nn.ReLU() if self._use_ReLU else nn.Tanh())

    def forward(self, x):
        # 1. 特征归一化
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        # 2. 映射到隐藏层维度
        x = self.fc1(x)

        # ================== 核心创新：注意力机制前向传播 ==================
        # 强行给数据增加一个序列维度 (batch_size, seq_len=1, hidden_size)
        x_attn = x.unsqueeze(1) 
        
        # 自注意力运算 (Query=Key=Value=x_attn)
        # 这一步使神经网络具备了评估当前特征重要性的能力
        attn_out, _ = self.attention(x_attn, x_attn, x_attn)
        
        # 残差连接 (Residual Connection) + LayerNorm，防止梯度消失
        x = self.attn_norm(x + attn_out.squeeze(1))
        # ==================================================================

        # 3. 经过后续隐藏层输出最终状态表征
        x = self.mlp(x)
        return x