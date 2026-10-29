import torch
import torch.nn as nn
from algorithms.utils.util import init, get_clones

class MLPBase(nn.Module):
    """ Standard MLP (标准 MAPPO 核心网络层) - 无注意力机制对比组 """
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()
        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        obs_dim = obs_shape[0] * self._stacked_frames

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.fc1 = nn.Sequential(
            init_(nn.Linear(obs_dim, self.hidden_size)),
            nn.ReLU() if self._use_ReLU else nn.Tanh()
        )

        self.mlp = nn.Sequential()
        for i in range(self._layer_N - 1):
            self.mlp.add_module(f"fc_{i}", init_(nn.Linear(self.hidden_size, self.hidden_size)))
            self.mlp.add_module(f"act_{i}", nn.ReLU() if self._use_ReLU else nn.Tanh())

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)
        x = self.fc1(x)
        x = self.mlp(x)
        return x