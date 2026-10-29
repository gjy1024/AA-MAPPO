"""Entity-token attention and tanh-Gaussian actor / centralized critic."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from .environment import TOKEN_DIM


class EntityEncoder(nn.Module):
    def __init__(self, tokens, hidden, heads, attention, actor=False):
        super().__init__()
        self.uses_attention = attention
        self.actor = actor
        if attention:
            self.embedding = nn.Sequential(nn.Linear(TOKEN_DIM, hidden), nn.Tanh())
            self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.0)
            self.norm = nn.LayerNorm(hidden)
            self.output = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        else:
            self.mlp = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(tokens * TOKEN_DIM, hidden),
                                     nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())

    def forward(self, tokens, return_attention=False):
        if not self.uses_attention:
            result, weights = self.mlp(tokens), None
        else:
            encoded = self.embedding(tokens)
            query = encoded[:, :1] if self.actor else encoded
            context, weights = self.attention(query, encoded, encoded,
                                               need_weights=return_attention,
                                               average_attn_weights=False)
            result = self.output(self.norm(query + context).mean(dim=1))
        return (result, weights) if return_attention else result


def transformed_log_prob(distribution, raw_actions):
    # Stable log(1 - tanh(u)^2), including saturated tails.
    log_jacobian = 2 * (math.log(2.0) - raw_actions - F.softplus(-2 * raw_actions))
    return (distribution.log_prob(raw_actions) - log_jacobian).sum(dim=-1)


class Actor(nn.Module):
    def __init__(self, config, token_count):
        super().__init__()
        self.encoder = EntityEncoder(token_count, config.hidden_size, config.num_heads,
                                     config.attention_mode in ("actor", "both"), actor=True)
        self.mean = nn.Linear(config.hidden_size, 2)
        nn.init.orthogonal_(self.mean.weight, gain=.01)
        nn.init.zeros_(self.mean.bias)
        self.log_std = nn.Parameter(torch.full((2,), config.initial_log_std))

    def distribution(self, tokens):
        return Normal(self.mean(self.encoder(tokens)), self.log_std.clamp(-5, 2).exp())

    def sample(self, tokens, deterministic=False):
        distribution = self.distribution(tokens)
        raw = distribution.mean if deterministic else distribution.sample()
        return raw.tanh(), raw, transformed_log_prob(distribution, raw)

    def evaluate_actions(self, tokens, raw_actions):
        distribution = self.distribution(tokens)
        log_probability = transformed_log_prob(distribution, raw_actions)
        # Monte Carlo entropy estimate of the transformed action distribution.
        sample = distribution.rsample()
        entropy = -transformed_log_prob(distribution, sample)
        return log_probability, entropy

    def attention_weights(self, tokens):
        _, weights = self.encoder(tokens, return_attention=True)
        if weights is None:
            return None
        return weights.mean(dim=1).squeeze(1)


class Critic(nn.Module):
    def __init__(self, config, token_count):
        super().__init__()
        self.encoder = EntityEncoder(token_count, config.hidden_size, config.num_heads,
                                     config.attention_mode in ("critic", "both"))
        self.value = nn.Linear(config.hidden_size, 1)

    def forward(self, tokens):
        return self.value(self.encoder(tokens)).squeeze(-1)


def make_networks(config, env):
    return Actor(config, env.actor_token_count), Critic(config, env.critic_token_count)

