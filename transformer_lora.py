import lora

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import torch
import math
from layers import general_conv3d_prenorm, fusion_prenorm
#from einops import rearrange

basic_dims = 8
transformer_basic_dims = 512
mlp_dim = 4096
num_heads = 8
depth = 1
num_modals = 4
patch_size = 8

class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)
        self.lora_0 = lora.MergedLinear(
            dim, dim*3,
            r=4,
            lora_alpha=1,
            lora_dropout=0.,
            enable_lora=[True, False, True],
            fan_in_fan_out=False,
            merge_weights=False
        )
        self.lora_1 = lora.MergedLinear(
            dim, dim*3,
            r=4,
            lora_alpha=1,
            lora_dropout=0.,
            enable_lora=[True, False, True],
            fan_in_fan_out=False,
            merge_weights=False
        )
        self.lora_2 = lora.MergedLinear(
            dim, dim*3,
            r=4,
            lora_alpha=1,
            lora_dropout=0.,
            enable_lora=[True, False, True],
            fan_in_fan_out=False,
            merge_weights=False
        )
        self.lora_3 = lora.MergedLinear(
            dim, dim*3,
            r=4,
            lora_alpha=1,
            lora_dropout=0.,
            enable_lora=[True, False, True],
            fan_in_fan_out=False,
            merge_weights=False
        )

    def forward(self, x, idx=None):
        hidden_states = x
        lora_attr_name = f"lora_{idx}" if idx is not None else "lora"
        lora_layer = getattr(self, lora_attr_name)
        B, N, C = x.shape
        qkv = (
            lora_layer(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, idx=None):
        if idx is not None:
            return self.fn(x, idx) + x  
        else:
            return self.fn(x) + x



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x, idx=None):
        if idx is not None:
            return self.dropout(self.fn(self.norm(x), idx))
        else:
            return self.dropout(self.fn(self.norm(x)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class Transformer_LoRA(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1, n_levels=1, n_points=4):
        super(Transformer_LoRA, self).__init__()
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.depth = depth
        for j in range(self.depth):
            self.cross_attention_list.append(
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )
            self.cross_ffn_list.append(
                Residual(
                    PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )

        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.cross_ffn_list = nn.ModuleList(self.cross_ffn_list)


    def forward(self, x, pos, idx):
        for j in range(self.depth):
            x = x + pos
            x = self.cross_attention_list[j](x, idx)
            x = self.cross_ffn_list[j](x)
        return x