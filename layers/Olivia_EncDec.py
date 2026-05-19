import torch
from torch import nn
from torch import Tensor
from layers.basics import SigmoidRange, Transpose, get_activation_fn, RMSNorm
from layers.pos_encoding import positional_encoding
from layers.Attention import HarmonicAttention

import torch.nn.functional as F
import math


class RegressionHead(nn.Module):
    def __init__(self, n_vars, d_model, output_dim, head_dropout, y_range=None):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars * d_model, output_dim)

    def forward(self, x):
        x = x[:, :, :, -1]
        x = self.flatten(x)
        x = self.dropout(x)
        y = self.linear(x)
        if self.y_range: y = SigmoidRange(*self.y_range)(y)
        return y


class ClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars * d_model, n_classes)

    def forward(self, x):
        x = x[:, :, :, -1]
        x = self.flatten(x)
        x = self.dropout(x)
        y = self.linear(x)
        return y


class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, patch_num, forecast_len, head_dropout=0, flatten=False):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model * patch_num

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.dropout(x)
            x = self.linear(x)
        return x.transpose(2, 1)


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        x = x.transpose(2, 3)
        x = self.linear(self.dropout(x))
        x = x.permute(0, 2, 1, 3)
        return x


class BackboneEncoder(nn.Module):
    def __init__(self, c_in, patch_num, patch_len, n_layers=3, d_model=128, n_heads=16, shared_embedding=True,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 res_attention=True, pre_norm=False, pe='zeros', learn_pe=True, verbose=False):
        super().__init__()
        self.n_vars = c_in
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding

        self.W_pos = positional_encoding(pe, learn_pe, patch_num, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                  pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                  store_attn=store_attn)

    def forward(self, x) -> Tensor:
        bs, patch_num, n_vars, _ = x.shape
        x = x.transpose(1, 2)

        u = torch.reshape(x, (bs * n_vars, patch_num, self.d_model))
        u = self.dropout(u + self.W_pos)

        z = self.encoder(u)
        z = torch.reshape(z, (-1, n_vars, patch_num, self.d_model))
        z = z.permute(0, 1, 3, 2)
        return z


class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, norm='BatchNorm', attn_dropout=0.,
                 dropout=0., activation='gelu', res_attention=False, n_layers=1,
                 pre_norm=False, store_attn=False):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                     attn_dropout=attn_dropout, dropout=dropout,
                                                     activation=activation, res_attention=res_attention,
                                                     pre_norm=pre_norm, store_attn=store_attn) for i in
                                     range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor, prefix_d: torch.Tensor = None):
        output = src
        scores = None
        if prefix_d is None:
            if self.res_attention:
                for mod in self.layers: output, scores = mod(output, prefix_d, prev=scores)
                return output
            else:
                for mod in self.layers: output = mod(output, prefix_d)
                return output
        else:
            if self.res_attention:
                for i, mod in enumerate(self.layers): output, scores = mod(output, prefix_d[i], prev=scores)
                return output
            else:
                for i, mod in enumerate(self.layers): output = mod(output, prefix_d[i])
                return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True,
                 activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        self.res_attention = res_attention
        self.self_attn = HarmonicAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout,
                                           proj_dropout=dropout, res_attention=res_attention)

        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = RMSNorm(d_model)

        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = RMSNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prefix_d=None, prev=None):
        if self.pre_norm:
            src = self.norm_attn(src)
        if prefix_d is None:
            if self.res_attention:
                src2, attn, scores = self.self_attn(src, src, src, prev)
            else:
                src2, attn = self.self_attn(src, src, src)
        else:
            K = torch.cat([prefix_d[0], src], dim=1)
            V = torch.cat([prefix_d[1], src], dim=1)
            if self.res_attention:
                src2, attn, scores = self.self_attn(src, K, V, prev)
            else:
                src2, attn = self.self_attn(src, K, V)
        if self.store_attn:
            self.attn = attn

        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        if self.pre_norm:
            src = self.norm_ffn(src)

        src2 = self.ff(src)

        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class TowerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                  pre_norm=pre_norm, activation=activation, res_attention=res_attention,
                                  n_layers=n_layers,
                                  store_attn=store_attn)

    def forward(self, x, prefix_d: torch.Tensor = None):
        bs, n_vars, d_model, patch_num = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(bs, n_vars * patch_num, d_model)
        x = self.encoder(x, prefix_d)
        x = x.reshape(bs, n_vars, patch_num, d_model)
        x = x.permute(0, 1, 3, 2)
        return x
