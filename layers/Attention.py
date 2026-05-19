import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional

class HarmonicAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False,
        num_reps: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.U = nn.Parameter(torch.randn(n_heads, d_model, self.d_head) * (1.0 / (d_model ** 0.5)))

        if num_reps is None:
            self.num_reps = max(1, self.d_head // 4)
        else:
            self.num_reps = num_reps

        self.gamma = nn.Parameter(torch.tensor(-1.0))

        self.to_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(proj_dropout)
        )

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention

    def forward(self, Q, K=None, V=None, prev=None, key_padding_mask=None, attn_mask=None):
        if K is None:
            K = Q
        if V is None:
            V = Q

        Z = Q
        B, L, D = Z.shape
        H = self.n_heads
        P = self.d_head
        M = self.num_reps

        # initialize, avg pooling
        q_ini = Z.mean(dim=1, keepdim=True).expand(B, M, D)

        # project
        z_sub = torch.einsum('hdp,bld->bhpl', self.U, Z)
        q_sub = torch.einsum('hdp,bmd->bhpm', self.U, q_ini)

        # extraction
        attn_logits = torch.einsum('bhpl,bhpm->bhlm', z_sub, q_sub)

        # padding mask
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(-1)
            attn_logits = attn_logits.masked_fill(mask, float('-inf'))

        A = F.softmax(attn_logits, dim=2)
        A = self.attn_dropout(A)

        reps_sub = torch.einsum('bhpl,bhlm->bhpm', z_sub, A)

        #  Resonator self-attention
        resonator_logits = torch.einsum('bhpm,bhpn->bhmn', reps_sub, reps_sub)
        resonator_attn = F.softmax(resonator_logits, dim=-1)
        resonator_attn = self.attn_dropout(resonator_attn)
        resonator_attn = torch.einsum('bhpm,bhmn->bhpn', reps_sub, resonator_attn)

        # broadcast
        heads = torch.einsum('bhpm,bhml->bhpl', resonator_attn, A.transpose(-2, -1))

        full_heads = torch.einsum('hdp,bhpl->bhdl', self.U, heads)
        full_heads = full_heads.sum(dim=1)
        full_heads = full_heads.transpose(1, 2).contiguous()

        update = Z + self.gamma * full_heads
        output = self.to_out(update)
        attn_weights = torch.matmul(A, A.transpose(-2, -1))

        if self.res_attention:
            attn_scores = resonator_logits
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights
