import re, os
from typing import Optional
import torch
from torch import nn
from layers.RevIN import RevIN
from layers.Olivia_EncDec import TSTEncoder, TowerEncoder,PredictionHead, PretrainHead
from layers.pos_encoding import positional_encoding



class LearnableOrthoTrans1D(nn.Module):
    """
    Learnable orthogonal transform along time axis for x: [B, T, 1].
    Provides:
      - encode(): remove heterogeneity (orthogonal transform)
      - decode(): restore to original space (inverse orthogonal transform)
    """
    def __init__(self, T, K=None, eps=1e-6, keep_k=None):
        super().__init__()
        self.T = T
        self.K = K if K is not None else T
        self.eps = eps
        self.keep_k = keep_k  # if not None -> lossy (not perfectly invertible)

        # Householder vectors: [K, T]
        self.v = nn.Parameter(torch.randn(self.K, self.T) * 0.02)

    def _build_Q(self):
        # Q is orthogonal by construction (product of Householder reflections)
        Q = torch.eye(self.T, device=self.v.device, dtype=self.v.dtype)
        for k in range(self.K):
            v = self.v[k]
            v = v / (v.norm(p=2) + self.eps)
            H = torch.eye(self.T, device=v.device, dtype=v.dtype) - 2.0 * torch.outer(v, v)
            Q = H @ Q
        return Q  # [T, T]


    def encode(self, x):
        """
        x: [B,T,1] in original (heterogeneous) space
        returns:
          z: [B,T,1] in orthogonal latent space (less heterogeneous)
          cache: dict for exact inverse (mu/std/Q and keep_k)
        """
        # self._check_input(x)
        Q = self._build_Q()  # [T,T]
        z = x.squeeze(-1) @ Q.t()   # [B,T]

        # optional low-rank truncation (lossy)
        if self.keep_k is not None:
            k = self.keep_k
            z = torch.cat([z[..., :k], torch.zeros_like(z[..., k:])], dim=1)

        z = z.unsqueeze(-1)  # [B,T,1]

        cache = {
            "Q": Q,           # [T,T]
            "keep_k": self.keep_k
        }
        return z, cache

    def decode(self, z, cache):
        """
        z: [B,T,1] in orthogonal latent space
        cache: returned by encode()
        returns:
          x_hat: [B,T,1] restored to original (heterogeneous) space
        """
        # self._check_input(z)
        if cache is None:
            Q = self._build_Q()  # [T,T]
        else:
            Q = cache["Q"]

        x0_hat = z.squeeze(-1) @ Q   # [B,T]
        x0_hat = x0_hat.unsqueeze(-1)

        return x0_hat

    def forward(self, x, return_cache=False):
        z, cache = self.encode(x)
        return (z, cache) if return_cache else z


class Model(nn.Module):
    def __init__(self, configs):

        super().__init__()

        assert configs.head_type in ['pretrain', 'prediction'], 'head type should be either pretrain or prediction'
        n_heads:int = 16
        head_dropout:float = 0.2
        individual:bool = False
        d_ff:int = 256
        norm:str = 'RMSNorm'
        attn_dropout:float = 0.
        dropout1:float = 0.
        dropout2:float = 0.2
        act:str = "silu"
        res_attention:bool = True
        pre_norm:bool = True
        store_attn:bool = False
        pe:str = 'zeros'
        learn_pe:bool = True
        self.freq_num:int = 1
        self.n_vars = configs.c_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        self.head_type = configs.head_type
        self.e_layers = configs.e_layers
        self.d_layers = configs.d_layers
        self.domain_len = configs.domain_len
        self.horizon_lengths = configs.horizon_lengths

        self.revin_layer_x = RevIN(self.n_vars, affine=True)

        # ortho
        self.ortho_trans = LearnableOrthoTrans1D(T=self.seq_len, K=self.seq_len//2)

        self.inverse_ortho_trans = [
            LearnableOrthoTrans1D(T=L, K=L//2)
            for L in self.horizon_lengths[1:]
        ]
        self.inverse_ortho_trans = nn.ModuleList(self.inverse_ortho_trans)
       
        # Projection
        self.projection_x = nn.Linear(self.seq_len, self.seq_len)

        # Patching
        self.patch_num = (max(self.seq_len, self.patch_len) - self.patch_len) // self.stride + 1
        tgt_len = self.patch_len  + self.stride * (self.patch_num - 1)
        self.s_begin = self.seq_len - tgt_len

        self.W_pos = positional_encoding(pe, learn_pe, self.patch_num * self.n_vars, self.d_model)
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        self.patch_embed_freq = nn.Linear(int(self.patch_len/2)+1, int(self.d_model/2)+1).to(torch.cfloat)

       
        # Encoder
        self.encoder = TSTEncoder(self.d_model, n_heads=n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout1, pre_norm=pre_norm, activation=act, res_attention=res_attention,
                                  n_layers=self.e_layers, store_attn=store_attn)

        self.decoder = TowerEncoder(self.d_model, n_heads=n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                dropout=dropout1, pre_norm=pre_norm, activation=act, res_attention=res_attention,
                                n_layers=self.d_layers, store_attn=store_attn)


        self.head = PretrainHead(self.d_model, self.patch_len, head_dropout)


        if configs.data == 'UTSD':
            setting = re.sub(r'_Olivia_', '_Olivia_CL_', configs.setting)
            print('loading pretrained encoder-decoder')
            state_dict = torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
           
            modules_to_load = {
                "module.ortho_trans": self.ortho_trans,
                "module.W_pos": self.W_pos,
                "module.patch_embed_freq": self.patch_embed_freq,
                "module.encoder": self.encoder,
                "module.decoder": self.decoder,
                "module.head": self.head,
            }

            for prefix, target in modules_to_load.items():
                if isinstance(target, torch.nn.Module):
                    module_state_dict = {
                        k.replace(f"{prefix}.", ""): v for k, v in state_dict.items() if k.startswith(prefix)
                    }
                    target.load_state_dict(module_state_dict, strict=True)
                else:
                    target.data.copy_(state_dict[prefix])

        # Frozen
        for param in [self.W_pos]:
            param.requires_grad = False
        for module in [self.patch_embed_freq, self.encoder, self.decoder, self.head]:
            for param in module.parameters():
                param.requires_grad = False


        # Head    
        pretrain_head_list = []
        for horizon_length in self.horizon_lengths:
            pretrain_head_list.append(PredictionHead(individual, self.n_vars, self.d_model, self.patch_num, horizon_length, head_dropout))
        self.pretrain_heads = nn.ModuleList(pretrain_head_list)
   
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x = self.revin_layer_x(x_enc, 'norm')

        # ortho trans
        x, _ = self.ortho_trans.encode(x)

        # projection 
        x = self.projection_x(x.permute(0, 2, 1)).permute(0, 2, 1)

        # patchify
        x = x[:, self.s_begin:, :]
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        bs, patch_num, n_vars, patch_len = x.shape

        # patch embedding
        x_fft = torch.fft.rfft(x, dim=-1)
        x_fft = self.patch_embed_freq(x_fft)
        x = torch.fft.irfft(x_fft, dim=-1, n=self.d_model)

        # pos embedding
        x = x.transpose(1, 2)
        u = torch.reshape(x, (-1, n_vars * patch_num, self.d_model))
        u = self.dropout1(u + self.W_pos)

        # encoder
        x = self.encoder(u)
        x = x.reshape(-1, n_vars, patch_num, self.d_model)
        x = x.permute(0, 1, 3, 2)

        # decoder
        x = self.decoder(x)

        # head
        y = [head(x) for head in self.pretrain_heads]
        for i in range(1, len(self.horizon_lengths)):
            y[i] = self.inverse_ortho_trans[i-1].decode(y[i], cache=None)

        y = [self.revin_layer_x(y_i, 'denorm') for y_i in y]

        x = self.head(x)
        x = x.reshape(bs, patch_num * patch_len, n_vars)

        x = self.revin_layer_x(x, 'denorm')
        return y, x


       