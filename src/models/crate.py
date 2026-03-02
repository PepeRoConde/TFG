import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., step_size=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        with torch.no_grad():
            init.kaiming_uniform_(self.weight)
        self.step_size = step_size
        self.lambd = 0.1

    def forward(self, x):
        # compute D^T * D * x
        x1 = F.linear(x, self.weight, bias=None)
        grad_1 = F.linear(x1, self.weight.t(), bias=None)
        # compute D^T * x
        grad_2 = F.linear(x, self.weight.t(), bias=None)
        # compute negative gradient update: step_size * (D^T * x - D^T * D * x)
        grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd

        output = F.relu(x + grad_update)
        return output


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., order='first'):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.order = order  # 'first' or 'second'

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.qkv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)

        # Compute (U^T Z)^T (U^T Z)
        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale

        if self.order == 'first':
            # First-order Neumann approximation
            # out = (U^T Z) * softmax((U^T Z)^T (U^T Z))
            attn = self.attend(dots)
            attn = self.dropout(attn)
            out = torch.matmul(attn, w)
        
        elif self.order == 'second':
            # Second-order Neumann approximation
            # out = out_1st - out_2nd
            
            # First order term: (U^T Z) * softmax((U^T Z)^T (U^T Z))
            attn_1st = self.attend(dots)
            attn_1st = self.dropout(attn_1st)
            out_1st = torch.matmul(attn_1st, w)
            
            # Second order term: (U^T Z) * softmax(((U^T Z)^T (U^T Z))^2)
            # Compute ((U^T Z)^T (U^T Z))^2
            dots_2nd = torch.matmul(dots, dots.transpose(-1,-2))
            attn_2nd = self.attend(dots_2nd)
            attn_2nd = self.dropout(attn_2nd)
            out_2nd = torch.matmul(attn_2nd, w)
            
            # Combine: subtract second order correction
            out = out_1st - out_2nd
        
        else:
            raise ValueError(f"order must be 'first' or 'second', got {self.order}")

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0., ista=0.1, order='first', shared_dict=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.depth = depth
        self.dim = dim
        self.order = order
        
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, order=order)),
                        PreNorm(dim, FeedForward(dim, dim, dropout=dropout, step_size=ista))
                    ]
                )
            )

        if shared_dict:
            self.weight = nn.Parameter(torch.Tensor(dim, dim))

            for i in range(depth):
                print(f'-> diccionario capa {i} : {self.layers[i][1].fn.weight}') # _1_ es la segunda PreNorm, _fn_ es el FeedForward
                self.layers[i][1].fn.weight = self.weight
                print(f'-> diccionario capa {i} : {self.layers[i][1].fn.weight}')

    def forward(self, x):
        depth = 0
        for attn, ff in self.layers:
            grad_x = attn(x) + x
            x = ff(grad_x)
        return x


class CRATE(nn.Module):
    def __init__(
            self, *, image_size, patch_size, num_classes, dim, depth, heads, pool='cls', channels=3, dim_head=64,
            dropout=0., emb_dropout=0., ista=0.1, order='first', shared_dict=False, no_pos=False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.no_pos = no_pos
        if not self.no_pos: self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, dropout, ista=ista, order=order, shared_dict=shared_dict)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        if not self.no_pos: x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        feature_pre = x
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
