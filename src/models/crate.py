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
    def __init__(self, dim, hidden_dim, dropout=0.0, step_size=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        with torch.no_grad():
            init.kaiming_uniform_(self.weight)
        self.step_size = step_size
        self.lambd = 0.1

    def forward(self, x):
        x1 = F.linear(x, self.weight, bias=None)
        grad_1 = F.linear(x1, self.weight.t(), bias=None)
        grad_2 = F.linear(x, self.weight.t(), bias=None)
        grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd

        output = F.relu(x + grad_update)
        return output


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        project_dim=None,
        dropout=0.0,
        order="first",
        linformer=False,
        share_proj="none",
        seq_len=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.order = order
        self.linformer = linformer
        self.share_proj = share_proj  # 'none', 'headwise', 'key-value', 'layerwise'

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        if self.linformer:
            assert seq_len is not None, "seq_len must be provided when linformer=True"
            self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        else:
            self.qkv = nn.Linear(dim, inner_dim, bias=False)

        self.project_dim = project_dim or dim_head * heads

        if self.linformer:
            n = seq_len

            if share_proj == "none":
                self.E = nn.Parameter(torch.randn(self.heads, n, self.project_dim))
                self.F = nn.Parameter(torch.randn(self.heads, n, self.project_dim))
            elif share_proj == "headwise":
                self.E = nn.Parameter(torch.randn(n, self.project_dim))
                self.F = nn.Parameter(torch.randn(n, self.project_dim))
            elif share_proj == "key-value":
                self.E = nn.Parameter(torch.randn(n, self.project_dim))
                self.F = self.E
            elif share_proj == "layerwise":
                self.E = nn.Parameter(torch.randn(1, n, self.project_dim))
                self.F = nn.Parameter(torch.randn(1, n, self.project_dim))
            else:
                raise ValueError(f"Invalid share_proj value: {share_proj}")

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, return_attention=False):
        qkv = self.qkv(x)

        if self.linformer:
            q, k, v = rearrange(
                qkv, "b n (three h d) -> three b h n d", three=3, h=self.heads
            )

            if self.share_proj == "none":
                k_proj = torch.einsum("b h n d, h n k -> b h k d", k, self.E)
                v_proj = torch.einsum("b h n d, h n k -> b h k d", v, self.F)
            elif self.share_proj in ["headwise", "key-value"]:
                k_proj = torch.einsum("b h n d, n k -> b h k d", k, self.E)
                v_proj = torch.einsum("b h n d, n k -> b h k d", v, self.F)
            elif self.share_proj == "layerwise":
                k_proj = torch.einsum("b h n d, o n k -> b h k d", k, self.E)
                v_proj = torch.einsum("b h n d, o n k -> b h k d", v, self.F)

            # dots = Q @ (E K)^T  ->  (b, h, n, d) @ (b, h, d, k) = (b, h, n, k)
            dots = torch.matmul(q, k_proj.transpose(-1, -2)) * self.scale

        else:
            w = rearrange(qkv, "b n (h d) -> b h n d", h=self.heads)
            dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale  # (b, h, n, n)

        if self.order == "first":
            attn = self.attend(dots)
            # aqui se plotea
            if return_attention:
                return attn
            attn = self.dropout(attn)
            if self.linformer:
                # (b, h, n, k) @ (b, h, k, d) = (b, h, n, d)
                out = torch.matmul(attn, v_proj)
            else:
                out = torch.matmul(attn, w)

        elif self.order == "second":
            # ---- First-order term ----
            attn_1st = self.attend(dots)
            attn_1st = self.dropout(attn_1st)
            if self.linformer:
                out_1st = torch.matmul(attn_1st, v_proj) * self.scale  # (b, h, n, d)
            else:
                out_1st = torch.matmul(attn_1st, w) * self.scale

            # dots @ dots^T da (b, h, n, n)
            dots_2nd = torch.matmul(dots, dots.transpose(-1, -2))  # (b, h, n, n)
            attn_2nd = self.attend(dots_2nd)
            # aqui haberia que plotear, para la segmentacion emrgente
            if return_attention:
                return attn_2nd
            attn_2nd = self.dropout(attn_2nd)
            if self.linformer:
                # attn_2nd es (b,h,n,n); muliplicamos con attn_1st (b,h,n,k) para conseguir (b,h,n,k)
                attn_2nd_k = torch.matmul(attn_2nd, attn_1st)  # (b, h, n, k)
                out_2nd = torch.matmul(attn_2nd_k, v_proj)  # (b, h, n, d)
            else:
                out_2nd = torch.matmul(attn_2nd, w)

            out = out_1st - out_2nd

        else:
            raise ValueError(f"order must be 'first' or 'second', got {self.order}")

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        dropout=0.0,
        ista=0.1,
        order="first",
        shared_u=False,
        shared_dict=False,
        linformer=False,
        project_dim=None,
        shared_proj="none",
        seq_len=None,
    ):
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
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                order=order,
                                linformer=linformer,
                                project_dim=project_dim,
                                share_proj=shared_proj,
                                seq_len=seq_len,
                            ),
                        ),
                        PreNorm(
                            dim, FeedForward(dim, dim, dropout=dropout, step_size=ista)
                        ),
                    ]
                )
            )

        if shared_u:
            if linformer:
                assert (
                    seq_len is not None
                ), "seq_len must be provided when linformer=True"
                self.qkv = nn.Linear(dim, dim_head * heads * 3, bias=False)
            else:
                self.qkv = nn.Linear(dim, dim_head * heads, bias=False)

            for i in range(depth):
                self.layers[i][1].fn.qkv = self.qkv

        if shared_dict:
            self.weight = nn.Parameter(torch.Tensor(dim, dim))
            for i in range(depth):
                self.layers[i][1].fn.weight = self.weight

    def forward(self, x):
        for attn, ff in self.layers:
            grad_x = attn(x) + x
            x = ff(grad_x)
        return x


class CRATE(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        ista=0.1,
        order="first",
        shared_u=False,
        shared_dict=False,
        no_pos=False,
        linformer=False,
        project_dim=None,
        shared_proj="none",
        gain=1.0,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        seq_len = num_patches + 1  # cls

        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        # Set gain on patch embedding linear layer
        init.xavier_uniform_(self.to_patch_embedding[2].weight, gain=gain)
        self.no_pos = no_pos
        if not self.no_pos:
            self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            dropout,
            ista=ista,
            order=order,
            shared_u=shared_u,
            shared_dict=shared_dict,
            linformer=linformer,
            project_dim=project_dim,
            shared_proj=shared_proj,
            seq_len=seq_len,
        )

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

    def get_last_selfattention(self, img, layer=5):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        for i, (attn, ff) in enumerate(self.transformer.layers):
            if i < layer:
                grad_x = attn(x) + x
                x = ff(grad_x)
            else:
                attn_map = attn(x, return_attention=True)
                return attn_map
