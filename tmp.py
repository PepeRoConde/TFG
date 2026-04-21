import matplotlib.pyplot as plt
import numpy as np
import sys

import torch
from einops import rearrange

from src.data.Online_Dataset import Online_Dataset

from src.utils.load_model import load_model

with torch.no_grad():
    patch_size = 105
    token_size = 15

    model = load_model(
        "data/weights/6efb21.pth.tar",
        #'data/weights/preentrenados/demo_01.pth.tar',
        "CRATE_enana",
        #'CRATE_base_demo',
        patch_size=patch_size,
        token_size=token_size,
        shared_u=False,
        shared_dict=False,
    )

    model.eval()

    ln1 = model.to_patch_embedding[1]  # LayerNorm(675)
    W = model.to_patch_embedding[2].weight  # (192, 675)
    W_pinv = torch.linalg.pinv(W)  # (patch_dim, dim)
    b = model.to_patch_embedding[2].bias  # (192,)
    ln2 = model.to_patch_embedding[3]  # LayerNorm(192)

    D = model.transformer.layers[0][1].fn.weight  # la capa 0, el FeedForward
    # >>> D.shape
    # torch.Size([192, 192])
    #
    #  como se usa en nn.functional.linear(), realmente se esta trasponiendo
    #  así que realmente ocurre que Z = Z @ Dt; por tanto las filas de D son
    #  las columnas de el diccionario que tenemos en la cabeza

    dataset = Online_Dataset("data/DRIVE/train", patch_size, aumento_datos=False)

    img_idx = sys.argv[1] if len(sys.argv) > 1 else 530
    img = dataset[int(img_idx)][0].unsqueeze(
        0
    )  # la 110-esima imagen (1 es la etiqueta)
    # >>> img.shape
    # torch.Size([1, 3, 75, 75])

    plt.imsave(
        f"data/plots/invertible/original_{img_idx}.png",
        rearrange(img[0], "c h w -> h w c").numpy(),
    )

    img_rsh = rearrange(
        img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=token_size, p2=token_size
    )

    # LayerNorm 1
    mu1 = img_rsh.mean(dim=-1, keepdim=True)
    sigma1 = img_rsh.std(dim=-1, keepdim=True, correction=0)
    img_rsh = (img_rsh - mu1) / (sigma1 + ln1.eps) * ln1.weight + ln1.bias

    # Linear
    img_rsh = torch.nn.functional.linear(img_rsh, W, b)

    # LayerNorm 2
    mu2 = img_rsh.mean(dim=-1, keepdim=True)
    sigma2 = img_rsh.std(dim=-1, keepdim=True, correction=0)
    Zish = (img_rsh - mu2) / (sigma2 + ln2.eps) * ln2.weight + ln2.bias

    Zish = torch.cat((model.cls_token, Zish), dim=1)
    # Zish += model.pos_embedding

    Zl = model.transformer(Zish)

    # >>> Zish.shape
    # torch.Size([1, 25, 192])

    vector = Zish[0][0]  # el vector de embedings asociado al primer parche | (dim,)

    patch = torch.nn.functional.linear(vector - b, W_pinv)  # (patch_dim,)

    def to_plot(img):
        if len(img.shape) == 4:
            img = img[0].numpy()
        if len(img.shape) == 3:
            img = img.numpy()
        if len(img.shape) == 1:
            img = rearrange(img.numpy(), "(h w c) -> h w c", c=3, h=15, w=15)
        return (img - img.min()) / (img.max() - img.min())

    img = to_plot(patch)
    plt.imsave("data/plots/invertible/img.png", img)

    def plot_Z_grid(
        Z,
        filename=f"data/plots/invertible/Z_grid_{img_idx}.png",
        patch_h=15,
        patch_w=15,
        c=3,
        grid_h=7,
        grid_w=None,
    ):
        if len(Z.shape) == 3:
            Z = Z[0]

        if grid_w is None:
            grid_w = grid_h

        patches = []

        print(f"shape {Z.shape}")
        print(f"cls token {Z[0].numpy()}")
        plt.bar(range(len(Z[0].numpy())), Z[0].numpy())
        plt.savefig(filename.replace("grid", "cls_token"))
        plt.clf()

        for i in range(grid_h * grid_w):
            vector = Z[i + 1]  # +1 por el cls token

            x2_hat = (vector - ln2.bias) / ln2.weight
            x2_hat = x2_hat * (sigma2[0, i] + ln2.eps) + mu2[0, i]
            x1_hat = torch.nn.functional.linear(x2_hat - b, W_pinv)
            x0_hat = (x1_hat - ln1.bias) / ln1.weight
            patch = x0_hat * (sigma1[0, i] + ln1.eps) + mu1[0, i]

            patches.append(
                rearrange(
                    patch.numpy(), "(p1 p2 c) -> p1 p2 c", p1=patch_h, p2=patch_w, c=c
                )
            )

        rows = [
            np.concatenate(patches[row * grid_w : (row + 1) * grid_w], axis=1)
            for row in range(grid_h)
        ]
        image = np.concatenate(rows, axis=0)
        image = (image - image.min()) / (image.max() - image.min())
        plt.imsave(filename, image)

    plot_Z_grid(
        Zish,
        f"data/plots/invertible/Z_grid_{img_idx}.png",
        grid_h=patch_size // token_size,
    )
    plot_Z_grid(
        Zl,
        f"data/plots/invertible/Zl_grid_{img_idx}.png",
        grid_h=patch_size // token_size,
    )
    plot_Z_grid(
        D,
        f"data/plots/invertible/D_grid_{img_idx}.png",
        grid_h=patch_size // token_size,
    )

    print("fin :)")
#
#
#    Z = model.to_patch_embedding(img)
#    #>>> Z.shape
#    #torch.Size([1, 25, 192])   # tokens, dim # 25 = (75/15)^2 ; dim = 192
#
#    Z = torch.cat((model.cls_token, Z), dim=1)
#
#    Z += model.pos_embedding[:, : (25 + 1)]
#
#    Z = model.transformer.layers[0][0](Z) # atencion
#
#
#
#
#    # -------------
#
#
#
#    def to_plot(img):
#        if len(img.shape) == 4:
#            img = rearrange(img[0].numpy(),'c h w -> h w c')
#        if len(img.shape) == 3:
#            img = rearrange(img.numpy(),'c h w -> h w c')
#        if len(img.shape) == 1:
#            img = rearrange(img.numpy(),'(c h w) -> h w c')
#        plt.imsave('data/plots/invertible/img.png', img)
#
#    # ----
#
#    vector = D[0]
#
#
#    W = model.to_patch_embedding[2].weight
#    #>>> W.shape
#    #torch.Size([192, 675])     # dim, patch_dim (675 = 15*15*3)
#    W_pinv = torch.linalg.pinv(W)
#
#    W = model.to_patch_embedding[2].weight      # (dim, patch_dim)
#    b = model.to_patch_embedding[2].bias        # (dim,)
#
#    W_pinv = torch.linalg.pinv(W)              # (patch_dim, dim)
#
#    patch = (vector - b) @ W_pinv.T
#
#    patch = torch.nn.functional.linear(vector - b, W_pinv)
#
#    patch = rearrange(patch,'(c h w) -> h w c', h=15,w=15,c=3)
#    patch = (patch - patch.min()) / (patch.max() - patch.min())
#    plt.imsave('data/plots/invertible/vector.png', patch.detach().numpy())
#
#
#    def atom_to_patch(vector, W, b, W_pinv, h=15, w=15, c=3):
#        """Map a 192-dim dictionary vector back to (h,w,c) pixel space."""
#        # Undo the linear: z = x @ W^T + b  =>  x ≈ (z - b) @ W_pinv^T
#        pixel = (vector - b) @ W_pinv.T
#        patch = rearrange(pixel, '(c h w) -> h w c', h=h, w=w, c=c)
#        patch = (patch - patch.min()) / (patch.max() - patch.min())
#        return patch.detach().numpy()
#
#    def plot_atom_grid(D, W, b, W_pinv, title, grid=(14,14), patch_hw=(15,15)):
#        """Plot all atoms in a grid. D is (192,192)."""
#        rows, cols = grid
#        n = rows * cols  # 196, we have 192 atoms so 4 cells will be empty
#        ph, pw = patch_hw
#
#        fig, axes = plt.subplots(rows, cols, figsize=(cols*1.2, rows*1.2))
#        fig.suptitle(title, fontsize=14)
#
#        for i, ax in enumerate(axes.flat):
#            ax.axis('off')
#            if i < D.shape[0]:
#                vec = D[i]  # row i  — swap to D[:, i] for columns
#                patch = atom_to_patch(vec, W, b, W_pinv)
#                ax.imshow(patch)
#
#        plt.tight_layout()
#        return fig
#
#    # --- load weights ---
#    W     = model.to_patch_embedding[2].weight   # (192, 675)
#    b     = model.to_patch_embedding[2].bias     # (192,)
#    W_pinv = torch.linalg.pinv(W)               # (675, 192)
#
#    # get D from layer 0 (or loop over layers)
#    layer_idx = 0
#    D = model.transformer.layers[layer_idx][1].fn.weight  # (192, 192)
#
#    # --- plot rows ---
#    DZ = torch.nn.functional.linear(Z,D)
#
#    fig_rows = plot_atom_grid(DZ, W, b, W_pinv, title=f'Layer {layer_idx} — D rows (analysis atoms)')
#    fig_rows.savefig(f'data/plots/invertible/layer{layer_idx}_D_rows.png', dpi=150, bbox_inches='tight')
#
#    # --- plot columns ---
#    fig_cols = plot_atom_grid(D.T, W, b, W_pinv, title=f'Layer {layer_idx} — D cols (synthesis atoms)')
#    fig_cols.savefig(f'data/plots/invertible/layer{layer_idx}_D_cols.png', dpi=150, bbox_inches='tight')
#
