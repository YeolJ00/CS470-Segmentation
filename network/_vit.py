import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# https://github.com/lucidrains/vit-pytorch

__all__ = ["ViT"]
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    """
    Input Tensor : (Batch, In-Channel, Height, Weight)
    Output Tensor: (Batch, Out-Channel, Height, Weight)
    """
    def __init__(self, image_size, patch_size, in_channels, out_channels, dim = 1024, depth = 6, heads = 16, mlp_dim = 2048, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.image_size = pair(image_size)
        image_height, image_width = self.image_size
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_h_patches = image_height // patch_height
        num_w_patches = image_width // patch_width
        num_patches = num_h_patches * num_w_patches
        patch_dim = in_channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', h = num_h_patches, ph = patch_height, w = num_w_patches, pw = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        restore_dim = out_channels * patch_height * patch_width
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(dim, restore_dim),
            Rearrange('b (h w) (ph pw c) -> b c (h ph) (w pw)', h = num_h_patches, ph = patch_height, w = num_w_patches, pw = patch_width),
        )

    def forward(self, img):
        input_img_size = img.shape[2:]
        if input_img_size[0] != self.image_size :
            img = F.interpolate(img, size=self.image_size, mode='bilinear', align_corners=False)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        out_img = self.from_patch_embedding(x)
        if input_img_size[0] != self.image_size :
            out_img = F.interpolate(out_img, size=input_img_size, mode='bilinear', align_corners=False)
        return out_img

    def __repr__(self) :
        return "Vision Transformer(Modified): (batch, in_channel, height, weight) -> (batch, out_channel, height, weight)"

if __name__ == '__main__' :
    v = ViT(
        image_size = 256,
        patch_size = 32,
        in_channels = 3,
        out_channels = 2,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    img = torch.randn(1, 3, 256, 256)
    preds = v(img)
    print(f'{img.shape} -> {preds.shape}')

    img = torch.randn(1, 3, 240, 240)
    preds = v(img)
    print(f'{img.shape} -> {preds.shape}')
