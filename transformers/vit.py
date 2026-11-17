"""
ViT is not limited to the next data; it can be applied to any data that can be
 represented as a sequence of tokens.
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

torch.manual_seed(0)

# Show original image with resizing
img_path = "data/tree.jpg"
img_size = 224  # ViT-B default
patch_size = 16  # ViT-B default
img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
# plt.imshow(img)
# plt.show()

# Show patches
to_tensor = transforms.ToTensor()
x = to_tensor(img).unsqueeze(0)  # [1,3,H,W]
B, C, H, W = x.shape
assert H % patch_size == 0 and W % patch_size == 0

rows = H // patch_size
cols = W // patch_size
N = rows * cols

# Extract non-overlapping patches -> [B, N, D] where D = C * patch_size * patch_size
unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
patches = unfold(x)  # [B, C*ps*ps, N]
patches = patches.transpose(1, 2)  # [B, N, C*ps*ps]

# Make the flattened token sequence panel larger
fig, (axL, axR) = plt.subplots(
    1, 2,
    figsize=(14, 4.8),  # was (12, 4)
    gridspec_kw={'width_ratios': [1.0, 1.9]}  # give more room to the right panel
)

# Left: original image with grid and patch indices (row-major)
axL.imshow(img)
for x in range(0, img_size, patch_size):
    axL.axvline(x, color='w', lw=0.5, alpha=0.6)
for y in range(0, img_size, patch_size):
    axL.axhline(y, color='w', lw=0.5, alpha=0.6)
for r in range(rows):
    for c in range(cols):
        i = r * cols + c
        axL.text(c * patch_size + patch_size / 2,
                 r * patch_size + patch_size / 2,
                 str(i), color='yellow', fontsize=6, ha='center', va='center')
axL.axis('off')

# Right: 1D flattened sequence as a long strip of patches
patches_CHW = patches[0].reshape(N, C, patch_size, patch_size)
strip = torch.cat([patches_CHW[i] for i in range(N)], dim=2)
axR.imshow(strip.permute(1, 2, 0).clamp(0, 1))
for x in range(0, N * patch_size + 1, patch_size):
    axR.axvline(x - 0.5, color='w', lw=0.5, alpha=0.6)
axR.axhline(-0.5, color='w', lw=0.5, alpha=0.6)
axR.axhline(patch_size - 0.5, color='w', lw=0.5, alpha=0.6)
axR.axis('off')

plt.tight_layout()
plt.show()
