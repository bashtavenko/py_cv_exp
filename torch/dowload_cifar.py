"""
Load CIFAR10 dataset if not already downloaded
Data is stored in pickle files
"""
import os

from matplotlib import pyplot as plt
from torchvision import datasets
from torchvision.datasets.utils import download_and_extract_archive

data_path = 'data/cifar10'

# Sometimes "download=True" fails
os.makedirs(data_path, exist_ok=True)
download_and_extract_archive(
    url="https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python"
        ".tar.gz",
    download_root=str(data_path),
    filename="cifar-10-python.tar.gz",
    md5="c58f30108f718f92721af3b95e74349a",
)
ds = datasets.CIFAR10(data_path, train=True, download=False)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'deer', 'horse', 'ship', 'truck']

fig = plt.figure(figsize=(8, 3))
num_classes = 10

for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    ax.set_title(class_names[i])
    # __getitem() returns (img, target)
    # img, label = ds[0]
    img = next(img for img, target in ds if target == i)
    plt.imshow(img)
plt.show()
