"""
Toy example of fully connected neural network with CIFAR10 dataset.
Accuracy as expected, horrible: 40%.
Assuming that CIFAR10 is already downloaded.
"""
import torch
from absl import app, flags, logging
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

FLAGS = flags.FLAGS

flags.DEFINE_string("images_dir", "data/cifar10", "Path to the CIFAR10")


def get_datasets():
    """
    Returns train and validation datasets with normalization
    """
    # train_dataset shape: (3, 32, 32))
    train_dataset = datasets.CIFAR10(FLAGS.images_dir, train=True,
                                     download=False,
                                     transform=transforms.ToTensor())
    val_dataset = datasets.CIFAR10(FLAGS.images_dir, train=False,
                                   download=False,
                                   transform=transforms.ToTensor())
    # Get min and std
    # Adds new image axis to the forth position (3, 32, 32, 50000)
    images = torch.stack([img for img, _ in train_dataset], dim=3)
    # images.view(3, -1) flattens the image tensor to a 2D
    # tensor with shape (3, 32*32*50000)
    mean = images.view(3, -1).mean(dim=1)  # Get min for each channel
    std = images.view(3, -1).std(dim=1)  # Get std for each channel
    logging.info(f"Mean: {mean} std: {std}")

    # Swap transform that includes normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_dataset.transform = transform
    val_dataset.transform = transform
    return train_dataset, val_dataset


def main(_):
    train_dataset, val_dataset = get_datasets()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = nn.Sequential(
        nn.Linear(3072, 512),  # 3072 = 3 * 32 * 32
        nn.Tanh(),
        nn.Linear(512, 10),
        nn.LogSoftmax(dim=1))

    n_epochs = 10
    learning_rate = 1e-2
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.NLLLoss()

    for epoch in range(n_epochs):
        for img, label in train_loader:
            # imge.shape: (64, 3, 32, 32)
            batch_size = img.shape[0]
            # Flattens each image to a 2D tensor (64, 3072)
            img_tensor = img.view(batch_size, -1)
            out = model(img_tensor)
            loss = loss_fn(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logging.info(f"Epoch: {epoch}, Loss: {loss.item():.2f}")

    # Metrics
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64,
                                             shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for img, label in val_loader:
            outputs = model(img.view(img.shape[0], -1))
            _, predicted = torch.max(outputs, dim=1)
            total += label.shape[0]
            correct += int((predicted == label).sum())

    logging.info(f"Accuracy: {correct / total:.2%}")


if __name__ == '__main__':
    app.run(main)
