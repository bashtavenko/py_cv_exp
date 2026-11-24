"""
Toy example of fully connected neural network with CIFAR10 dataset.
nn.Module subclass with an explicit forward pass.
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


class FullyConnectedNet(nn.Module):
    """
    Fully connected network with one hidden layer.
    Input: batch of images (batch_size, 3, 32, 32).
    Output: log-probabilities of each class (batch_size, 10).
    """

    def __init__(self, in_features=3072, hidden_features=512, n_classes=10):
        super().__init__()  # Required: registers parameters and submodules
        # Layers assigned as attributes are registered automatically,
        # so model.parameters() will find their weights and biases.
        self.fc1 = nn.Linear(in_features, hidden_features)  # 3072 = 3 * 32 * 32
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(hidden_features, n_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x.shape: (batch_size, 3, 32, 32)
        # Flattens each image, keeping the batch dimension: (batch_size, 3072)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)  # (batch_size, 512)
        x = self.activation(x)  # (batch_size, 512)
        x = self.fc2(x)  # (batch_size, 10)
        return self.log_softmax(x)  # (batch_size, 10)


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
    # Get mean and std
    # Adds new image axis to the forth position (3, 32, 32, 50000)
    images = torch.stack([img for img, _ in train_dataset], dim=3)
    # images.view(3, -1) flattens the image tensor to a 2D
    # tensor with shape (3, 32*32*50000)
    mean = images.view(3, -1).mean(dim=1)  # Get mean for each channel
    std = images.view(3, -1).std(dim=1)  # Get std for each channel
    logging.info(f"Mean: {mean} std: {std}")

    # Swap transform that includes normalization.
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

    model = FullyConnectedNet()

    n_epochs = 10
    learning_rate = 1e-2
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.NLLLoss()

    model.train()
    for epoch in range(n_epochs):
        for img, label in train_loader:
            # img.shape: (64, 3, 32, 32), flattening happens in forward()
            out = model(img)
            loss = loss_fn(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logging.info(f"Epoch: {epoch}, Loss: {loss.item():.2f}")

    # Metrics
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for img, label in val_loader:
            outputs = model(img)
            _, predicted = torch.max(outputs, dim=1)
            total += label.shape[0]
            correct += int((predicted == label).sum())

    logging.info(f"Accuracy: {correct / total:.2%}")


if __name__ == '__main__':
    app.run(main)
