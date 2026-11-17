"""
ViT model for CIFAR-10 classification, i.e. representation transfer.
Replaces built-in ImageNet weights with fine-tuned weights for CIFAR-10.
This is a pure classification model and not object detection.
It probably has 86M parameters. Although latency is not good.
"""
import os
import time
from typing import Tuple

from absl import app
from absl import flags
from absl import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.models import ViT_B_16_Weights, vit_b_16
import torchvision.transforms as T

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "mode",
    "train",
    ["train", "eval"],
    "Execution mode: 'train' to fine-tune and save, 'eval' to load checkpoint and run inference.",
)
flags.DEFINE_string("data_dir", "./data", "Directory to store/load CIFAR-10.")
flags.DEFINE_string(
    "checkpoint_path",
    "",
    "Path to save/load checkpoint. Defaults to <data_dir>/vit_cifar.pt if empty.",
)
flags.DEFINE_integer("epochs", 5, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 128, "Batch size for training and evaluation.")
flags.DEFINE_float("lr", 1e-4, "Learning rate.")
flags.DEFINE_float("weight_decay", 1e-4, "Weight decay for AdamW.")
flags.DEFINE_integer("num_workers", 4, "Number of DataLoader worker processes.")
flags.DEFINE_integer("log_every_n_steps", 50, "Log training metrics every N steps.")
flags.DEFINE_bool(
    "freeze_backbone",
    False,
    "If True, freeze the ViT encoder and only train the linear classification head.",
)
flags.DEFINE_bool(
    "use_amp",
    True,
    "Use Automatic Mixed Precision (FP16/BF16) on CUDA to speed up training/inference.",
)


def get_dataloaders(
        weights: ViT_B_16_Weights,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        eval_only: bool = False,
) -> Tuple[DataLoader | None, DataLoader]:
    """Builds CIFAR-10 DataLoaders resized for ViT-B/16 (224x224)."""
    preprocess = weights.transforms()
    mean, std = preprocess.mean, preprocess.std
    pin_memory = torch.cuda.is_available()

    test_transform = T.Compose(
        [
            T.Resize((224, 224), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if eval_only:
        return None, test_loader

    train_transform = T.Compose(
        [
            T.Resize((224, 224), antialias=True),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def build_model(
        num_classes: int, freeze_backbone: bool = False, load_pretrained_backbone: bool = True
) -> Tuple[nn.Module, ViT_B_16_Weights]:
    """Builds ViT-B/16 and replaces the classification head."""
    weights = ViT_B_16_Weights.DEFAULT
    # Skip downloading ImageNet weights during eval since we load our own checkpoint
    model = vit_b_16(weights=weights if load_pretrained_backbone else None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    return model, weights


@torch.no_grad()
def evaluate_top1(
        model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool = False
) -> Tuple[float, float]:
    """Computes Top-1 accuracy (%) and total evaluation time (seconds)."""
    model.eval()
    correct = 0
    total = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)

        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    return 100.0 * correct / total, elapsed


def run_inference(ckpt_path: str, device: torch.device, use_amp: bool) -> None:
    """Loads a saved checkpoint and evaluates Top-1 accuracy on the test set."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    logging.info("Loading checkpoint from %s...", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    model, weights = build_model(num_classes=10, load_pretrained_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    _, test_loader = get_dataloaders(
        weights=weights,
        data_dir=FLAGS.data_dir,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        eval_only=True,
    )

    top1, eval_time = evaluate_top1(model, test_loader, device, use_amp=use_amp)
    saved_score = checkpoint.get("score", None)

    if saved_score is not None:
        logging.info(
            "Inference Complete | Test Top-1: %.2f%% (Checkpoint recorded score: %.2f%%) | Time: %.2fs",
            top1,
            saved_score,
            eval_time,
        )
    else:
        logging.info(
            "Inference Complete | Test Top-1: %.2f%% | Time: %.2fs",
            top1,
            eval_time,
        )


def run_training(ckpt_path: str, device: torch.device, use_amp: bool) -> None:
    """Fine-tunes ViT-B/16 on CIFAR-10 and saves the best checkpoint."""
    model, weights = build_model(
        num_classes=10,
        freeze_backbone=FLAGS.freeze_backbone,
        load_pretrained_backbone=True,
    )
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(
        "Loaded ViT-B/16 | Trainable params: %s / %s (freeze_backbone=%s)",
        f"{trainable_params:,}",
        f"{total_params:,}",
        FLAGS.freeze_backbone,
    )

    train_loader, test_loader = get_dataloaders(
        weights=weights,
        data_dir=FLAGS.data_dir,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        eval_only=False,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FLAGS.lr,
        weight_decay=FLAGS.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FLAGS.epochs * len(train_loader)
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_train_time = 0.0
    best_top1 = 0.0

    for epoch in range(1, FLAGS.epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()

        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for step, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * targets.size(0)
            train_correct += (outputs.argmax(dim=1) == targets).sum().item()
            train_total += targets.size(0)

            if step % FLAGS.log_every_n_steps == 0:
                logging.info(
                    "Epoch [%d/%d] Step [%d/%d] | Loss: %.4f | Train Top-1: %.2f%%",
                    epoch,
                    FLAGS.epochs,
                    step,
                    len(train_loader),
                    running_loss / train_total,
                    100.0 * train_correct / train_total,
                )

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start
        total_train_time += epoch_time

        val_top1, _ = evaluate_top1(model, test_loader, device, use_amp=use_amp)

        logging.info(
            "Epoch [%d/%d] Summary | Time: %.2fs | Train Loss: %.4f | Test Top-1: %.2f%%",
            epoch,
            FLAGS.epochs,
            epoch_time,
            running_loss / train_total,
            val_top1,
        )

        # Save whenever a new best Top-1 score is achieved
        if val_top1 > best_top1:
            best_top1 = val_top1
            os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "score": best_top1,
            }
            torch.save(checkpoint, ckpt_path)
            logging.info("Saved new best checkpoint (Top-1: %.2f%%) -> %s", best_top1, ckpt_path)

    logging.info(
        "Training Complete | Total Train Time: %.2fs | Best Test Top-1 Accuracy: %.2f%%",
        total_train_time,
        best_top1,
    )


def main(argv) -> None:
    del argv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = FLAGS.use_amp and device.type == "cuda"
    ckpt_path = FLAGS.checkpoint_path or os.path.join(FLAGS.data_dir, "vit_cifar.pt")

    logging.info("Mode: %s | Device: %s | AMP: %s", FLAGS.mode, device, use_amp)

    if FLAGS.mode == "eval":
        run_inference(ckpt_path=ckpt_path, device=device, use_amp=use_amp)
    else:
        run_training(ckpt_path=ckpt_path, device=device, use_amp=use_amp)


if __name__ == "__main__":
    app.run(main)
