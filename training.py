
import copy
from collections.abc import Iterable

import torch
from torch import nn
from torch.optim import Optimizer
from tqdm.auto import tqdm


def train(
    model: nn.Module,
    train_iterator: Iterable,
    valid_iterator: Iterable,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    desc: str = "Training",
    max_epochs: int = 50,
    patience: int = 3,
    min_delta: float = 0.0,
) -> tuple[dict[str, list[float]], dict[str, torch.Tensor]]:
    """Train a classifier with validation-loss-based early stopping.

    Supports binary and multiclass single-label classification.

    Args:
        model: Model returning logits shaped [batch_size, num_classes].
        train_iterator: Iterable yielding (X, y) training batches.
        valid_iterator: Iterable yielding (X, y) validation batches.
        optimizer: Optimizer used to update model parameters.
        loss_fn: Loss function, typically nn.CrossEntropyLoss().
        device: Device used for training and evaluation.
        desc: Description displayed in the epoch progress bar.
        max_epochs: Maximum number of training epochs.
        patience: Number of consecutive epochs without sufficient
            validation-loss improvement before stopping.
        min_delta: Minimum decrease in validation loss required to
            count as an improvement.

    Returns:
        A tuple containing:
            - results: Training and validation loss and accuracy histories.
            - best_model_state_dict: Model weights from the epoch with
              the lowest validation loss.

    Note:
        The model is restored to its best checkpoint before returning.
    """
    if max_epochs < 1:
        raise ValueError("max_epochs must be at least 1.")
    if patience < 1:
        raise ValueError("patience must be at least 1.")
    if min_delta < 0:
        raise ValueError("min_delta cannot be negative.")

    device = torch.device(device)
    model.to(device)

    results = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
    }

    best_valid_loss = float("inf")
    best_model_state_dict = None
    epochs_without_improvement = 0

    for epoch in tqdm(range(max_epochs), desc=desc):
        train_loss, train_acc = train_step(
            model=model,
            iterator=train_iterator,
            epoch=epoch,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        valid_loss, valid_acc = valid_step(
            model=model,
            iterator=valid_iterator,
            epoch=epoch,
            loss_fn=loss_fn,
            device=device,
        )

        # Record metrics for this epoch
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["valid_loss"].append(valid_loss)
        results["valid_acc"].append(valid_acc)

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"valid_loss: {valid_loss:.4f} | "
            f"valid_acc: {valid_acc:.4f}"
        )

        # Save an independent copy whenever validation loss improves
        if valid_loss < best_valid_loss - min_delta:
            best_valid_loss = valid_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Stop if validation loss has not improved for 'patience' epochs
        if epochs_without_improvement >= patience:
            print(
                "Early stopping: validation loss did not improve "
                f"for {patience} consecutive epochs."
            )
            break

    return results, best_model_state_dict


def train_step(
    model: nn.Module,
    iterator: Iterable,
    epoch: int,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch and return average loss and accuracy."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        iterator,
        desc=f"Training Epoch {epoch + 1}",
        leave=False,
    )

    for X, y in progress_bar:
        X = X.to(device)
        y = y.to(device).long()

        # Forward pass: produce one logit per class
        logits = model(X)
        loss = loss_fn(logits, y)

        # Calculate gradients and update model parameters
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size

        # Choose the class with the highest logit for each sample
        predictions = logits.detach().argmax(dim=1)
        total_correct += (predictions == y).sum().item()
        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            accuracy=f"{(predictions == y).float().mean().item():.4f}",
        )

    if total_samples == 0:
        raise ValueError("The training iterator contains no samples.")

    return total_loss / total_samples, total_correct / total_samples


def valid_step(
    model: nn.Module,
    iterator: Iterable,
    epoch: int,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate one epoch without calculating gradients."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        iterator,
        desc=f"Validation Epoch {epoch + 1}",
        leave=False,
    )

    with torch.inference_mode():
        for X, y in progress_bar:
            X = X.to(device)
            y = y.to(device).long()

            logits = model(X)
            loss = loss_fn(logits, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size

            # The same prediction rule works for any number of classes
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == y).sum().item()
            total_samples += batch_size

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                accuracy=f"{(predictions == y).float().mean().item():.4f}",
            )

    if total_samples == 0:
        raise ValueError("The validation iterator contains no samples.")

    return total_loss / total_samples, total_correct / total_samples