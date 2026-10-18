
from collections.abc import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm


def plot_loss_curves(results: dict) -> tuple:
    """Plot training and validation loss and accuracy across epochs.

    Args:
        results: Dictionary containing train_loss, valid_loss,
            train_acc, and valid_acc lists.

    Returns:
        A tuple containing the figure and its axes.
    """
    train_loss = results["train_loss"]
    valid_loss = results["valid_loss"]
    train_acc = results["train_acc"]
    valid_acc = results["valid_acc"]

    # Start epochs at 1 rather than 0
    epochs = range(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Plot loss
    axes[0].plot(epochs, train_loss, label="Train Loss")
    axes[0].plot(epochs, valid_loss, label="Validation Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Plot accuracy
    axes[1].plot(epochs, train_acc, label="Train Accuracy")
    axes[1].plot(epochs, valid_acc, label="Validation Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    return fig, axes


def display_pred_results(
    model: nn.Module,
    test_iterator: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a binary classifier and return metrics and a confusion matrix.

    Assumes class 0 is negative and class 1 is positive.

    Args:
        model: Trained classification model returning logits shaped
            [batch_size, num_classes].
        test_iterator: Iterable yielding (X, y) batches.
        device: Device on which to run evaluation.

    Returns:
        A tuple containing a metrics DataFrame and a confusion matrix
        DataFrame.
    """
    device = torch.device(device)
    model.to(device)
    model.eval()

    # Accumulate counts across the entire test set before computing metrics
    total_tn = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.inference_mode():
        for X, y in tqdm(test_iterator, desc="Evaluating"):
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            preds = logits.argmax(dim=1)

            total_tn += ((preds == 0) & (y == 0)).sum().item()
            total_tp += ((preds == 1) & (y == 1)).sum().item()
            total_fp += ((preds == 1) & (y == 0)).sum().item()
            total_fn += ((preds == 0) & (y == 1)).sum().item()

    total_samples = total_tn + total_tp + total_fp + total_fn

    if total_samples == 0:
        raise ValueError("The test iterator contains no samples.")

    # Compute metrics globally, rather than averaging per-batch metrics
    accuracy = (total_tp + total_tn) / total_samples
    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )
    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    results_df = pd.DataFrame({
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1],
    })

    # Rows represent actual classes; columns represent predicted classes
    confusion_matrix_df = pd.DataFrame(
        [
            [total_tn, total_fp],
            [total_fn, total_tp],
        ],
        index=["Negative", "Positive"],
        columns=["Negative", "Positive"],
    )
    confusion_matrix_df.index.name = "Actual"
    confusion_matrix_df.columns.name = "Predicted"

    return results_df, confusion_matrix_df


def get_best_result(results: dict) -> dict:
    """Return the configuration with the highest final validation accuracy.

    Args:
        results: Dictionary mapping configuration keys to training
            history dictionaries containing valid_acc lists.

    Returns:
        A dictionary containing the best configuration and its results.
    """
    if not results:
        raise ValueError("Results cannot be empty.")

    for key, history in results.items():
        if not history.get("valid_acc"):
            raise ValueError(
                f"Configuration {key!r} has no validation accuracy history."
            )

    # select the configuration with the highest final valid_acc
    best_key = max(
        results,
        key=lambda key: results[key]["valid_acc"][-1],
    )

    return {best_key: results[best_key]}