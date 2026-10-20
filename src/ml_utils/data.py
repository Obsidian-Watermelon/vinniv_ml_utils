
import torch, nbformat
from torchtext import data
from pathlib import Path


def handle_cuda_as_device() -> torch.device:
    """Select the compute device and enable TF32 on supported CUDA GPUs.

    TF32 can accelerate certain floating-point matrix operations on
    supported NVIDIA GPUs. It may introduce small numerical differences.

    Returns:
        The CUDA device if available; otherwise, the CPU device.
    """
    if torch.cuda.is_available():
        # TF32 matmul support is available on NVIDIA Ampere GPUs
        # (compute capability 8.0) and newer
        capability = torch.cuda.get_device_capability()

        torch.backends.cuda.matmul.allow_tf32 = capability >= (8, 0)

        return torch.device("cuda")

    return torch.device("cpu")


def get_bucket_iterators(
    train_data,
    valid_data,
    test_data,
    batch_size: int,
    device: torch.device,
):
    """Create BucketIterators for training, validation, and testing.

    Examples within each batch are sorted by text length, which helps
    organize variable-length sequences for text classification.

    Args:
        train_data: Training dataset.
        valid_data: Validation dataset.
        test_data: Test dataset.
        batch_size: Number of examples per batch.
        device: Device on which iterator batches should be placed.

    Returns:
        A tuple containing the training, validation, and test iterators.
    """
    return data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        sort_key=lambda example: len(example.text),
        sort_within_batch=True,
        device=device,
    )
    
def load_notebook(path):
    notebook = nbformat.read(path, as_version=4)
    documents = []

    for i, cell in enumerate(notebook.cells):
        if cell.cell_type not in {"markdown", "code"}:
            continue

        content = cell.source.strip()
        if not content:
            continue

        documents.append({
            "text": content,
            "cell_type": cell.cell_type,
            "cell_index": i,
            "source": str(path),
        })

    return documents