
import torch, nbformat, re
from torchtext import data
from pathlib import Path
from urllib.parse import urljoin


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
    """
    Load a Jupyter notebook and extract its Markdown and code cells.

    Reads a `.ipynb` file and converts each non-empty Markdown or code
    cell into a dictionary containing the cell's text, type, index, and
    source file path. Empty cells and other cell types are skipped.

    Args:
        path (str | pathlib.Path): Path to the Jupyter notebook file.

    Returns:
        list[dict]: A list of dictionaries, one per extracted cell.
            Each dictionary contains:
                - "text" (str): The cell's source content.
                - "cell_type" (str): Either "markdown" or "code".
                - "cell_index" (int): The cell's index in the notebook.
                - "source" (str): Path to the notebook file.

    Raises:
        FileNotFoundError: If the specified notebook file does not exist.
        OSError: If the file cannot be read.
        nbformat.reader.NotJSONError: If the file is not valid notebook JSON.

    Example:
        >>> from pathlib import Path
        >>> documents = load_notebook(Path("tutorial.ipynb"))
        >>> print(documents[0]["text"])
    """
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

def notebook_to_documents(cells):
    """
    Convert extracted notebook cells into section-based documents for RAG.

    Groups notebook content under Markdown headings and creates one document
    for each section. Content appearing before the first recognized heading
    is ignored. Heading markup is excluded from the document text, while
    the heading text is stored as the document title.

    Args:
        cells (Iterable[dict]): Notebook cells produced by `load_notebook()`.
            Each cell is expected to contain:
                - text (str): The cell's source text.
                - cell_type (str): The cell type, such as "markdown" or "code".
                - cell_index (int): The cell's index in the original notebook.
                - source (str): The path or identifier of the source notebook.

    Returns:
        list[dict]: A list of documents. Each document contains:
            - text (str): The section's content, with collected lines joined
              by blank lines.
            - title (str): The Markdown heading identifying the section.
            - source (str): The source notebook path or identifier.
            - cell_indices (list[int]): Indices of notebook cells associated
              with the section.

    Notes:
        - Recognizes ATX headings (# Heading) and Setext headings
          (underlined with equals signs or hyphens).
        - Setext underline lines are excluded from the document content.
        - Only headings in Markdown cells start new sections.
        - The function does not split sections into size-limited chunks.
        - The returned list includes a final document even if its content
          or title is empty.

    Example:
        cells = load_notebook("tutorial.ipynb")
        documents = notebook_to_documents(cells)
    """
    ATX = re.compile(r"^ {0,3}#{1,6}(?:[ \t]+|$)")
    SETEXT = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")
    LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    def is_heading(line1: str, line2: str) -> bool:
        return ATX.match(line1) or SETEXT.match(line2)
    
    def extract_links(text: str):
        base_url = "https://docs.pytorch.org/tutorials/beginner/basics/"
        matches = LINK.findall(text)
        texts_and_urls = [{"title": link_text, "url": urljoin(base_url, url)} for link_text, url in matches]
        text_without_markdown = LINK.sub(r"\1", text)

        return texts_and_urls, text_without_markdown
    
    documents = []
    texts = []
    text = ""
    title = ""
    source = ""
    indices = []
    links = []
    
    for cell in cells:
        indices.append(cell['cell_index'])
        lines = cell['text'].splitlines()
        
        for i, line in enumerate(lines):
            if len(indices) < 1:
                indices.append(cell['cell_index'])
                
            if SETEXT.match(line):
                # Skip header markup lines
                continue
            
            if i+1 == len(lines) and len(title) > 0:
                # Last line so append line
                texts.append(line)
                
                # If markdown strip text and append links
                if is_markdown:
                    # Replace ticks and stars while stripping hyphens and whitespaces
                    joined = ' '.join(texts).replace("`", "").replace("*", "").strip("-").strip()
                    only_links = re.fullmatch(rf"(?:\s*{LINK}\s*)+", joined) is not None
                    
                    # Add "Relevant Links" to text if only hyperlinks
                    if only_links:
                        new_links, _ = extract_links(joined)
                        new_text = " Relevant Links"
                    else:
                        new_links, new_text = extract_links()
                        
                    links += new_links
                    text += new_text
                        
                else:
                    text += ' '.join(texts) 
                    
                continue
            
            is_markdown = cell['cell_type'] == 'markdown'
            line_is_heading = is_markdown and is_heading(line, lines[i+1])
            
            if len(title) > 0 and line_is_heading:
                # New header so create new document
                documents.append({
                    'text': text,
                    'title': title,
                    'source': source,
                    'cell_indices': indices,
                    'links': links
                })
                texts = []
                title = ""
                source = ""
                indices = []
            
            if line_is_heading:
                # Add the header and skip to new line
                title = ATX.sub("", line)
                source = cell['source']
                continue
            
            if len(title) < 1:
                # Don't add text without title
                continue

            texts.append(line)
            
    documents.append({
        'text': text,
        'title': title,
        'source': source,
        'cell_indices': indices,
        'links': links
    })
    
    # Return documents only with text
    return [document for document in documents if document.get('text') != ""]