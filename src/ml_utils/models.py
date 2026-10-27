
import torch
from torch import nn
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from torchinfo import summary


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments."""
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    """Embedding-based text classifier with masked mean pooling."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        padding_idx: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [seq_len, batch_size] from the legacy TorchText iterator
        x = x.transpose(0, 1)  # [batch_size, seq_len]

        padding_mask = x != self.embedding.padding_idx
        embeddings = self.embedding(x)

        # Exclude padding tokens from the mean
        mask = padding_mask.unsqueeze(-1).to(embeddings.dtype)
        pooled = (embeddings * mask).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp_min(1)

        return self.classifier(pooled)


class Transformer(nn.Module):
    """Transformer encoder classifier with positional embeddings."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        padding_idx: int,
        dropout_rate: float = 0.0,
        max_length: int = 5000,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.positional_embedding = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embedding_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [seq_len, batch_size] from the legacy TorchText iterator
        x = x.transpose(0, 1)  # [batch_size, seq_len]

        padding_mask = x == self.embedding.padding_idx
        seq_len = x.shape[1]

        if seq_len > self.positional_embedding.num_embeddings:
            raise ValueError(
                "Input sequence exceeds the maximum positional "
                f"embedding length ({self.positional_embedding.num_embeddings})."
            )

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        x = self.embedding(x) + self.positional_embedding(positions)

        # Prevent attention to padding tokens
        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask,
        )

        # Pool only non-padding positions
        mask = (~padding_mask).unsqueeze(-1).to(x.dtype)
        pooled = (x * mask).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp_min(1)

        return self.classifier(pooled)


class RAG:
    """
    Retrieve relevant documents using cosine similarity between embeddings.

    Document embeddings are normalized when created, allowing cosine
    similarity to be calculated efficiently using a dot product.
    """

    def __init__(
        self,
        documents: list[dict],
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        """
        Initialize the retriever and precompute document embeddings.

        Args:
            documents: Documents containing at least a non-empty 'text' field.
            embedding_model_name: Hugging Face or Sentence Transformers
                model identifier used to encode documents and queries.

        Raises:
            ValueError: If documents are empty or a document has no text.
            TypeError: If a document is not a dictionary or its text is not
                a string.
            KeyError: If a document is missing the 'text' field.
        """
        if not documents:
            raise ValueError("At least one document is required.")

        for i, document in enumerate(documents):
            if not isinstance(document, dict):
                raise TypeError(f"Document {i} must be a dictionary.")

            if "text" not in document:
                raise KeyError(f"Document {i} is missing the 'text' field.")

            if not isinstance(document["text"], str):
                raise TypeError(f"Document {i} text must be a string.")

            if not document["text"].strip():
                raise ValueError(f"Document {i} has empty text.")

        self.documents = documents
        self.embedding_model = SentenceTransformer(embedding_model_name)

        document_texts = [document["text"] for document in documents]

        self.document_embeddings = self.embedding_model.encode(
            document_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        threshold: float = 0.4,
    ) -> list[dict]:
        """
        Retrieve the highest-scoring documents that meet a similarity threshold.

        Args:
            query: The question or text used to find relevant documents.
            top_k: Maximum number of documents to consider, ranked by score.
            threshold: Minimum cosine similarity score for a document to
                be returned.

        Returns:
            Copies of the matching documents, ordered by descending
            similarity score. Each returned document includes a 'score'
            field. Returns an empty list if no documents meet the threshold.

        Raises:
            ValueError: If the query is empty, top_k is less than 1,
                or threshold is outside the range [-1, 1].
            TypeError: If query is not a string.
        """
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between -1.0 and 1.0.")

        query_embedding = self.embedding_model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        # Normalized embeddings make the dot product equivalent to
        # cosine similarity.
        scores = self.document_embeddings @ query_embedding

        k = min(top_k, len(self.documents))
        top_scores, top_indices = torch.topk(scores, k=k)

        results = []

        for score, index in zip(top_scores.tolist(), top_indices.tolist()):
            if score >= threshold:
                results.append({
                    **self.documents[index],
                    "score": score,
                })

        return results


class LLM:
    """
    Generate answers from retrieved evidence using a causal language model.
    """

    def __init__(
        self,
        device: torch.device | str,
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """
        Load the tokenizer and language model onto the requested device.

        Args:
            device: Device used for inference, such as 'cuda', 'mps', or 'cpu'.
            model_name: Hugging Face model identifier.
        """
        self.device = torch.device(device)
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Decoder-only models may not define a separate padding token.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device)
        self.model.eval()

    def generate_answer(
        self,
        query: str,
        retrieved_documents: list[dict],
        prompt: str,
        max_new_tokens: int = 300,
        enable_thinking: bool = False,
    ) -> str:
        """
        Generate an answer grounded in retrieved documents.

        Args:
            query: The question to answer.
            retrieved_documents: Documents returned by the RAG retriever.
                Each document should contain 'source' and 'text' fields.
            prompt: System instructions defining the model's behavior.
            max_new_tokens: Maximum number of new tokens to generate.
            enable_thinking: Whether to enable Qwen3's thinking mode
                through its chat template.

        Returns:
            The generated answer as a string. If no documents are retrieved,
            the model receives an explicit message indicating that no evidence
            is available.

        Raises:
            ValueError: If the query or prompt is empty, or max_new_tokens
                is less than 1.
            TypeError: If query or prompt is not a string.
        """
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string.")

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1.")

        if retrieved_documents:
            context_parts = []

            for i, document in enumerate(retrieved_documents):
                if "text" not in document:
                    raise KeyError(
                        f"Retrieved document {i} is missing the 'text' field."
                    )

                source = document.get("source", "Unknown source")
                text = document["text"]

                context_parts.append(
                    f"Source: {source}\n{text}"
                )

            context = "\n\n".join(context_parts)
        else:
            context = "No evidence was retrieved."

        messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": (
                    f"Evidence:\n{context}\n\n"
                    f"Question: {query}"
                ),
            },
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only newly generated tokens, excluding the prompt.
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0, input_length:]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return answer.strip()


def show_summary(
    model: nn.Module,
    input_size: tuple[int, ...],
):
    """Display model architecture and parameter counts."""
    return summary(
        model=model,
        input_size=input_size,
        dtypes=[torch.long],
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "trainable",
        ],
        col_width=20,
        row_settings=["var_names"],
    )