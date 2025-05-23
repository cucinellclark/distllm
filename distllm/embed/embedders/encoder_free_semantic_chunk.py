"""Encoder-free Semantic Chunk Embedder.

This module extends the semantic chunking implementation but works without requiring
an encoder by using pre-computed embeddings.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from pydantic import Field
from torch.utils.data import DataLoader

from distllm.embed.datasets.utils import InMemoryDataset
from distllm.embed.embedders.base import EmbedderResult
from distllm.embed.embedders.semantic_chunk import (
    SemanticChunkEmbedder,
    SemanticChunkEmbedderConfig,
    build_chunks,
    calculate_distances_between_buffer,
)
from distllm.embed.encoders.base import Encoder
from distllm.embed.poolers.base import Pooler


class EncoderFreeSemanticChunkEmbedderConfig(SemanticChunkEmbedderConfig):
    """Configuration for the encoder-free semantic chunk embedder."""

    name: Literal["encoder_free_semantic_chunk"] = "encoder_free_semantic_chunk"  # type: ignore[assignment]
    embeddings_key: str = Field(
        "embeddings",
        description="The key in the metadata dictionary that contains the pre-computed embeddings.",
    )


def compute_encoder_free_semantic_chunks(
    dataloader: DataLoader,
    breakpoint_percentile_threshold: int,
    min_chunk_length: int,
    embeddings_key: str,
) -> InMemoryDataset:
    """Compute semantic chunked embeddings without using an encoder.

    Parameters
    ----------
    dataloader : DataLoader
        The dataloader to use for batching the data.
    breakpoint_percentile_threshold : int
        The percentile of cosine dissimilarity that must be exceeded
        between a group of sentences and the next to form a chunk.
    min_chunk_length : int
        The minimum length of a chunk (number of characters) to
        filter out any small chunks.
    embeddings_key : str
        The key in the metadata dictionary that contains the pre-computed embeddings.

    Returns
    -------
    InMemoryDataset
        The dataset with the semantically-chunked text and metadata.

    Raises
    ------
    ValueError
        If the dataloader dataset does not have metadata.
    ValueError
        If the metadata does not have a path.
    ValueError
        If the metadata does not have embeddings.
    """
    if dataloader.dataset.metadata is None:
        raise ValueError("Metadata is required for semantic chunking.")

    if dataloader.dataset.metadata[0].get("path") is None:
        raise ValueError("Metadata path is required for semantic chunking.")

    # Check if pre-computed embeddings exist
    if dataloader.dataset.metadata[0].get(embeddings_key) is None:
        raise ValueError(
            f"Pre-computed embeddings with key '{embeddings_key}' are required for encoder-free semantic chunking."
        )

    # Group the data such that we only compute distances between
    # buffers within the same document.
    document_indices = []
    current_idx = 0
    current_doc = dataloader.dataset.metadata[0]["path"]
    for i, metadata in enumerate(dataloader.dataset.metadata):
        if metadata["path"] != current_doc:
            document_indices.append((current_idx, i))
            current_idx = i
            current_doc = metadata["path"]
    document_indices.append((current_idx, len(dataloader.dataset)))

    # Get pre-computed embeddings from metadata
    buffer_embeds = np.array([
        metadata[embeddings_key] for metadata in dataloader.dataset.metadata
    ])

    dataset_indices = []
    for doc_start, doc_end in document_indices:
        # Calculate distances between sentence groups
        distances = calculate_distances_between_buffer(
            buffer_embeds[doc_start:doc_end],
        )

        # Chunk the sentences into semantic groups
        index_groups = build_chunks(distances, breakpoint_percentile_threshold)
        
        dataset_indices.extend(
            [
                (doc_start + start_idx, doc_start + end_idx)
                for start_idx, end_idx in index_groups
            ],
        )

    # Group the data by the index groups
    data = []
    for start, end in dataset_indices:
        group = dataloader.dataset.metadata[start:end]
        chunk = "".join(g["sentence"] for g in group)
        data.append(chunk)

    # Get the metadata for the chunks
    metadata = [
        dataloader.dataset.metadata[start] for start, _ in dataset_indices
    ]

    # Apply a length filter to remove small chunks
    filter_indices = [
        i for i, x in enumerate(data) if len(x) > min_chunk_length
    ]
    data = [data[i] for i in filter_indices]
    metadata = [metadata[i] for i in filter_indices]

    # Drop the splits from the metadata
    for meta in metadata:
        meta.pop("sentence")

    return InMemoryDataset(data, metadata)


class EncoderFreeSemanticChunkEmbedder(SemanticChunkEmbedder):
    """Embedder for semantic chunking without requiring an encoder."""

    def __init__(self, config: EncoderFreeSemanticChunkEmbedderConfig) -> None:
        """Initialize the embedder with the configuration."""
        super().__init__(config)
        self.config = config

    def embed(
        self,
        dataloader: DataLoader,
        encoder: Optional[Encoder] = None,
        pooler: Optional[Pooler] = None,
    ) -> EmbedderResult:
        """Embed the sequences without using an encoder.

        Parameters
        ----------
        dataloader : DataLoader
            The dataloader to use for batching the data.
        encoder : Optional[Encoder]
            Not used in this embedder.
        pooler : Optional[Pooler]
            Not used in this embedder.

        Returns
        -------
        EmbedderResult
            Dataclass with the embeddings, text, and optional metadata.
        """
        dataset = compute_encoder_free_semantic_chunks(
            dataloader=dataloader,
            breakpoint_percentile_threshold=self.config.breakpoint_percentile_threshold,
            min_chunk_length=self.config.min_chunk_length,
            embeddings_key=self.config.embeddings_key,
        )

        # At this point, we assume each document in metadata already has pre-computed embeddings
        # We use these embeddings directly for the chunked data
        chunked_embeds = np.array([
            metadata[self.config.embeddings_key] for metadata in dataset.metadata
        ])

        # Apply normalization if configured
        if self.config.normalize_embeddings:
            norms = np.linalg.norm(chunked_embeds, axis=1, keepdims=True)
            chunked_embeds = chunked_embeds / norms

        # Return the result
        return EmbedderResult(
            embeddings=chunked_embeds,
            text=dataset.data,
            metadata=dataset.metadata,
        ) 