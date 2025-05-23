"""Encoder-free Semantic Chunk Embedder.

This module extends the semantic chunking implementation but works without requiring
an encoder by using pre-computed embeddings.
"""

from __future__ import annotations

from typing import Literal, Optional
import json
import requests
import torch
import numpy as np
from pydantic import Field
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    # VLLM parameters
    server: str = Field(
        ...,
        description="Server name or IP for the vLLM embedding service",
    )
    port: int = Field(
        ...,
        description="The port the vLLM embedding service is listening on",
    )
    api_key: str = Field(
        ...,
        description="The API key for the vLLM embedding server",
    )
    model: str = Field(
        ...,
        description="The model ID that the vLLM embedding server is running",
    )


def compute_embeddings(
    dataloader: DataLoader,
    server: str,
    port: int,
    api_key: str,
    model: str,
) -> np.ndarray:
    """Compute embeddings using a vLLM API endpoint.

    Parameters
    ----------
    dataloader : DataLoader
        The dataloader to use for batching the data.
    server : str
        Server name or IP for the vLLM embedding service.
    port : int
        The port the vLLM embedding service is listening on.
    api_key : str
        The API key for the vLLM embedding server.
    model : str
        The model ID that the vLLM embedding server is running.

    Returns
    -------
    np.ndarray
        The computed embeddings for each sentence in the dataloader.
    """
    # Determine the total number of sentences
    total_sentences = len(dataloader.dataset)
    
    # Initialize an array for the embeddings
    # We'll determine the embedding size from the first API call
    first_batch = next(iter(dataloader))
    
    # Get the texts from the batch
    if isinstance(first_batch, dict):
        texts = first_batch['text']
    else:
        texts = [dataloader.dataset.data[i] for i in range(len(first_batch))]
    
    # Make the API call to get embeddings
    url = f'http://{server}:{port}/v1/embeddings'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }
    payload = {
        'model': model,
        'input': texts,
    }
    
    response = requests.post(
        url,
        headers=headers,
        data=json.dumps(payload),
    )
    
    if response.status_code != 200:
        raise ValueError(f"Error from embedding service: {response.status_code}, {response.text}")
    
    result = response.json()
    first_embeddings = np.array(result['data'][0]['embedding'])
    embedding_size = first_embeddings.shape[0]
    
    # Initialize the full embeddings array
    all_embeddings = np.zeros((total_sentences, embedding_size), dtype=np.float32)
    
    # Reset the dataloader iterator
    dataloader_iter = iter(dataloader)
    
    # Process all batches, starting with the first one we already fetched
    start_idx = 0
    batch_count = 0
    
    for batch in tqdm(dataloader, desc="Computing embeddings"):
        # Get the texts from the batch
        if isinstance(batch, dict):
            texts = batch['text']
        else:
            texts = [dataloader.dataset.data[i] for i in range(len(batch))]
        
        # Make the API call to get embeddings
        payload = {
            'model': model,
            'input': texts,
        }
        
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
        )
        
        if response.status_code != 200:
            raise ValueError(f"Error from embedding service: {response.status_code}, {response.text}")
        
        result = response.json()
        
        # Extract embeddings from the result
        batch_size = len(texts)
        for i in range(batch_size):
            all_embeddings[start_idx + i] = np.array(result['data'][i]['embedding'])
        
        start_idx += batch_size
        batch_count += 1
    
    return all_embeddings


def compute_encoder_free_semantic_chunks(
    dataloader: DataLoader,
    breakpoint_percentile_threshold: int,
    min_chunk_length: int,
    embeddings_key: str,
    server: str = None,
    port: int = None,
    api_key: str = None,
    model: str = None,
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
    server : str, optional
        Server name or IP for the vLLM embedding service.
    port : int, optional
        The port the vLLM embedding service is listening on.
    api_key : str, optional
        The API key for the vLLM embedding server.
    model : str, optional
        The model ID that the vLLM embedding server is running.

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
    """
    if dataloader.dataset.metadata is None:
        raise ValueError("Metadata is required for semantic chunking.")

    if dataloader.dataset.metadata[0].get("path") is None:
        raise ValueError("Metadata path is required for semantic chunking.")

    # Check if we need to compute embeddings with vLLM or use pre-computed ones
    if server and port and api_key and model:
        # Compute embeddings using vLLM API
        buffer_embeds = compute_embeddings(
            dataloader=dataloader,
            server=server,
            port=port,
            api_key=api_key,
            model=model,
        )
    else:
        # Check if pre-computed embeddings exist
        if dataloader.dataset.metadata[0].get(embeddings_key) is None:
            raise ValueError(
                f"Pre-computed embeddings with key '{embeddings_key}' are required when not providing vLLM parameters."
            )
        
        # Get pre-computed embeddings from metadata
        buffer_embeds = np.array([
            metadata[embeddings_key] for metadata in dataloader.dataset.metadata
        ])

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
            server=self.config.server,
            port=self.config.port,
            api_key=self.config.api_key,
            model=self.config.model,
        )

        # At this point, we either have pre-computed embeddings in metadata or need to compute them
        if hasattr(self.config, 'server') and self.config.server:
            # Re-create a dataloader for the chunked dataset to compute embeddings
            chunk_dataloader = DataLoader(
                dataset,
                batch_size=self.config.chunk_batch_size,
                shuffle=False,
            )
            # Compute embeddings for the chunked data
            chunked_embeds = compute_embeddings(
                dataloader=chunk_dataloader,
                server=self.config.server,
                port=self.config.port,
                api_key=self.config.api_key,
                model=self.config.model,
            )
        else:
            # Use pre-computed embeddings from metadata
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