"""Adapter for RemoteEmbedding to be used with the Encoder interface."""

from __future__ import annotations

import json
import requests
import torch
from typing import Literal
from transformers import BatchEncoding, PreTrainedTokenizer, AutoTokenizer

from distllm.utils import BaseConfig

# Redefine RemoteEmbeddingConfig and RemoteEmbedding here to avoid circular imports
class RemoteEmbeddingConfig(BaseConfig):
    """Configuration for remote embedding service."""
    
    name: Literal['remote_embedding'] = 'remote_embedding'  # type: ignore[assignment]
    
    server: str
    port: int
    api_key: str
    model: str
    embedding_size: int
    tokenizer_name: str | None = None


class RemoteEmbedding:
    """Client that calls a remote embedding service API."""
    
    def __init__(self, config: RemoteEmbeddingConfig) -> None:
        """Initialize the remote embedding client."""
        self.server = config.server
        self.port = config.port
        self.api_key = config.api_key
        self.model = config.model
        self.embedding_size = config.embedding_size
        
        # Load the tokenizer locally - we only send encoded tokens to the server
        tokenizer_name = config.tokenizer_name or config.model
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, 
            trust_remote_code=True,
        )
        
    def get_embeddings(self, batch_encoding: BatchEncoding) -> torch.Tensor:
        """Get embeddings by sending to remote server."""
        # Convert to dict for JSON serialization
        batch_dict = {
            key: value.tolist() if hasattr(value, 'tolist') else value
            for key, value in batch_encoding.items()
        }
        
        url = f'http://{self.server}:{self.port}/v1/embeddings'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }
        payload = {
            'model': self.model,
            'input': batch_dict,
        }
        
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
        )
        
        if response.status_code == 200:
            result = response.json()
            embeddings = torch.tensor(result['embeddings'])
            return embeddings
        else:
            raise ValueError(f"Error from embedding service: {response.status_code}, {response.text}")


class RemoteEmbeddingAdapterConfig(BaseConfig):
    """Configuration for the RemoteEmbeddingAdapter."""
    
    name: Literal['remote_embedding'] = 'remote_embedding'  # type: ignore[assignment]
    
    server: str
    port: int
    api_key: str
    model: str
    embedding_size: int
    tokenizer_name: str | None = None


class RemoteEmbeddingAdapter:
    """Adapter that allows RemoteEmbedding to be used as an Encoder."""
    
    def __init__(self, config: RemoteEmbeddingAdapterConfig) -> None:
        """Initialize the RemoteEmbeddingAdapter."""
        # Create a RemoteEmbeddingConfig from the adapter config
        remote_config = RemoteEmbeddingConfig(
            name='remote_embedding',
            server=config.server,
            port=config.port,
            api_key=config.api_key,
            model=config.model,
            embedding_size=config.embedding_size,
            tokenizer_name=config.tokenizer_name,
        )
        self._remote_embedding = RemoteEmbedding(remote_config)
        
        # Device and dtype are placeholders since actual computation happens remotely
        self._device = torch.device('cpu')
        self._dtype = torch.float32

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the encoder."""
        return self._dtype

    @property
    def device(self) -> torch.device:
        """Get the device of the encoder."""
        return self._device

    @property
    def embedding_size(self) -> int:
        """Get the embedding size of the encoder."""
        return self._remote_embedding.embedding_size

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer of the encoder."""
        return self._remote_embedding.tokenizer

    def encode(self, batch_encoding: BatchEncoding) -> torch.Tensor:
        """Encode the sequence by sending to remote server.
        
        Parameters
        ----------
        batch_encoding : BatchEncoding
            The batch encoding of the sequence.
            
        Returns
        -------
        torch.Tensor
            The embeddings of the sequence
            (shape: [num_sequences, sequence_length, embedding_size])
        """
        return self._remote_embedding.get_embeddings(batch_encoding) 