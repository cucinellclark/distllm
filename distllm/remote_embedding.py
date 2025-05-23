"""Remote embedding implementation for distributed computation."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import requests
import torch
from pydantic import Field
from pydantic import field_validator
from transformers import BatchEncoding, AutoTokenizer, PreTrainedTokenizer

# Remove circular imports - these will be imported inside the functions that need them
from distllm.utils import BaseConfig
from distllm.timer import Timer

from distllm.embed.datasets import DatasetConfigs
from distllm.embed.embedders import EmbedderConfigs
from distllm.embed.poolers import PoolerConfigs
from distllm.embed.writers import WriterConfigs


class RemoteEmbeddingConfig(BaseConfig):
    """Configuration for remote embedding service."""
    
    name: Literal['remote_embedding'] = 'remote_embedding'  # type: ignore[assignment]
    
    server: str = Field(
        ...,
        description='Server name or IP you are connecting to',
    )
    port: int = Field(
        ...,
        description='The port the embedding service is listening on',
    )
    api_key: str = Field(
        ...,
        description='The API key for the embedding server',
    )
    model: str = Field(
        ...,
        description='The model ID that the embedding server is running',
    )
    embedding_size: int = Field(
        ...,
        description='The embedding size of the embedding model',
    )


class RemoteEmbedding:
    """Client that calls a remote embedding service API."""
    
    def __init__(self, config: RemoteEmbeddingConfig) -> None:
        """Initialize the remote embedding client."""
        self.server = config['server']
        self.port = config['port']
        self.api_key = config['api_key']
        self.model = config['model']
        self.embedding_size = config['embedding_size']
        
    def get_embeddings(self, query: str | list[str]) -> torch.Tensor:
        """Get embeddings by sending to remote server.
        
        Parameters
        ----------
        query : str | list[str]
            A single query string or list of query strings to embed
            
        Returns
        -------
        torch.Tensor
            The embeddings of the sequence
            (shape: [num_sequences, sequence_length, embedding_size])
        """
        # Convert single string to list
        if isinstance(query, str):
            query = [query]
            
        url = f'http://{self.server}:{self.port}/v1/embeddings'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }
        payload = {
            'model': self.model,
            'input': query,
        }
        
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
        )
        
        if response.status_code == 200:
            result = response.json()
            embeddings = torch.tensor(result['data'][0]['embedding'])
            return embeddings
        else:
            raise ValueError(f"Error from embedding service: {response.status_code}, {response.text}")


def process_embeddings(
    input_path: Path,
    output_dir: Path,
    dataset_kwargs: dict[str, Any],
    pooler_kwargs: dict[str, Any],
    embedder_kwargs: dict[str, Any],
    writer_kwargs: dict[str, Any],
) -> None:
    """Embed a single file and save a numpy array with embeddings."""
    from distllm.embed.datasets import get_dataset
    from distllm.embed.embedders import get_embedder
    from distllm.embed.poolers import get_pooler
    from distllm.embed.writers import get_writer

    # Time the embedding process
    timer = Timer('finished-embedding', input_path).start()

    # Initialize the dataset
    dataset = get_dataset(dataset_kwargs)

    # Initialize the pooler
    pooler = get_pooler(pooler_kwargs)

    # Initialize the embedder
    embedder = get_embedder(embedder_kwargs)

    # Initialize the writer
    writer = get_writer(writer_kwargs)

    # Initialize the dataloader
    with Timer('loaded-dataset', input_path):
        dataloader = dataset.get_dataloader(input_path)

    # Compute the embeddings
    with Timer('computed-embeddings', input_path):
        result = embedder.embed(dataloader, pooler)

    # Create the output directory for the embedding dataset
    dataset_dir = output_dir / f'{uuid4()}'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Write the result to disk
    with Timer('wrote-embeddings', input_path):
        writer.write(dataset_dir, result)

    # Stop the timer to log the process time
    timer.stop()


class Config(BaseConfig):
    """Configuration for remote embedding."""

    # An input directory containing the files to embed.
    input_dir: Path
    # An output directory to save the embeddings.
    output_dir: Path
    # A set of glob patterns to match the input files.
    glob_patterns: list[str] = Field(default=['*'])
    # Settings for reading the input files.
    dataset_config: DatasetConfigs
    # Settings for the pooler.
    pooler_config: PoolerConfigs
    # Settings for the embedder.
    embedder_config: EmbedderConfigs
    # Settings for the writer.
    writer_config: WriterConfigs

    @field_validator('input_dir', 'output_dir')
    @classmethod
    def resolve_path(cls, value: Path) -> Path:
        """Resolve the path to an absolute path."""
        return value.resolve()


if __name__ == '__main__':
    # Parse arguments from the command line
    parser = ArgumentParser(description='Embed text using remote embedding')
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to the .yaml configuration file',
    )
    args = parser.parse_args()

    # Load the configuration
    config = Config.from_yaml(args.config)

    # Create a directory for the embeddings
    embedding_dir = config.output_dir / 'embeddings'

    # Make the output directory
    embedding_dir.mkdir(parents=True, exist_ok=True)

    # Log the configuration
    config.write_yaml(config.output_dir / 'config.yaml')
    
    # Collect all input files
    input_files = []
    for pattern in config.glob_patterns:
        input_files.extend(list(config.input_dir.glob(pattern)))

    # Log the input files to stdout
    print(f'Found {len(input_files)} input files to embed')
    
    # Process each file
    for input_file in input_files:
        print(f'Processing {input_file}')
        process_embeddings(
            input_path=input_file,
            output_dir=embedding_dir,
            dataset_kwargs=config.dataset_config.model_dump(),
            pooler_kwargs=config.pooler_config.model_dump(),
            embedder_kwargs=config.embedder_config.model_dump(),
            writer_kwargs=config.writer_config.model_dump(),
        ) 
