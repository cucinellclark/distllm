"""Distributed inference for generating embeddings without requiring an encoder."""

from __future__ import annotations

import functools
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Optional

from parsl.concurrent import ParslPoolExecutor
from pydantic import Field
from pydantic import field_validator

from distllm.embed import DatasetConfigs
from distllm.embed import EmbedderConfigs
from distllm.embed import PoolerConfigs
from distllm.embed import WriterConfigs
from distllm.parsl import ComputeConfigs
from distllm.utils import BaseConfig


def embedding_worker(  # noqa: PLR0913
    input_path: Path,
    output_dir: Path,
    dataset_kwargs: dict[str, Any],
    pooler_kwargs: dict[str, Any],
    embedder_kwargs: dict[str, Any],
    writer_kwargs: dict[str, Any],
    encoder_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    """Embed a single file and save a numpy array with embeddings without requiring an encoder.

    Parameters
    ----------
    input_path : Path
        The path to the input file to embed.
    output_dir : Path
        The directory to save the embeddings.
    dataset_kwargs : dict[str, Any]
        The dataset configuration.
    pooler_kwargs : dict[str, Any]
        The pooler configuration.
    embedder_kwargs : dict[str, Any]
        The embedder configuration.
    writer_kwargs : dict[str, Any]
        The writer configuration.
    encoder_kwargs : Optional[dict[str, Any]]
        The encoder configuration, optional for encoder-free embedders.
    """
    # Imports are here since this function is called in a parsl process
    from uuid import uuid4

    from distllm.embed import get_dataset
    from distllm.embed import get_embedder
    from distllm.embed import get_pooler
    from distllm.embed import get_writer
    from distllm.timer import Timer

    # Optional encoder import
    encoder = None
    if encoder_kwargs:
        from distllm.embed import get_encoder
        with Timer('loaded-encoder', input_path):
            encoder = get_encoder(encoder_kwargs, register=True)

    # Time the worker function
    timer = Timer('finished-embedding', input_path).start()

    # Initialize the dataset
    dataset = get_dataset(dataset_kwargs)

    # Initialize the pooler
    pooler = get_pooler(pooler_kwargs)

    # Initialize the embedder
    embedder = get_embedder(embedder_kwargs)

    # Initialize the writer
    writer = get_writer(writer_kwargs)

    # Initialize the dataloader - note that for encoder-free embedders,
    # the dataset must be able to work without an encoder
    with Timer('loaded-dataset', input_path):
        dataloader = dataset.get_dataloader(input_path, encoder)

    # Compute the embeddings
    with Timer('computed-embeddings', input_path):
        result = embedder.embed(dataloader, encoder, pooler)

    # Create the output directory for the embedding dataset
    dataset_dir = output_dir / f'{uuid4()}'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Write the result to disk
    with Timer('wrote-embeddings', input_path):
        writer.write(dataset_dir, result)

    # Stop the timer to log the worker time
    timer.stop()


class EncoderFreeConfig(BaseConfig):
    """Configuration for distributed inference without requiring an encoder."""

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
    # Settings for the parsl compute backend.
    compute_config: ComputeConfigs

    @field_validator('input_dir', 'output_dir')
    @classmethod
    def resolve_path(cls, value: Path) -> Path:
        """Resolve the path to an absolute path."""
        return value.resolve()


if __name__ == '__main__':
    # Parse arguments from the command line
    parser = ArgumentParser(description='Embed text without requiring an encoder')
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to the .yaml configuration file',
    )
    args = parser.parse_args()

    # Load the configuration
    config = EncoderFreeConfig.from_yaml(args.config)

    # Create a directory for the embeddings
    embedding_dir = config.output_dir / 'embeddings'

    # Make the output directory
    embedding_dir.mkdir(parents=True, exist_ok=True)

    # Log the configuration
    config.write_yaml(config.output_dir / 'config.yaml')

    # Set the static arguments of the worker function
    worker_fn = functools.partial(
        embedding_worker,
        output_dir=embedding_dir,
        dataset_kwargs=config.dataset_config.model_dump(),
        pooler_kwargs=config.pooler_config.model_dump(),
        embedder_kwargs=config.embedder_config.model_dump(),
        writer_kwargs=config.writer_config.model_dump(),
    )

    # Collect all input files
    input_files = []
    for pattern in config.glob_patterns:
        input_files.extend(list(config.input_dir.glob(pattern)))

    # Log the input files to stdout
    print(f'Found {len(input_files)} input files to embed')

    # Set the parsl compute settings
    parsl_config = config.compute_config.get_config(
        config.output_dir / 'parsl',
    )

    # Distribute the input files across processes
    with ParslPoolExecutor(parsl_config) as pool:
        list(pool.map(worker_fn, input_files)) 