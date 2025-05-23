"""Encoder-free dataset that works with pre-computed embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from torch.utils.data import DataLoader

from distllm.embed.datasets.utils import InMemoryDataset
from distllm.embed.datasets.utils import SimpleDataCollator
from distllm.embed.encoders.base import Encoder
from distllm.utils import BaseConfig


class EncoderFreeDatasetConfig(BaseConfig):
    """Configuration for the encoder-free dataset."""

    name: Literal['encoder_free'] = 'encoder_free'  # type: ignore[assignment]

    # The name of the text field in the jsonl file
    text_field: str = Field(
        'text',
        description='The name of the text field in the jsonl file',
    )
    
    # The name of the embeddings field in the jsonl file
    embeddings_field: str = Field(
        'embeddings',
        description='The name of the embeddings field in the jsonl file',
    )
    
    # Number of data workers for batching
    num_data_workers: int = Field(
        4,
        description='Number of data workers for batching',
    )
    
    # Inference batch size
    batch_size: int = Field(
        8,
        description='Inference batch size',
    )
    
    # Whether to pin memory for the dataloader
    pin_memory: bool = Field(
        True,
        description='Whether to pin memory for the dataloader',
    )


class EncoderFreeDataset:
    """Encoder-free dataset that works with pre-computed embeddings."""

    def __init__(self, config: EncoderFreeDatasetConfig) -> None:
        """Initialize the dataset with the configuration."""
        self.config = config

    def get_dataloader(
        self,
        data_file: Path,
        encoder: Optional[Encoder] = None,
    ) -> DataLoader:
        """Instantiate a dataloader for the dataset.

        Parameters
        ----------
        data_file : Path
            The file to read.
        encoder : Optional[Encoder]
            Not used in this dataset.

        Returns
        -------
        DataLoader
            The dataloader instance.
        """
        # Read the jsonl file
        lines = data_file.read_text().strip().split('\n')
        content = [json.loads(line) for line in lines]

        # Extract the text data
        data = [item[self.config.text_field] for item in content]
        
        # Extract metadata including embeddings
        metadata = []
        for item in content:
            meta = {
                'path': str(data_file),
                self.config.embeddings_field: item[self.config.embeddings_field]
            }
            # Add any other fields as metadata
            for key, value in item.items():
                if key not in [self.config.text_field, self.config.embeddings_field]:
                    meta[key] = value
            metadata.append(meta)

        # Instantiate the dataloader with the simple collator
        return DataLoader(
            pin_memory=self.config.pin_memory,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_data_workers,
            dataset=InMemoryDataset(data, metadata),
            collate_fn=SimpleDataCollator(),
        ) 