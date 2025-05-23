"""Simple JSONL chunk dataset that doesn't rely on an encoder.

This implementation extends JsonlChunkDataset but uses a simple collator
instead of an encoder-based one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Literal

from torch.utils.data import DataLoader

from distllm.embed.datasets.jsonl_chunk import JsonlChunkDataset, JsonlChunkDatasetConfig
from distllm.embed.datasets.utils import InMemoryDataset


class SimpleDataCollator:
    """Simple data collator that doesn't use a tokenizer."""

    def __call__(self, batch: list[str]) -> list[str]:
        """Return the batch as is without tokenization."""
        return batch


class SimpleChunkDatasetConfig(JsonlChunkDatasetConfig):
    """Configuration for the SimpleChunkDataset."""

    # Override the name
    name: Literal['simple_chunk'] = 'simple_chunk'  # type: ignore[assignment]


class SimpleChunkDataset(JsonlChunkDataset):
    """Jsonl chunking dataset that doesn't rely on an encoder."""

    def __init__(self, config: SimpleChunkDatasetConfig):
        """Initialize the dataset."""
        super().__init__(config)
        self.config = config

    def get_dataloader(
        self,
        data_file: Path,
    ) -> DataLoader:
        """Instantiate a dataloader for the dataset without needing an encoder.

        Parameters
        ----------
        data_file : Path
            The file to read.

        Returns
        -------
        DataLoader
            The dataloader instance.

        Raises
        ------
        ValueError
            If the metadata is empty.
        """
        # Read the jsonl file
        lines = data_file.read_text().strip().split('\n')
        content: list[dict[str, Any]] = [json.loads(line) for line in lines]

        # Extract the text data
        data = [item.pop(self.config.text_field) for item in content]

        # Extract the metadata
        metadata = content

        # Check if metadata is empty
        if not metadata or any(not item for item in metadata):
            raise ValueError('Metadata is empty. Please check the jsonl file.')

        # Split the data based on the split criteria
        splits = [self.splitter(text) for text in data]

        # Group each text split into windowed buffers
        buffers, metadatas = [], []
        for idx, split in enumerate(splits):
            bufs = self._sentences_to_buffers(split, self.config.buffer_size)
            buffers.extend(bufs)
            for sentence in split:
                mdata = metadata[idx].copy()
                mdata['sentence'] = sentence
                metadatas.append(mdata)

        # Apply a length filter to remove any small buffers
        filter_indices = [
            i
            for i, buf in enumerate(buffers)
            if len(buf) > self.config.min_buffer_length
        ]
        buffers = [buffers[i] for i in filter_indices]
        metadatas = [metadatas[i] for i in filter_indices]

        # Instantiate the dataloader with our simple collator
        return DataLoader(
            pin_memory=self.config.pin_memory,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_data_workers,
            dataset=InMemoryDataset(buffers, metadatas),
            collate_fn=SimpleDataCollator(),
        )
        
    def _sentences_to_buffers(self, split: list[str], buffer_size: int) -> list[str]:
        """Group split into buffers (moved from the main module)."""
        buffers = []
        for i in range(len(split)):
            combined = ''.join(
                split[j]
                for j in range(
                    max(0, i - buffer_size),
                    min(i + 1 + buffer_size, len(split)),
                )
            )
            buffers.append(combined)
        return buffers 