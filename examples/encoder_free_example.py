#!/usr/bin/env python
"""Example script for using the encoder-free semantic chunker.

This example shows how to use the encoder-free semantic chunker with pre-computed embeddings.
"""

import json
import numpy as np
from pathlib import Path

from distllm.embed.datasets import get_dataset
from distllm.embed.embedders import get_embedder
from distllm.embed.poolers import get_pooler


def create_sample_data(output_file: Path, num_samples: int = 10) -> None:
    """Create sample data with text and pre-computed embeddings.
    
    Parameters
    ----------
    output_file : Path
        The path to save the sample data.
    num_samples : int, optional
        The number of samples to generate, by default 10
    """
    # Create sample data with pre-computed embeddings
    data = []
    for i in range(num_samples):
        # Generate a random embedding vector (normally you'd use a real embedding model)
        embedding = np.random.randn(768).tolist()  # Common embedding size
        
        # Create a sample text
        text = f"This is sample text number {i}. It contains some sentences that will be chunked. " \
               f"These sentences will be grouped together based on semantic similarity. " \
               f"Even though we don't have a real encoder, we can still use the semantic chunking " \
               f"algorithm with our pre-computed embeddings. This is useful when embeddings are " \
               f"computed elsewhere or stored separately."
        
        # Create a sample entry
        entry = {
            "text": text,
            "embeddings": embedding,
            "metadata": f"Sample {i}"
        }
        data.append(entry)
    
    # Write the data to a jsonl file
    with open(output_file, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')


def run_encoder_free_embedding(input_file: Path) -> None:
    """Run the encoder-free semantic chunker on the input file.
    
    Parameters
    ----------
    input_file : Path
        The path to the input file with pre-computed embeddings.
    """
    # Configure dataset
    dataset_config = {
        "name": "encoder_free",
        "text_field": "text",
        "embeddings_field": "embeddings",
        "batch_size": 2,
    }
    
    # Configure pooler (required by API but not used)
    pooler_config = {
        "name": "first_token",
    }
    
    # Configure embedder
    embedder_config = {
        "name": "encoder_free_semantic_chunk",
        "breakpoint_percentile_threshold": 90,
        "min_chunk_length": 10,  # Small for demo purposes
        "chunk_batch_size": 2,
        "normalize_embeddings": True,
        "embeddings_key": "embeddings",
    }
    
    # Initialize components
    dataset = get_dataset(dataset_config)
    pooler = get_pooler(pooler_config)
    embedder = get_embedder(embedder_config)
    
    # Load data and get dataloader
    dataloader = dataset.get_dataloader(input_file)
    
    # Run the embedder
    result = embedder.embed(dataloader, None, pooler)
    
    # Print the results
    print(f"Generated {len(result.embeddings)} semantic chunks")
    for i, (text, embedding) in enumerate(zip(result.text, result.embeddings)):
        embedding_norm = np.linalg.norm(embedding)
        print(f"\nChunk {i+1}:")
        print(f"  Text: {text[:100]}...")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Embedding norm: {embedding_norm:.4f}")
        if result.metadata:
            print(f"  Metadata: {result.metadata[i]}")


if __name__ == "__main__":
    # Create a temporary directory for the sample data
    tmp_dir = Path("./tmp")
    tmp_dir.mkdir(exist_ok=True)
    
    # Create the sample file path
    sample_file = tmp_dir / "sample_data.jsonl"
    
    # Create sample data
    create_sample_data(sample_file)
    
    # Run the encoder-free embedding
    run_encoder_free_embedding(sample_file)
    
    print(f"\nSample data file created at: {sample_file}")
    print("You can inspect this file to see the format required for pre-computed embeddings.") 