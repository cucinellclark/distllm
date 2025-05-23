# RemoteEncoder Embedding

This module allows you to embed documents using a RemoteEncoder without needing distributed computation (no compute_config required).

## Overview

The RemoteEncoder connects to a remote embedding service, sending tokenized content to the server for embedding. This implementation provides a simpler alternative to the distributed embedding system when you:

1. Only need to use a remote embedding service
2. Don't need the complexity of distributed computation
3. Want to avoid setting up a compute_config

## Usage

### 1. Create a configuration file

Create a YAML configuration file like the example below:

```yaml
# An input directory containing the files to embed.
input_dir: /path/to/your/input
# An output directory to save the embeddings.
output_dir: /path/to/your/output
# A set of glob patterns to match the input files.
glob_patterns: ['*.jsonl']

# Settings for reading the input files.
dataset_config:
  name: jsonl_chunk
  buffer_size: 4
  batch_size: 2

# Settings for the encoder - using RemoteEncoder
encoder_config:
  name: remote
  server: your-server-address
  port: 9993
  api_key: your-api-key
  model: Salesforce/SFR-Embedding-Mistral  # The model name running on the remote server
  embedding_size: 4096  # The embedding size of the model
  # tokenizer_name: optional-custom-tokenizer  # Uncomment if using a different tokenizer
    
# Settings for the pooler.
pooler_config:
  name: mean  # You can also use: last_token, cls, etc.

# Settings for the embedder.
embedder_config:
  name: semantic_chunk
  chunk_batch_size: 2
  normalize_embeddings: true

# Settings for the writer.
writer_config:
  name: huggingface
```

### 2. Run the script

```bash
python -m distllm.remote_embedding --config /path/to/your/config.yaml
```

## Key Differences from Distributed Embedding

The main differences between this implementation and the standard distributed embedding are:

1. **No compute_config**: This implementation doesn't require a compute configuration.
2. **Sequential Processing**: Files are processed sequentially rather than in parallel.
3. **Simpler Setup**: Fewer dependencies and configuration options.

## When to Use

Use this implementation when:

- You're only using a RemoteEncoder (server-based embedding)
- You don't need distributed processing
- You want a simpler configuration

Use the standard distributed embedding when:

- You need to process many files in parallel
- You're using a local encoder that requires GPU resources
- You need fine-grained control over compute resources 