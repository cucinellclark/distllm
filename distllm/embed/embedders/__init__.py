"""Module for embedders."""

from __future__ import annotations

from typing import Any
from typing import Union

from distllm.embed.embedders.base import Embedder
from distllm.embed.embedders.base import EmbedderResult
from distllm.embed.embedders.encoder_free_semantic_chunk import EncoderFreeSemanticChunkEmbedder
from distllm.embed.embedders.encoder_free_semantic_chunk import EncoderFreeSemanticChunkEmbedderConfig
from distllm.embed.embedders.full_sequence import FullSequenceEmbedder
from distllm.embed.embedders.full_sequence import FullSequenceEmbedderConfig
from distllm.embed.embedders.semantic_chunk import SemanticChunkEmbedder
from distllm.embed.embedders.semantic_chunk import SemanticChunkEmbedderConfig
from distllm.utils import BaseConfig

EmbedderConfigs = Union[
    FullSequenceEmbedderConfig,
    SemanticChunkEmbedderConfig,
    EncoderFreeSemanticChunkEmbedderConfig,
]

STRATEGIES: dict[str, tuple[type[BaseConfig], type[Embedder]]] = {
    'full_sequence': (FullSequenceEmbedderConfig, FullSequenceEmbedder),
    'semantic_chunk': (SemanticChunkEmbedderConfig, SemanticChunkEmbedder),
    'encoder_free_semantic_chunk': (EncoderFreeSemanticChunkEmbedderConfig, EncoderFreeSemanticChunkEmbedder),
}


def get_embedder(kwargs: dict[str, Any]) -> Embedder:
    """Get the instance based on the kwargs.

    Currently supports the following strategies:
    - full_sequence
    - semantic_chunk
    - encoder_free_semantic_chunk

    Parameters
    ----------
    kwargs : dict[str, Any]
        The configuration. Contains a `name` argument
        to specify the strategy to use.

    Returns
    -------
    Embedder
        The instance.

    Raises
    ------
    ValueError
        If the `name` is unknown.
    """
    name = kwargs.get('name', '')
    strategy = STRATEGIES.get(name)
    if not strategy:
        raise ValueError(
            f'Unknown embedder name: {name}.'
            f' Available: {set(STRATEGIES.keys())}',
        )

    # Get the config and classes
    config_cls, cls = strategy

    return cls(config_cls(**kwargs))
