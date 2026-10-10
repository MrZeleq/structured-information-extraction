"""Relation Extraction module."""

from structured_ie.relations.base import BaseRelationExtractor
from structured_ie.relations.glirel_extractor import GLiRELExtractor

__all__ = ["BaseRelationExtractor", "GLiRELExtractor"]
