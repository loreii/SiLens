"""
SiLens Test Fixtures
====================

Shared test data, utilities, and fixtures for the test suite.

Modules:
- test_vectors: Reproducible test vector generation
- test_images: Sample test images

Usage:
    from tests.fixtures import generate_test_vectors, create_test_image
"""

from .test_vectors import (
    generate_reproducible_vectors,
    generate_edge_cases,
    generate_attention_vectors,
    generate_transformer_vectors,
    KnownAnswerTest,
)
