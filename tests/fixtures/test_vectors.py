#!/usr/bin/env python3
"""
SiLens Test Fixtures - Test Vector Generation
==============================================

Generates reproducible test vectors for verification.

This module provides:
- Reproducible random test vectors with fixed seeds
- Known-answer test (KAT) vectors
- Edge case generators
- Test vectors for specific modules

Usage:
    from tests.fixtures.test_vectors import (
        generate_reproducible_vectors,
        generate_edge_cases
    )
    
    vectors = generate_reproducible_vectors(
        module='attention',
        num_vectors=100,
        seed=42
    )
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import json


@dataclass
class KnownAnswerTest:
    """Known-answer test vector with pre-computed expected output."""
    name: str
    inputs: Dict[str, np.ndarray]
    expected_outputs: Dict[str, np.ndarray]
    description: str = ""
    tolerance: float = 1e-5
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'tolerance': self.tolerance,
            'inputs': {k: v.tolist() for k, v in self.inputs.items()},
            'expected_outputs': {k: v.tolist() for k, v in self.expected_outputs.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'KnownAnswerTest':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            description=data.get('description', ''),
            tolerance=data.get('tolerance', 1e-5),
            inputs={k: np.array(v) for k, v in data['inputs'].items()},
            expected_outputs={k: np.array(v) for k, v in data['expected_outputs'].items()},
        )


def generate_reproducible_vectors(
    module: str,
    num_vectors: int = 100,
    seed: int = 42,
    **kwargs
) -> List[Dict[str, np.ndarray]]:
    """
    Generate reproducible test vectors for a specific module.
    
    Args:
        module: Module name ('popcount', 'attention', 'transformer', etc.)
        num_vectors: Number of vectors to generate
        seed: Random seed for reproducibility
        **kwargs: Module-specific parameters
        
    Returns:
        List of test vectors as dictionaries
    """
    np.random.seed(seed)
    
    if module == 'popcount':
        return _generate_popcount_vectors(num_vectors, **kwargs)
    elif module == 'ternary_mac':
        return _generate_ternary_mac_vectors(num_vectors, **kwargs)
    elif module == 'binary_dot':
        return _generate_binary_dot_vectors(num_vectors, **kwargs)
    elif module == 'attention':
        return generate_attention_vectors(num_vectors, seed, **kwargs)
    elif module == 'transformer':
        return generate_transformer_vectors(num_vectors, seed, **kwargs)
    elif module == 'softmax':
        return _generate_softmax_vectors(num_vectors, **kwargs)
    elif module == 'gelu':
        return _generate_gelu_vectors(num_vectors, **kwargs)
    elif module == 'layer_norm':
        return _generate_layer_norm_vectors(num_vectors, **kwargs)
    else:
        raise ValueError(f"Unknown module: {module}")


def _generate_popcount_vectors(num_vectors: int, width: int = 512) -> List[Dict]:
    """Generate popcount test vectors."""
    vectors = []
    
    # Edge cases
    vectors.append({
        'input': np.zeros(width, dtype=np.uint8),
        'expected': 0,
        'name': 'all_zeros'
    })
    vectors.append({
        'input': np.ones(width, dtype=np.uint8),
        'expected': width,
        'name': 'all_ones'
    })
    
    # Single bit
    for i in range(min(width, 16)):
        inp = np.zeros(width, dtype=np.uint8)
        inp[i] = 1
        vectors.append({
            'input': inp,
            'expected': 1,
            'name': f'single_bit_{i}'
        })
    
    # Random patterns
    for i in range(num_vectors - len(vectors)):
        inp = np.random.randint(0, 2, size=width, dtype=np.uint8)
        vectors.append({
            'input': inp,
            'expected': int(np.sum(inp)),
            'name': f'random_{i}'
        })
    
    return vectors


def _generate_ternary_mac_vectors(
    num_vectors: int,
    num_elements: int = 256,
    act_width: int = 8
) -> List[Dict]:
    """Generate ternary MAC test vectors."""
    vectors = []
    
    # All +1 weights
    act = np.arange(1, num_elements + 1, dtype=np.uint8)
    weights = np.ones(num_elements, dtype=np.int8)
    vectors.append({
        'activations': act,
        'weights': weights,
        'expected': int(np.sum(act)),
        'name': 'all_positive'
    })
    
    # All -1 weights
    vectors.append({
        'activations': act,
        'weights': -np.ones(num_elements, dtype=np.int8),
        'expected': -int(np.sum(act)),
        'name': 'all_negative'
    })
    
    # All zero weights
    vectors.append({
        'activations': act,
        'weights': np.zeros(num_elements, dtype=np.int8),
        'expected': 0,
        'name': 'all_zero'
    })
    
    # Random mixed
    for i in range(num_vectors - 3):
        act = np.random.randint(0, 256, size=num_elements, dtype=np.uint8)
        weights = np.random.choice([-1, 0, 1], size=num_elements).astype(np.int8)
        expected = int(np.sum(act.astype(np.int32) * weights.astype(np.int32)))
        
        vectors.append({
            'activations': act,
            'weights': weights,
            'expected': expected,
            'name': f'random_{i}'
        })
    
    return vectors


def _generate_binary_dot_vectors(
    num_vectors: int,
    width: int = 512
) -> List[Dict]:
    """Generate binary dot product test vectors."""
    vectors = []
    
    # All match (all 1s)
    vectors.append({
        'a': np.ones(width, dtype=np.uint8),
        'b': np.ones(width, dtype=np.uint8),
        'expected': width,  # All +1 * +1 = width
        'name': 'all_ones'
    })
    
    # All opposite
    vectors.append({
        'a': np.ones(width, dtype=np.uint8),
        'b': np.zeros(width, dtype=np.uint8),
        'expected': -width,  # All +1 * -1 = -width
        'name': 'all_opposite'
    })
    
    # Random
    for i in range(num_vectors - 2):
        a = np.random.randint(0, 2, size=width, dtype=np.uint8)
        b = np.random.randint(0, 2, size=width, dtype=np.uint8)
        # Convert {0,1} to {-1,+1}
        a_signed = 2 * a.astype(np.int32) - 1
        b_signed = 2 * b.astype(np.int32) - 1
        expected = int(np.sum(a_signed * b_signed))
        
        vectors.append({
            'a': a,
            'b': b,
            'expected': expected,
            'name': f'random_{i}'
        })
    
    return vectors


def _generate_softmax_vectors(
    num_vectors: int,
    seq_len: int = 8,
    frac_bits: int = 6
) -> List[Dict]:
    """Generate softmax test vectors."""
    vectors = []
    
    # Equal inputs
    vectors.append({
        'input': np.zeros(seq_len, dtype=np.float32),
        'expected': np.ones(seq_len, dtype=np.float32) / seq_len,
        'name': 'equal_inputs'
    })
    
    # One dominant
    inp = np.array([-3.0] * seq_len, dtype=np.float32)
    inp[0] = 3.0
    exp_inp = np.exp(inp - np.max(inp))
    expected = exp_inp / np.sum(exp_inp)
    vectors.append({
        'input': inp,
        'expected': expected,
        'name': 'one_dominant'
    })
    
    # Random
    for i in range(num_vectors - 2):
        inp = np.random.randn(seq_len).astype(np.float32) * 2
        exp_inp = np.exp(inp - np.max(inp))
        expected = exp_inp / np.sum(exp_inp)
        
        vectors.append({
            'input': inp,
            'expected': expected,
            'name': f'random_{i}'
        })
    
    return vectors


def _generate_gelu_vectors(
    num_vectors: int,
    width: int = 16
) -> List[Dict]:
    """Generate GELU activation test vectors."""
    vectors = []
    
    # Key values
    key_vals = [-3, -2, -1, 0, 1, 2, 3]
    for val in key_vals:
        inp = np.full(width, val, dtype=np.float32)
        # Exact GELU
        expected = 0.5 * inp * (1 + np.tanh(np.sqrt(2/np.pi) * (inp + 0.044715 * inp**3)))
        
        vectors.append({
            'input': inp,
            'expected': expected,
            'name': f'key_value_{val}'
        })
    
    # Random
    for i in range(num_vectors - len(key_vals)):
        inp = np.random.uniform(-4, 4, size=width).astype(np.float32)
        expected = 0.5 * inp * (1 + np.tanh(np.sqrt(2/np.pi) * (inp + 0.044715 * inp**3)))
        
        vectors.append({
            'input': inp,
            'expected': expected,
            'name': f'random_{i}'
        })
    
    return vectors


def _generate_layer_norm_vectors(
    num_vectors: int,
    dim: int = 64,
    eps: float = 1e-6
) -> List[Dict]:
    """Generate layer normalization test vectors."""
    vectors = []
    
    # Sequential
    inp = np.arange(dim, dtype=np.float32)
    mean = np.mean(inp)
    var = np.var(inp)
    expected = (inp - mean) / np.sqrt(var + eps)
    
    vectors.append({
        'input': inp,
        'gamma': np.ones(dim, dtype=np.float32),
        'beta': np.zeros(dim, dtype=np.float32),
        'expected': expected,
        'name': 'sequential'
    })
    
    # Constant (zero variance)
    inp = np.full(dim, 5.0, dtype=np.float32)
    expected = np.zeros(dim, dtype=np.float32)  # (x - mean) = 0
    
    vectors.append({
        'input': inp,
        'gamma': np.ones(dim, dtype=np.float32),
        'beta': np.zeros(dim, dtype=np.float32),
        'expected': expected,
        'name': 'constant'
    })
    
    # Random
    for i in range(num_vectors - 2):
        inp = np.random.randn(dim).astype(np.float32)
        gamma = np.random.randn(dim).astype(np.float32) * 0.1 + 1
        beta = np.random.randn(dim).astype(np.float32) * 0.1
        
        mean = np.mean(inp)
        var = np.var(inp)
        normalized = (inp - mean) / np.sqrt(var + eps)
        expected = gamma * normalized + beta
        
        vectors.append({
            'input': inp,
            'gamma': gamma,
            'beta': beta,
            'expected': expected,
            'name': f'random_{i}'
        })
    
    return vectors


def generate_attention_vectors(
    num_vectors: int = 100,
    seed: int = 42,
    embed_dim: int = 64,
    num_heads: int = 4,
    seq_lengths: List[int] = None
) -> List[Dict]:
    """Generate attention test vectors."""
    np.random.seed(seed)
    
    if seq_lengths is None:
        seq_lengths = [1, 4, 8, 16]
    
    vectors = []
    
    # Edge cases
    # All zeros
    vectors.append({
        'input': np.zeros((4, embed_dim), dtype=np.float32),
        'name': 'all_zeros',
        'seq_len': 4
    })
    
    # Identity pattern
    vectors.append({
        'input': np.eye(min(8, embed_dim), embed_dim, dtype=np.float32),
        'name': 'identity',
        'seq_len': min(8, embed_dim)
    })
    
    # Random vectors
    for i in range(num_vectors - 2):
        seq_len = seq_lengths[i % len(seq_lengths)]
        inp = np.random.randn(seq_len, embed_dim).astype(np.float32) * 0.1
        
        vectors.append({
            'input': inp,
            'name': f'random_seq{seq_len}_{i}',
            'seq_len': seq_len
        })
    
    return vectors


def generate_transformer_vectors(
    num_vectors: int = 100,
    seed: int = 42,
    embed_dim: int = 64,
    seq_lengths: List[int] = None
) -> List[Dict]:
    """Generate transformer block test vectors."""
    np.random.seed(seed)
    
    if seq_lengths is None:
        seq_lengths = [4, 8, 16, 32]
    
    vectors = []
    
    for i in range(num_vectors):
        seq_len = seq_lengths[i % len(seq_lengths)]
        inp = np.random.randn(seq_len, embed_dim).astype(np.float32) * 0.1
        
        vectors.append({
            'input': inp,
            'name': f'transformer_seq{seq_len}_{i}',
            'seq_len': seq_len
        })
    
    return vectors


def generate_edge_cases(module: str, **kwargs) -> List[KnownAnswerTest]:
    """
    Generate edge case test vectors for a module.
    
    Args:
        module: Module name
        **kwargs: Module-specific parameters
        
    Returns:
        List of KnownAnswerTest objects
    """
    edge_cases = []
    
    if module == 'popcount':
        width = kwargs.get('width', 512)
        
        edge_cases.append(KnownAnswerTest(
            name='popcount_all_zeros',
            inputs={'data': np.zeros(width, dtype=np.uint8)},
            expected_outputs={'count': np.array([0])},
            description='All zero bits'
        ))
        
        edge_cases.append(KnownAnswerTest(
            name='popcount_all_ones',
            inputs={'data': np.ones(width, dtype=np.uint8)},
            expected_outputs={'count': np.array([width])},
            description='All one bits'
        ))
    
    elif module == 'softmax':
        seq_len = kwargs.get('seq_len', 8)
        
        edge_cases.append(KnownAnswerTest(
            name='softmax_equal',
            inputs={'x': np.zeros(seq_len, dtype=np.float32)},
            expected_outputs={'y': np.ones(seq_len, dtype=np.float32) / seq_len},
            description='Equal inputs produce uniform output'
        ))
        
        # One very large value
        x = np.full(seq_len, -100.0, dtype=np.float32)
        x[0] = 0.0
        edge_cases.append(KnownAnswerTest(
            name='softmax_one_hot',
            inputs={'x': x},
            expected_outputs={'y': np.eye(1, seq_len, dtype=np.float32).flatten()},
            description='One dominant value produces near-one-hot output',
            tolerance=0.01
        ))
    
    elif module == 'gelu':
        width = kwargs.get('width', 16)
        
        edge_cases.append(KnownAnswerTest(
            name='gelu_zero',
            inputs={'x': np.zeros(width, dtype=np.float32)},
            expected_outputs={'y': np.zeros(width, dtype=np.float32)},
            description='GELU(0) = 0'
        ))
        
        # Large positive values - should approach identity
        x = np.full(width, 5.0, dtype=np.float32)
        edge_cases.append(KnownAnswerTest(
            name='gelu_large_positive',
            inputs={'x': x},
            expected_outputs={'y': x},  # GELU(x) ≈ x for large x
            description='Large positive values approach identity',
            tolerance=0.1
        ))
    
    return edge_cases


def save_test_vectors(vectors: List[Dict], output_path: str):
    """Save test vectors to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    serializable = []
    for v in vectors:
        sv = {}
        for k, val in v.items():
            if isinstance(val, np.ndarray):
                sv[k] = val.tolist()
            else:
                sv[k] = val
        serializable.append(sv)
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)


def load_test_vectors(input_path: str) -> List[Dict]:
    """Load test vectors from JSON file."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Convert lists back to numpy arrays
    vectors = []
    for v in data:
        nv = {}
        for k, val in v.items():
            if isinstance(val, list):
                nv[k] = np.array(val)
            else:
                nv[k] = val
        vectors.append(nv)
    
    return vectors


if __name__ == "__main__":
    # Generate sample vectors for each module
    modules = ['popcount', 'ternary_mac', 'binary_dot', 'softmax', 'gelu', 'layer_norm']
    
    for module in modules:
        vectors = generate_reproducible_vectors(module, num_vectors=10)
        print(f"{module}: generated {len(vectors)} vectors")
        
        edge_cases = generate_edge_cases(module)
        print(f"  Edge cases: {len(edge_cases)}")
