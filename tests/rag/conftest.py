from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Sequence

import numpy as np
import pytest

from my_friend.rag.analyzer import EnglishAnalyzer

PAPER = r"""# Attention Is All You Need

## Abstract

We propose the Transformer, a model architecture based solely on attention mechanisms.

## 1 Introduction

Recurrent neural networks have been the dominant approach for sequence modeling.

## 3 Model Architecture

### 3.2 Scaled Dot-Product Attention

We call our particular attention scaled dot-product attention. The input consists of queries and keys of dimension $d_k$.

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where the scaling factor $1/\sqrt{d_k}$ counteracts large dot products pushing the softmax into small gradients.

### 3.5 Positional Encoding

We use sine and cosine functions of different frequencies for positional encoding.

```python
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```

## 6 Experiments

Table 2: BLEU scores on WMT 2014.

| Model | EN-DE | EN-FR |
|-------|-------|-------|
| Transformer (base) | 27.3 | 38.1 |
| Transformer (big) | 28.4 | 41.8 |

We train with label smoothing and the RMSNorm variant in an ablation.
"""


@pytest.fixture(scope="session")
def analyzer() -> EnglishAnalyzer:
    return EnglishAnalyzer()


class HashEmbedder:
    """결정적 bag-of-stems 해싱 임베더 (D=256). 모델 다운로드 없이 dense 경로 검증용."""

    name = "test/hash-256"

    def __init__(self, analyzer: EnglishAnalyzer, dim: int = 256) -> None:
        self.analyzer, self.dim = analyzer, dim

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in self.analyzer.analyze(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)

    embed_documents = _embed
    embed_queries = _embed


class FakeLLM:
    def __init__(self, response: Dict[str, Any]) -> None:
        self.response = response
        self.calls: List[str] = []

    def generate_json(self, system: str, user: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(user)
        return self.response


@pytest.fixture
def hash_embedder(analyzer) -> HashEmbedder:
    return HashEmbedder(analyzer)
