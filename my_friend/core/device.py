"""디바이스 해석: cuda → mps → cpu.

torch는 지연 import — RAG 인덱스 로드/BM25 검색 경로는 torch 없이도 돌아야 한다.
"""
from __future__ import annotations

import os
from typing import Optional


def resolve_device(preferred: Optional[str] = None) -> str:
    """`MY_FRIEND_DEVICE` env > 인자 > 자동 탐지 순서."""
    preferred = os.environ.get("MY_FRIEND_DEVICE", preferred)
    if preferred:
        return preferred

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        # MPS 미지원 op은 CPU로 폴백 (torch import 이전에 세팅돼야 효과가 있으므로 엔트리포인트에서도 설정 권장)
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"
