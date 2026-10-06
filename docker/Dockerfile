# syntax=docker/dockerfile:1.7
#
# 단일 Dockerfile, 두 타깃:
#   Mac (Apple Silicon) : linux/arm64 + CPU torch   — 컨테이너에서는 Metal(MPS) 접근 불가
#   Linux + NVIDIA GPU  : linux/amd64 + CUDA torch  — CUDA 런타임은 torch 휠(nvidia-* 패키지)에 포함,
#                                                   호스트엔 드라이버 + nvidia-container-toolkit만 필요
# 같은 python:3.11-slim 베이스를 쓰고 torch 휠 인덱스만 바꾼다 → nvidia/cuda 베이스(+deadsnakes) 대비
# 이미지 구성이 단순하고 두 타깃의 파이썬 환경이 동일하다.
#
#   docker compose --profile cpu  up --build     # Mac
#   docker compose --profile cuda up --build     # Linux/CUDA

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.14.1
ARG MERMAID_VERSION=12.1.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    HF_HOME=/cache/huggingface \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

RUN apt-get update \
 && apt-get install -y --no-install-recommends tini \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) torch 먼저 (가장 큰 레이어, 의존성 변경과 분리해 캐시 유지)
RUN pip install --index-url ${TORCH_INDEX_URL} "torch==${TORCH_VERSION}"

# 2) 나머지 의존성 — torch는 이미 만족되므로 재설치되지 않는다
COPY requirements.txt .
RUN pip install -r requirements.txt \
 && python -c "import torch; print('torch', torch.__version__, 'cuda build:', torch.version.cuda)"

# 3) 오프라인 실행 대비 자산 선적재
#    - tiktoken BPE (RAG 분석기/청커가 첫 실행 시 다운로드하는 파일)
#    - mermaid.min.js (Mermaid.jsx가 /public/vendor를 CDN보다 먼저 시도)
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
 && mkdir -p /app/ui/public/vendor \
 && python -c "import urllib.request as u; u.urlretrieve('https://cdn.jsdelivr.net/npm/mermaid@${MERMAID_VERSION}/dist/mermaid.min.js', '/app/ui/public/vendor/mermaid.min.js')"

COPY my_friend ./my_friend
COPY ui ./ui

# 생성 코드를 실행하는 프로세스이므로 root로 돌리지 않는다
RUN useradd --create-home --uid 1000 app \
 && mkdir -p /cache/huggingface /app/data /app/workspace \
 && chown -R app:app /app /cache /opt/tiktoken
USER app

WORKDIR /app/ui
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4)" || exit 1

# tini: PID 1에서 시그널 전달 + 샌드박스 subprocess 좀비 회수
ENTRYPOINT ["/usr/bin/tini", "--"]
# 워커 1개 고정: SQLite 체크포인트(/app/data)는 단일 writer, 세션별 실행 lock도 프로세스 로컬
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000", "--headless"]
