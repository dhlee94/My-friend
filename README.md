# my-friend

논문 아이디어를 PyTorch / Lightning 코드로 옮기는 LangGraph 기반 페어 프로그래밍 에이전트.

## 로컬 실행 (Mac, MPS — 권장)

Python ≥ 3.11 필요 (시스템 Python 3.9에서는 UI가 동작하지 않음).

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env   # ANTHROPIC_API_KEY 입력
set -a; source .env; set +a
cd ui && chainlit run app.py          # http://localhost:8000
```

## Docker

```bash
cp .env.example .env
docker compose --profile cpu  up --build   # Mac (Apple Silicon): arm64 + CPU torch
docker compose --profile cuda up --build   # Linux + NVIDIA: amd64 + CUDA 13.0 torch
```

컨테이너 안에서는 Metal(MPS)을 쓸 수 없으므로 Mac에서 GPU 가속이 필요하면 로컬 실행을 사용합니다.

## RAG 인덱스 (선택)

```bash
python -m my_friend.rag.pipeline build --input papers/ --out data/index
```

`data/index`가 있으면 Scout/Planner가 로컬 논문·코드를 함께 검색합니다.

## 테스트

```bash
pytest -q
```
