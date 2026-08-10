FROM python:3.12-slim

# Determinism-friendly defaults. The agent may override thread counts, but the
# hash seed is fixed so repeated runs of the same script behave the same way.
ENV PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=8 \
    HF_HOME=/app/.hf \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/0.9.7/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# CUDA torch, NOT the CPU wheel. This task has a GPU, and the CPU wheel index --
# copied from the CPU-only tasks in this repo -- silently produces a build that
# cannot see it. `torch.cuda.is_available()` returns False, everything falls back
# to CPU, and the reference recipe (2 epochs over 38,420 pairs) takes 6.7 hours
# against a 4-hour timeout. It fails as a timeout with no error message, an hour
# after the interesting part of the run. The default index is the CUDA build.
RUN uv pip install --system torch==2.6.0 \
    && uv pip install --system \
        transformers==4.49.0 \
        safetensors==0.4.5 \
        huggingface_hub==0.28.1 \
        scikit-learn==1.5.2 \
        pandas==2.2.3 \
        numpy==1.26.4 \
        scipy==1.14.1

WORKDIR /app

# Bake the base model so the agent starts from a pinned checkpoint rather than
# whatever the Hub serves at run time. Files are listed explicitly rather than by
# glob so that no unnecessary pickle is pulled in.
ARG BASE_MODEL=distilroberta-base
ARG BASE_REVISION=fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${BASE_MODEL}', revision='${BASE_REVISION}', \
                  local_dir='/app/base_model', \
                  allow_patterns=['config.json','pytorch_model.bin','model.safetensors', \
                                  'vocab.json','merges.txt','tokenizer.json', \
                                  'tokenizer_config.json','special_tokens_map.json'])" \
    && rm -rf /app/.hf

# Training pairs only. The held-out prompts are never present in this image.
COPY data/ /app/data/

RUN mkdir -p /app/final_model /logs/agent

CMD ["/bin/bash"]
