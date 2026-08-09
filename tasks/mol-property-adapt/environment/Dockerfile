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

# CPU-only torch. The CPU wheel index keeps the image ~2 GB smaller than default.
RUN uv pip install --system --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.6.0 \
    && uv pip install --system \
        transformers==4.49.0 \
        safetensors==0.4.5 \
        huggingface_hub==0.28.1 \
        rdkit==2024.9.4 \
        scikit-learn==1.5.2 \
        pandas==2.2.3 \
        numpy==1.26.4 \
        scipy==1.14.1

WORKDIR /app

# Bake the base model so the agent starts from a pinned checkpoint rather than
# whatever the Hub serves at run time.
# This checkpoint ships pytorch_model.bin, not safetensors. Files are listed
# explicitly rather than by glob so that training_args.bin - an unnecessary pickle -
# is never pulled in.
ARG BASE_MODEL=DeepChem/ChemBERTa-77M-MLM
ARG BASE_REVISION=ed8a5374f2024ec8da53760af91a33fb8f6a15ff
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${BASE_MODEL}', revision='${BASE_REVISION}', \
                  local_dir='/app/base_model', \
                  allow_patterns=['config.json','pytorch_model.bin','model.safetensors', \
                                  'vocab.json','merges.txt','tokenizer.json', \
                                  'tokenizer_config.json','special_tokens_map.json'])" \
    && rm -rf /app/.hf

# Training data only. The private held-out region is never present in this image.
COPY data/ /app/data/

RUN mkdir -p /app/final_model /logs/agent

CMD ["/bin/bash"]
