FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/grader/hf \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    PYTHONHASHSEED=0

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
      transformers==4.49.0 \
      safetensors==0.4.5 \
      huggingface_hub==0.28.1 \
      scipy==1.14.1 \
      pandas==2.2.3 \
      numpy==1.26.4

# Base model for anti-substitution checks (baked; verifier has no network).
ARG ESM_REV=c731040fcd8d73dceaa04b0a8e6329b345b0f5df
RUN python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "facebook/esm2_t6_8M_UR50D",
    revision="c731040fcd8d73dceaa04b0a8e6329b345b0f5df",
    local_dir="/grader/base_model",
    local_dir_use_symlinks=False,
)
print("grader base model ready")
PY

# Entire tests/ build context is baked in; includes private_test/ and grade.py.
COPY . /tests/
RUN mkdir -p /logs/verifier /app \
 && chmod 700 /logs/verifier \
 && chmod 755 /tests/test.sh

WORKDIR /tests
