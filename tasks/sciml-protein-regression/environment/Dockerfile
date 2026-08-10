FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf \
    TRANSFORMERS_CACHE=/models/hf \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    PYTHONHASHSEED=0

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# CUDA build, not the CPU wheel. task.toml requests gpus = 1 and Modal does attach
# one (verified: an L40S on the trial container), but the CPU-only wheel has no
# CUDA support compiled in, so torch.cuda.is_available() returns False regardless
# of the hardware. The first GPU trial ran the whole fine-tune on CPU with the
# GPU sitting at 0% utilisation and 5 MiB used -- visible in the agent log as
# 8,961 steps/epoch, i.e. batch 2 over 17,922 sequences, which is solve.sh's CPU
# branch. Allocating a GPU and writing device code is not enough; the wheel has
# to be able to see it.
RUN pip install --no-cache-dir \
      torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
 && pip install --no-cache-dir \
      transformers==4.49.0 \
      safetensors==0.4.5 \
      huggingface_hub==0.28.1 \
      datasets==3.3.2 \
      accelerate==1.4.0 \
      scikit-learn==1.5.2 \
      scipy==1.14.1 \
      pandas==2.2.3 \
      numpy==1.26.4 \
      sentencepiece==0.2.0

# Pin and pre-bake the base model so agents start from the same checkpoint.
ARG ESM_REV=c731040fcd8d73dceaa04b0a8e6329b345b0f5df
RUN python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "facebook/esm2_t6_8M_UR50D",
    revision="c731040fcd8d73dceaa04b0a8e6329b345b0f5df",
    local_dir="/models/esm2_t6_8M_UR50D",
    local_dir_use_symlinks=False,
)
print("base model ready")
PY

COPY data/ /data/
RUN mkdir -p /app /logs/agent \
 && chmod 755 /app

WORKDIR /app
