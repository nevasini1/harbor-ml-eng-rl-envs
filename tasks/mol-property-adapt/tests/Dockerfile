FROM python:3.12-slim

# Single-threaded and fixed-seed: the grader must return the same reward for the
# same checkpoint. Offline mode makes an accidental Hub fetch fail loudly rather
# than silently pulling a different model than the one baked in.
ENV PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/0.9.7/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

RUN uv pip install --system --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.6.0 \
    && uv pip install --system \
        transformers==4.49.0 \
        safetensors==0.4.5 \
        scikit-learn==1.5.2 \
        pandas==2.2.3 \
        numpy==1.26.4

# Grader sources.
COPY grade.py test.sh /tests/
RUN chmod +x /tests/test.sh

# Everything the grader needs, baked in: the verifier runs with no network.
#   grader/private       - held-out test molecules and labels, plus anchors
#   grader/base_model    - reference copy of the base, for the lineage check
#   grader/public_hashes - sha256 of public checkpoints, for the substitution check
COPY grader/ /grader/

# The reward channel is created by test.sh, but make the intent explicit here too.
RUN mkdir -p /logs/verifier && chmod 700 /logs/verifier

WORKDIR /tests
