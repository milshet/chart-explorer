# The endpoint (task 4.3) as an image (task 4.4). Built from the repository root by compose.yaml:
#
#   MODEL_DIR=~/merged/run_a_r16 docker compose up --build
#
# Three rules, each one a way a container ends up serving a model nobody measured:
#   1. Installs from the lockfile and resolves nothing. requirements-serve.txt is a verbatim subset of
#      requirements-vm03.lock, installed with --no-deps and checked with pip check: a newer transformers cannot
#      slip in, and a missing dependency fails the build instead of resolving silently.
#   2. Weights are mounted, never baked. The verified merge (~7 GB) is mounted read-only at /model; the image holds
#      code only, so it can be rebuilt without re-pushing weights, and what it serves is identified by the mounted
#      merge_manifest.json, not by whatever was copied in at build time.
#   3. Healthy means VERIFIED. The healthcheck reads /health, which reports the merge manifest; a container serving
#      an unverified checkpoint is unhealthy even while it answers.

# Python 3.12.3: the interpreter requirements-vm03.lock was generated with.
FROM python:3.12.3-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    HOME=/tmp

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-deps -r requirements-serve.txt && pip check

# Code only: .dockerignore keeps data/, results/, logs/ and .git out of the build context.
COPY scripts/ scripts/

# Fail the build, not the first request, if the serving code cannot import its stack.
RUN python -c "import torch, torchvision, transformers, accelerate, fastapi, uvicorn, PIL; \
print('torch', torch.__version__, 'transformers', transformers.__version__)" \
 && python -c "import sys; sys.path.insert(0, 'scripts'); from serve import app, service, decoding, generation, healthcheck"

RUN useradd --create-home --uid 10001 serve
USER serve

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD ["python", "scripts/serve/healthcheck.py"]

# The served model was trained with reasoning targets (Run A); the adapter's train_config.json is not in the
# container, so the output style is passed explicitly.
ENTRYPOINT ["python", "scripts/serve/app.py", "--model", "/model", "--host", "0.0.0.0", "--port", "8000", \
            "--targets", "reasoning"]
