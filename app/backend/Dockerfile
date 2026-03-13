FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv/app/backend

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

ARG TORCH_VERSION=2.6.0

COPY backend/requirements.txt /tmp/requirements.txt
RUN grep -v '^torch==' /tmp/requirements.txt > /tmp/requirements.docker.txt \
    && pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}+cpu" \
    && pip install -r /tmp/requirements.docker.txt

COPY backend /srv/app/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
