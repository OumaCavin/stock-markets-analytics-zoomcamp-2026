# Container image for the crypto perpetuals trading project.
#
#   docker build -t crypto-perps-strategy .
#   docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/output:/app/output" crypto-perps-strategy
#
# The image carries the code and the dependencies only. The raw archives live in a
# mounted volume so that the image stays small and the data is never baked in.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY notebooks ./notebooks
COPY Makefile run_pipeline.sh ./

RUN mkdir -p data/raw data/cache output models

VOLUME ["/app/data", "/app/output"]

ENTRYPOINT ["python", "-m", "src.pipeline"]
CMD ["--regime", "cache"]
