# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install .

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

RUN groupadd -r aura && useradd -r -g aura -d /app -s /sbin/nologin aura

WORKDIR /app

COPY --from=builder /install /usr/local
COPY aura/ ./aura/
COPY verifier/ ./verifier/
COPY config/ ./config/

RUN chown -R aura:aura /app

USER aura

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "aura.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
