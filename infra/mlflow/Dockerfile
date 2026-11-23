# MLflow tracking server with the two drivers the official image lacks:
# psycopg2 for the Postgres backend store, boto3 for the S3 artifact store.
# Same MLflow version as uv.lock, so clients and server agree.
FROM python:3.11.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv

RUN uv pip install --system --no-cache \
        mlflow==3.18.0 \
        psycopg2-binary==2.9.13 \
        boto3==1.43.117

RUN useradd --create-home --uid 10001 mlflow
USER mlflow

EXPOSE 5000
