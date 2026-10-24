FROM python:3.12.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8001 \
    SCORING_MODEL=logreg_v1

WORKDIR /app
COPY scoring/ ./scoring/

USER 10001:10001
EXPOSE 8001
CMD ["python", "-m", "scoring.app"]
