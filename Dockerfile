# Production REST API image — lightweight (gate core = pydantic only; API adds
# fastapi/uvicorn). NO LLM/embedding/torch — the gate consumes run-logs, not models.
FROM python:3.11-slim

WORKDIR /app

# Install API deps first (layer cached unless requirements change).
# requirements-api.txt = requirements-gate.txt (pydantic) + fastapi + uvicorn.
COPY requirements-gate.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# App code + the rule catalog that suggestions parse at request time.
COPY app/ ./app/
COPY docs/remediation_catalog.md ./docs/remediation_catalog.md

EXPOSE 8000

# uvicorn serving the FastAPI app. Clients POST run-log/attribution dir paths.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
