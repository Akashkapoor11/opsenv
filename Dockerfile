FROM python:3.11-slim

LABEL maintainer="akashkapoor12004@gmail.com"
LABEL description="OpsEnv — Production Incident Response Environment for AI Agents"

WORKDIR /app

# Critical: set PYTHONPATH so all imports resolve from /app
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Default inference configuration (override with -e at docker run time)
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
ENV HF_TOKEN=""
ENV OPENAI_API_KEY=""

# Install curl for healthcheck probe
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (leverages Docker layer cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source
COPY . /app

# Validate imports and basic functionality at build time
# Catches broken code before deployment — not at runtime
RUN python3 -c "from server.app import app; from server.environment import OpsEnv; from models import IncidentAction; env = OpsEnv(); r = env.reset(); assert r.observation is not None; assert r.observation.task_name == 'classify_severity'; r2 = env.step(IncidentAction(severity='P1')); assert 0.0 <= r2.reward <= 1.0; print('Build-time validation passed')"

EXPOSE 7860

# HF Spaces health probe — must respond within 10s
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", \
    "--workers", "1", "--timeout-keep-alive", "30"]


