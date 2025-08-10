FROM python:3.9-slim

WORKDIR /app

# Install runtime deps (with curl for healthcheck)
RUN apt-get update && apt-get install -y \
    gcc g++ curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
 && apt-get purge -y gcc g++ \
 && apt-get autoremove -y

COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Environment variables
ENV PYTHONPATH=/app \
    FLASK_APP=api_server.py \
    TRANSFORMERS_CACHE=/app/.cache

EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD gunicorn api_server:create_app --bind 0.0.0.0:${PORT:-5000}
