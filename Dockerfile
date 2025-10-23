FROM python:3.11-slim AS base

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
# Mount or copy artifacts & demographics at runtime:
#   -v /host/model:/app/model
#   -v /host/data:/app/data  

EXPOSE 8000
# CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", \
#      "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4", "--timeout", "60"]
