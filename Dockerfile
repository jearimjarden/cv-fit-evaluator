FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl

RUN useradd -m appuser

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/storage /app/logs && \
    chown -R appuser:appuser /app/storage /app/logs

COPY . .

USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.core.main_api:app", "--host", "0.0.0.0", "--port", "8000"]