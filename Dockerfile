FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data

RUN mkdir -p models
RUN python -c "from src.pipeline.train import run; run()"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]