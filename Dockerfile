FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENV AIPIPE_TOKEN=""

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]