FROM python:3.12-slim
WORKDIR /app
COPY perplexity-ai /deps/perplexity-ai
RUN pip install --no-cache-dir /deps/perplexity-ai
COPY perplexity-proxy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY perplexity-proxy .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
