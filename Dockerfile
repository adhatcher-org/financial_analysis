FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HOME_LLM_CONFIG=/app/config.toml

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install .

EXPOSE 8123

CMD ["uvicorn", "home_llm.api:app", "--host", "0.0.0.0", "--port", "8123"]
