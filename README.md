# Home LLM

Local-first question answering over personal financial and insurance documents.

Home LLM indexes documents you already have, stores structured metadata in PostgreSQL, optionally stores embeddings in Qdrant, and can ask Ollama to answer questions using retrieved evidence.

## What It Does

- Ingests OCR text and metadata from Paperless-ngx
- Can also ingest local `.pdf`, `.txt`, `.md`, and `.csv` files if you explicitly configure local folder scanning
- Extracts lightweight facts such as balances, payments, policy numbers, and due dates
- Supports keyword search over indexed excerpts
- Supports retrieval-augmented question answering with source-backed results
- Exposes both a CLI and a FastAPI service

This project is a retrieval system, not a trained foundation model. Your original files stay where they already live.

## Architecture

- Source documents: Paperless-ngx over HTTP, or local folders visible to the runtime if you explicitly enable folder-based ingestion
- PostgreSQL: documents, chunks, and extracted facts
- Qdrant: optional vector search index
- Ollama: optional local chat and embedding models

If Qdrant is disabled, search falls back to PostgreSQL-backed text search. If Ollama is disabled, `ask` returns retrieved excerpts without generating a final answer.

## Privacy

- Raw files are read from folders you configure
- Original files are not copied into the project
- Parsed text, chunks, and extracted facts are stored in PostgreSQL
- Embeddings and chunk payloads are stored in Qdrant only if enabled
- No cloud API is required by default

## Requirements

- Python 3.11+
- PostgreSQL
- Optional: Ollama
- Optional: Qdrant
- Optional: Paperless-ngx

## Quick Start

1. Create a virtual environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install .
```

For development tools:

```bash
pip install ".[dev]"
```

2. Copy the sample config:

```bash
cp config.example.toml config.toml
```

3. Configure PostgreSQL.

The app needs PostgreSQL connection settings. You can either provide a full DSN or split the connection into separate environment variables:

```bash
export HOME_LLM_POSTGRES_USER="home_llm_user"
export HOME_LLM_POSTGRES_HOST="192.168.50.4"
export HOME_LLM_POSTGRES_PORT="5432"
export HOME_LLM_POSTGRES_PASSWORD="YOUR_POSTGRES_PASSWORD"
export HOME_LLM_POSTGRES_DATABASE="home_llm"
```

If you prefer, `HOME_LLM_POSTGRES_DSN` still works too.

4. Configure your ingestion source.

For a Paperless-first setup:

```toml
[paperless]
enabled = true
base_url = "https://paperless.example.com"
token = "YOUR_PAPERLESS_TOKEN"
page_size = 100

[ingest]
source_dirs = []
chunk_size = 1400
chunk_overlap = 200
extensions = [".pdf", ".txt", ".md", ".csv"]
```

When Paperless is enabled, `ingest` pulls OCR text from Paperless and ignores local `source_dirs` for that run.

If you want local folder scanning instead, use settings like:

```toml
[ingest]
source_dirs = [
  "/Users/yourname/Documents/Finance/*/*/2026",
  "/Users/yourname/Documents/Insurance/*/2026",
]
chunk_size = 1400
chunk_overlap = 200
extensions = [".pdf", ".txt", ".md", ".csv"]
```

`source_dirs` supports exact paths and glob patterns.

If you run folder-based ingestion in Docker on Unraid, these must be container-visible paths from mounted shares, for example:

```toml
[ingest]
source_dirs = [
  "/data/finance/banks/*/2026",
  "/data/finance/investments/*/2026",
  "/data/insurance/*/2026",
]
```

5. Optional: configure Ollama:

```toml
[ollama]
base_url = "http://127.0.0.1:11434"
chat_model = "llama3.1:8b"
embedding_model = "embeddinggemma:latest"
enabled = true
```

If you want generated answers or vector search, make sure the configured models are available in your Ollama instance.

6. Optional: enable Qdrant:

```toml
[qdrant]
enabled = true
base_url = "http://YOUR-QDRANT-HOST:6333"
collection = "financial_documents"
api_key = ""
```

7. Run ingestion:

```bash
python -m home_llm --config config.toml ingest
```

8. Query the index:

```bash
python -m home_llm --config config.toml ask "What recurring debt payments do I appear to have?"
python -m home_llm --config config.toml search mortgage
python -m home_llm --config config.toml facts --limit 25
```

## Configuration Reference

Paperless-first example config:

```toml
[postgres]
dsn = ""
dsn_env_var = "HOME_LLM_POSTGRES_DSN"
user = ""
user_env_var = "HOME_LLM_POSTGRES_USER"
host = ""
host_env_var = "HOME_LLM_POSTGRES_HOST"
port = ""
port_env_var = "HOME_LLM_POSTGRES_PORT"
password = ""
password_env_var = "HOME_LLM_POSTGRES_PASSWORD"
database = ""
database_env_var = "HOME_LLM_POSTGRES_DATABASE"

[ollama]
base_url = "http://127.0.0.1:11434"
chat_model = "llama3.1:8b"
embedding_model = "embeddinggemma:latest"
enabled = true

[qdrant]
enabled = true
base_url = "http://YOUR-QDRANT-HOST:6333"
collection = "financial_documents"
api_key = ""

[ingest]
source_dirs = []
chunk_size = 1400
chunk_overlap = 200
extensions = [".pdf", ".txt", ".md", ".csv"]

[query]
top_k = 6

[paperless]
enabled = true
base_url = "https://paperless.example.com"
token = "YOUR_PAPERLESS_TOKEN"
page_size = 100
```

Notes:

- `postgres.dsn_env_var` defaults to `HOME_LLM_POSTGRES_DSN`
- `postgres.user_env_var`, `host_env_var`, `port_env_var`, `password_env_var`, and `database_env_var` default to `HOME_LLM_POSTGRES_USER`, `HOME_LLM_POSTGRES_HOST`, `HOME_LLM_POSTGRES_PORT`, `HOME_LLM_POSTGRES_PASSWORD`, and `HOME_LLM_POSTGRES_DATABASE`
- `qdrant.enabled = false` disables vector search if you want keyword-only retrieval
- `ollama.enabled = false` disables answer generation
- `query.top_k` is the default retrieval depth for `ask` and `search`
- `paperless.enabled = true` makes `ingest` use Paperless instead of local `source_dirs`

## CLI

Top-level usage:

```bash
python -m home_llm --config config.toml <command>
```

Commands:

- `ingest`: index configured documents
- `ask <question>`: retrieve evidence and optionally generate an answer
- `search <query>`: return matching excerpts only
- `facts`: list extracted facts

Examples:

```bash
python -m home_llm --config config.toml ask "What insurance policies are mentioned?"
python -m home_llm --config config.toml ask "What balances are shown?" --top-k 8
python -m home_llm --config config.toml search escrow --top-k 5
python -m home_llm --config config.toml facts --limit 50
```

`--config` is a top-level argument, so it must appear before the subcommand.

## FastAPI

Run the API:

```bash
source .venv/bin/activate
export HOME_LLM_CONFIG="$(pwd)/config.toml"
uvicorn home_llm.api:app --host 0.0.0.0 --port 8123
```

The API reads configuration from `HOME_LLM_CONFIG` and falls back to `config.toml`.

Endpoints:

- `GET /health`
- `POST /ask`
- `GET /facts`
- `GET /search`
- `POST /search`

Examples:

```bash
curl http://127.0.0.1:8123/health
```

```bash
curl -X POST http://127.0.0.1:8123/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What recurring debt payments do I appear to have?","top_k":6}'
```

```bash
curl "http://127.0.0.1:8123/search?query=mortgage&top_k=5"
```

```bash
curl -X POST http://127.0.0.1:8123/search \
  -H "Content-Type: application/json" \
  -d '{"query":"mortgage","top_k":5}'
```

```bash
curl "http://127.0.0.1:8123/facts?limit=25"
```

## Unraid

For Unraid, the simplest model is:

- run Home LLM as a container
- mount your config file into the container
- point PostgreSQL, Qdrant, Ollama, and optionally Paperless at reachable hostnames or IPs on your network
- ingest documents from Paperless-ngx

Example `config.toml` pattern for Unraid:

```toml
[postgres]
dsn = ""
dsn_env_var = "HOME_LLM_POSTGRES_DSN"
user = ""
user_env_var = "HOME_LLM_POSTGRES_USER"
host = ""
host_env_var = "HOME_LLM_POSTGRES_HOST"
port = ""
port_env_var = "HOME_LLM_POSTGRES_PORT"
password = ""
password_env_var = "HOME_LLM_POSTGRES_PASSWORD"
database = ""
database_env_var = "HOME_LLM_POSTGRES_DATABASE"

[ollama]
base_url = "http://YOUR-OLLAMA-HOST:11434"
chat_model = "llama3.1:8b"
embedding_model = "embeddinggemma:latest"
enabled = true

[qdrant]
enabled = true
base_url = "http://YOUR-QDRANT-HOST:6333"
collection = "financial_documents"
api_key = ""

[ingest]
source_dirs = []
chunk_size = 1400
chunk_overlap = 200
extensions = [".pdf", ".txt", ".md", ".csv"]

[query]
top_k = 6

[paperless]
enabled = true
base_url = "https://paperless.example.com"
token = "YOUR_PAPERLESS_TOKEN"
page_size = 100
```

Important for Unraid:

- for a Paperless-based setup, you do not need to mount document shares into the container
- split PostgreSQL variables are usually easier to manage in Unraid than embedding a DSN in the file
- if you enable Qdrant, make sure the embedding model configured in Ollama is available to the Home LLM container over the network
- if you disable Paperless and switch to folder-based ingestion later, then `source_dirs` must match mounted paths inside the container

Example Unraid Postgres parameters:

```text
HOME_LLM_POSTGRES_USER=home_llm_user
HOME_LLM_POSTGRES_HOST=192.168.50.4
HOME_LLM_POSTGRES_PORT=5432
HOME_LLM_POSTGRES_PASSWORD=YOUR_POSTGRES_PASSWORD
HOME_LLM_POSTGRES_DATABASE=home_llm
```

## Docker

Build the image:

```bash
docker build -t home-llm:latest .
```

Run it:

```bash
docker run -d \
  --name home-llm \
  -p 8123:8123 \
  -e HOME_LLM_CONFIG=/app/config.toml \
  -e HOME_LLM_POSTGRES_USER="home_llm_user" \
  -e HOME_LLM_POSTGRES_HOST="192.168.50.4" \
  -e HOME_LLM_POSTGRES_PORT="5432" \
  -e HOME_LLM_POSTGRES_PASSWORD="YOUR_POSTGRES_PASSWORD" \
  -e HOME_LLM_POSTGRES_DATABASE="home_llm" \
  -v "$(pwd)/config.toml:/app/config.toml:ro" \
  --restart unless-stopped \
  home-llm:latest
```

Or with Compose:

```bash
docker compose -f docker-compose.home-llm.yml up -d --build
```

The container starts the FastAPI service on port `8123`.

For a Paperless-based Unraid setup, you usually only need to mount the config file:

```yaml
services:
  home-llm:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: home-llm
    restart: unless-stopped
    ports:
      - "8123:8123"
    environment:
      HOME_LLM_CONFIG: /app/config.toml
      HOME_LLM_POSTGRES_USER: home_llm_user
      HOME_LLM_POSTGRES_HOST: 192.168.50.4
      HOME_LLM_POSTGRES_PORT: "5432"
      HOME_LLM_POSTGRES_PASSWORD: YOUR_POSTGRES_PASSWORD
      HOME_LLM_POSTGRES_DATABASE: home_llm
    volumes:
      - ./config.toml:/app/config.toml:ro
```

If you switch to folder-based ingestion later, add read-only mounts for those shares and update `source_dirs` to the mounted container paths.

## Make Targets

The repository includes a `Makefile` for common tasks:

- `make install`
- `make install-dev`
- `make lint`
- `make typecheck`
- `make test`
- `make coverage`
- `make security`
- `make check`
- `make api`
- `make docker-build`

## Ingestion Behavior

- Files with unsupported extensions are ignored
- Duplicate resolved paths are skipped during local discovery
- Files with no extractable text are reported as skipped
- Ingest clears and replaces stored chunks and facts for documents it reprocesses
- If Qdrant is enabled, document vectors are replaced for the reingested file

Scanned PDFs without embedded text still need OCR unless you ingest from Paperless.

## Good First Questions

- What accounts and balances are mentioned in my latest statements?
- What recurring payments appear across these documents?
- Which insurance policies and coverage amounts are visible?
- What due dates or premium amounts are explicitly stated?
- What mortgage or escrow amounts can you find?

## Limitations

- It can miss facts or misread noisy documents
- Answer quality depends on document quality and model quality
- It does not replace a CPA, attorney, financial planner, or insurance agent
- Important numbers should always be checked against the cited source material

## Project Layout

- `src/home_llm/`: application code
- `tests/`: test suite
- `config.example.toml`: sample configuration
- `Dockerfile`: API container image
- `docker-compose.home-llm.yml`: compose example for the API service
