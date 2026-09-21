# Chapter 3 Task 3: OpenSearch Document Retrieval

This project keeps the supplied ingestion pipeline and client unchanged. It turns `query.py` into an OpenAI-compatible RAG service that:

1. Embeds the latest user question.
2. Searches the OpenSearch vector index.
3. Applies exact metadata filters when requested.
4. Builds a bounded prompt from the selected chunks.
5. Calls the separate local SLM service.
6. Returns the answer with a retrieval trace.
7. Writes the same trace to `outputs/retrieval_traces/`.

## Project layout

```text
.
├── client.py                 # Unchanged OpenAI-compatible client
├── ingest.py                 # Unchanged ingestion pipeline
├── query.py                  # OpenSearch retrieval and RAG REST service
├── slm_server_flask.py       # Existing SLM server; default port changed to 8001
├── requirements.txt
└── common
    ├── config.py
    ├── embeddings.py
    ├── logging.py
    └── opensearch_client.py
```

## 1. Install dependencies

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the shared dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GGUF inference, install the llama.cpp Python package that matches your platform:

```bash
python -m pip install llama-cpp-python
```

For MLX inference on Apple Silicon, install MLX instead:

```bash
python -m pip install mlx-lm
```

## 2. Configure OpenSearch

The defaults expect OpenSearch at `127.0.0.1:9200` without TLS. Override them when your deployment differs:

```bash
export OPENSEARCH_HOST=127.0.0.1
export OPENSEARCH_PORT=9200
export OPENSEARCH_USER=admin
export OPENSEARCH_PASSWORD=admin
export OPENSEARCH_USE_SSL=false
export OPENSEARCH_VERIFY_CERTS=false
export OPENSEARCH_INDEX=bbc-vector-chunks
```

Configure the embedding model used by both ingestion and retrieval:

```bash
export EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

The ingestion and query processes must use the same embedding model. A different model can produce a different vector dimension or an incompatible vector space.

## 3. Prepare the document corpus

The unchanged ingestion program expects this directory structure:

```text
bbc/
├── business/
│   ├── 001.txt
│   └── 002.txt
├── entertainment/
├── politics/
├── sport/
└── tech/
```

Each first-level directory becomes the `category` metadata field. The relative file path becomes the `path` field, and the file name becomes the `title` field.

## 4. Ingest the documents

Run the supplied ingestion pipeline:

```bash
python ingest.py \
  --data-dir ./bbc \
  --chunk-size 2048 \
  --chunk-overlap 256 \
  --batch-size 32
```

The program creates the index when needed, generates normalized embeddings, and writes these fields:

```text
path
category
title
text
embedding
```

Re-running ingestion replaces chunks for each source path instead of accumulating duplicate copies.

The supplied `ingest.py` preserves the document text as read from UTF-8 files. It does not add a separate whitespace or Unicode text-normalization pass. The shared embedding helper normalizes the generated vectors before indexing.

## 5. Start the local SLM service

The RAG service uses port `8000`, so the local SLM service defaults to port `8001` in this project:

```bash
export SLM_SERVER_PORT=8001
python slm_server_flask.py
```

Check the model service:

```bash
curl http://127.0.0.1:8001/health
```

## 6. Start the RAG service

In another terminal:

```bash
source .venv/bin/activate

export RAG_SERVER_PORT=8000
export SLM_BASE_URL=http://127.0.0.1:8001/v1
export SLM_MODEL=local-llm
export RAG_TOP_K=5
export RAG_NUM_CANDIDATES=25

python query.py
```

Check the complete service path:

```bash
curl http://127.0.0.1:8000/health
```

A healthy response requires both OpenSearch and the downstream SLM endpoint.

## 7. Query through the unchanged client

The supplied client already defaults to the RAG service on port `8000`:

```bash
python client.py --question "How much did Google purchase Windsurf for?"
```

The client prints the generated answer. The RAG service writes the corresponding trace under:

```text
outputs/retrieval_traces/<trace-id>.json
```

## 8. Query with filters and evaluation labels

The OpenAI chat endpoint accepts optional RAG fields in addition to the standard request fields:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer local-agent' \
  -d '{
    "model": "local-rag-agent",
    "messages": [
      {
        "role": "user",
        "content": "How much did Google purchase Windsurf for?"
      }
    ],
    "temperature": 0.0,
    "top_k": 5,
    "num_candidates": 25,
    "metadata_filters": {
      "category": "business"
    },
    "expected_source_paths": [
      "business/expected-document.txt"
    ]
  }' | python -m json.tool
```

Supported exact-match metadata filters are:

```text
category
path
title
```

For multiple accepted values, provide a list:

```json
{
  "metadata_filters": {
    "category": ["business", "tech"]
  }
}
```

Known-answer evaluation can use either source paths or chunk IDs:

```json
{
  "expected_source_paths": ["business/001.txt"],
  "expected_chunk_ids": ["stable-opensearch-document-id"]
}
```

## Retrieval trace fields

Each response contains `retrieval_trace_id` and `retrieval_trace`. The saved trace includes:

- The question, index, `top_k`, candidate count, and metadata filters.
- Embedding, search, retrieval, generation, and end-to-end latency.
- Index document count and stored byte size.
- Selected chunk IDs, ranks, similarity scores, metadata, source paths, and text previews.
- Filter correctness for every selected result.
- Exact duplicate count and duplicate rate based on chunk content hashes.
- Average, highest, and lowest similarity scores.
- Known-answer coverage, precision at `k`, and recall at `k` when expected sources are supplied.
- Downstream SLM settings and token usage.

When no expected source labels are supplied, precision and recall are `null`. Similarity scores remain available as a retrieval signal, but they are not the same as human relevance judgments.

## Additional Resources

Here are some additional resources for subjects the chapter assumes you are familiar with, but might need a refresher on.

### RAG: Data preparation for chunking

- [The Ultimate Guide to Chunking Strategies for RAG Applications](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089)
- [Chunking Strategies for RAG Systems](https://github.com/deepshamenghani/chunking_strategies_langchain)
