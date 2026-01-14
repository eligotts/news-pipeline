# News Pipeline

Consolidated news scraper and ingestion pipeline.

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys and database credentials

# Run full pipeline (scrape + ingest)
python scripts/run_all.py
```

## Scripts

- `scripts/run_all.py` - Run full pipeline (scraper + ingestion steps 1-5)
- `scripts/run_scraper.py` - Run scraper only
- `scripts/run_ingestor.py` - Run ingestor only (with step selection)

## Pipeline Steps

1. **drain_pubsub** - Fetch articles from Pub/Sub and persist
2. **embedding_jobs** - Generate embeddings and cluster articles
3. **stance_jobs** - Extract entity stance/role via LLM
4. **cluster_jobs** - Refresh cluster metadata and generate summaries
5. **topic_jobs** - Discover and assign topics
6. **graph_jobs** - (archived) Entity co-occurrence computation
7. **cache_jobs** - (archived) Redis/local cache warmers

## Configuration

See `.env.example` for all configuration options.
