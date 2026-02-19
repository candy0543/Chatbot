## RAG Chatbot 

**Setup**
- Create and activate a Python env, then install deps:

```bash
pip install -r requirements.txt
```

- Copy env template and set variables:

```bash
copy .env.example .env
```

- Place documents under: data/source
	- Supported: .txt, .md, .pdf, .docx


This creates/updates a Chroma index at data/index.
**Ingest from Notion (recursive)**

- Set environment variables in `.env` or your shell:

	- `NOTION_API_KEY`: Notion internal integration token
	- One of:
		- `NOTION_DATABASE_ID`: Database ID to crawl and ingest
		- `NOTION_ROOT_PAGE_ID`: Root page ID to crawl recursively (follows child pages and databases)
	- For Vertex AI usage:
		- `LLM_PROVIDER=vertex`
		- `GOOGLE_PROJECT`, `GOOGLE_LOCATION`
		- `VERTEX_EMBEDDING_MODEL=text-embedding-004` (default)
		- Optional Vector Search: `USE_VERTEX_VECTOR_SEARCH=true`, plus `VERTEX_INDEX_ID`, `VERTEX_INDEX_ENDPOINT_ID`

```bash
python -m app.ingestion.notion_ingest
```
**Ingest from swagger**
```bash
python -m app.ingestion.swagger_ingest
```
**Upload on Google Cloud**

```bash
python -m app.insgestion.gcs_upload
```

**Run Chainlit UI**

```bash
chainlit run app/ui/app.py -w
```

Open the URL shown in terminal and start chatting.


**Google Vertex AI**
- Install GCP dependencies (already in requirements).
- Auth: set `GOOGLE_APPLICATION_CREDENTIALS` to your service account JSON or set `GOOGLE_CREDENTIALS` in `.env`.
- Set `GOOGLE_PROJECT`, `GOOGLE_LOCATION`, `VERTEX_CHAT_MODEL` (e.g., `gemini-1.5-pro`), `VERTEX_EMBEDDING_MODEL` (e.g., `text-embedding-004`).
- To use Vertex AI Vector Search, set `USE_VERTEX_VECTOR_SEARCH=true` and provide `VERTEX_INDEX_ID` and `VERTEX_INDEX_ENDPOINT_ID` for an existing index/endpoint. If not set, the app uses local Chroma.

**Sync documents to Google Cloud Storage (Vertex AI Search)**
- Set `GCS_BUCKET_NAME` (and optional `GCS_BUCKET_PREFIX`) in `.env` so the app knows where to upload raw docs.
- Run `python -m app.ingestion.gcs_upload --bucket <your-bucket> --prefix vertex/search` to copy everything under `data/source` to `gs://<bucket>/<prefix>`.
- Use `--dry-run` to preview, `--force` to re-upload unchanged files, and `--delete-extra` to remove remote files not present locally.
- Point your Vertex AI Search data store at the same bucket/prefix so new files become available after a sync.


**Notes**
- OpenAI/Azure keys are required unless you switch provider and adjust embeddings.
- Re-run ingestion after adding/updating documents.





