# Knowledge Graph RAG

A Streamlit application that extracts knowledge-graph triplets from JSON, web, or PDF data, stores them in Neo4j Aura, and optionally stores document chunks in FAISS for vector and hybrid search.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create `.env` in the project folder:

```env
OPENAI_API_KEY=your-openai-key
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password

# Optional LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=knowledge-graph-rag
```

Never commit `.env` or share its keys. Rotate credentials that have been exposed.

## Run the application

```powershell
.\venv\Scripts\Activate.ps1
streamlit run main.py
```

Open the URL shown by Streamlit, usually `http://localhost:8501`.

In the **Ingestion** tab, choose Local JSON, Government Portal, or Upload PDF. Wait for the success message. Graph triplets are written to Neo4j; document chunks are written to FAISS.

## Test Neo4j Aura

Open the Aura instance, select **Query**, and change the database selector at the top from:

```text
system
```

to:

```text
neo4j
```

The `system` database is for administration and does not allow `MATCH` queries over application data.

Run this query to display nodes and relationships:

```cypher
MATCH (s:Entity)-[r]->(o:Entity)
RETURN s, r, o
LIMIT 100
```

Select the **Graph** result view to see the nodes and arrows.

To inspect relationship names as text:

```cypher
MATCH (s:Entity)-[r]->(o:Entity)
RETURN s.name AS source, type(r) AS relationship, o.name AS target
LIMIT 100
```

To check whether nodes exist:

```cypher
MATCH (n:Entity)
RETURN n
LIMIT 100
```

To count nodes and relationships:

```cypher
MATCH (n:Entity)
RETURN count(n) AS node_count;

MATCH ()-[r]->()
RETURN count(r) AS relationship_count;
```

If nodes exist but relationships do not, rerun ingestion and check the Streamlit status messages. Relationship names are normalized, for example `provides information about` becomes `PROVIDES_INFORMATION_ABOUT`.

## Compare retrieval methods

In the **Enquiry** tab, enter a question. Three results appear side by side:

- **Vector-only** searches FAISS semantically.
- **Graph-only** generates a read-only Cypher query and searches Neo4j.
- **Hybrid** combines FAISS and Neo4j context.

## LangSmith

With `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` set, LangChain traces are sent to the project named by `LANGSMITH_PROJECT`. Set tracing to `false` to disable it.
