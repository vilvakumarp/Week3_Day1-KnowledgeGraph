import os
import json
import re
import time
import streamlit as st
import requests
from bs4 import BeautifulSoup
from typing import List, Optional, cast, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph
from neo4j import GraphDatabase, Query
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. SETUP & CONFIGURATION
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Type casting to avoid Pylance "str | None" warnings
openai_api_key = str(OPENAI_API_KEY) if OPENAI_API_KEY else ""
uri = str(NEO4J_URI) if NEO4J_URI else ""
user = str(NEO4J_USERNAME) if NEO4J_USERNAME else ""
pwd = str(NEO4J_PASSWORD) if NEO4J_PASSWORD else ""

# Initialize LLM and Embeddings
LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
llm = ChatOpenAI(model=LLM_MODEL, api_key=SecretStr(openai_api_key))
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=SecretStr(openai_api_key))

# Initialize Neo4j Driver
driver = GraphDatabase.driver(uri, auth=(user, pwd))

# Vector DB Path
VECTOR_DB_PATH = "faiss_index"

def get_vector_db():
    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

# 2. DATA MODELS
class Triplet(BaseModel):
    subject: str = Field(description="The entity the relationship starts from")
    relation: str = Field(description="The relationship between the subject and object")
    object: str = Field(description="The entity the relationship ends at")

class KnowledgeGraph(BaseModel):
    triplets: List[Triplet]

# 3. WORKFLOW FUNCTIONS
def load_json_data(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return json.dumps(data, indent=2)

def scrape_portal_data(url: str) -> str:
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        schemes = []
        table = soup.find('table')
        if table:
            rows = table.find_all('tr')
            for row in rows[1:]:
                cols = row.find_all('td')
                if cols:
                    schemes.append({"scheme_name": cols[0].text.strip(), "details": " ".join([c.text.strip() for c in cols])})
        return json.dumps(schemes, indent=2)
    except Exception as e:
        return f"Error scraping: {e}"

def extract_graph_data(text: str):
    structured_llm = llm.with_structured_output(KnowledgeGraph)
    prompt = ChatPromptTemplate.from_template(
        "Extract a knowledge graph from the following text. "
        "Identify key entities and the relationships between them. Return them as triplets. "
        "Text: {text}"
    )
    chain = prompt | structured_llm
    try:
        result = chain.invoke({"text": text})
        if isinstance(result, KnowledgeGraph):
            return result
        # If the LLM returns a dict or generic BaseModel, convert it to KnowledgeGraph
        return KnowledgeGraph(**result if isinstance(result, dict) else result.dict())
    except Exception:
        return KnowledgeGraph(triplets=[])

def store_in_graph(kg_data: KnowledgeGraph):
    with driver.session() as session:
        for triplet in kg_data.triplets:
            # Relationship types are Cypher identifiers, so spaces and punctuation
            # must be converted to underscores before being inserted into the query.
            relation = re.sub(r"[^A-Za-z0-9_]", "_", triplet.relation.strip().upper())
            relation = re.sub(r"_+", "_", relation).strip("_") or "RELATED_TO"
            if relation[0].isdigit():
                relation = f"REL_{relation}"

            query = f"""
            MERGE (s:Entity {{name: $subject}})
            MERGE (o:Entity {{name: $object}})
            MERGE (s)-[:{relation}]->(o)
            """
            session.run(
                Query(cast(Any, query)),
                subject=triplet.subject.strip(),
                object=triplet.object.strip(),
            ).consume()

def to_safe_cypher(response: Any) -> str:
    """Extract a read-only Cypher query from the model response."""
    text = str(response).strip()
    fenced = re.search(r"```(?:cypher)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    query = fenced.group(1).strip() if fenced else text

    # Models sometimes prefix the query with an explanation.
    match = re.search(r"\b(MATCH|OPTIONAL MATCH|WITH)\b", query, re.IGNORECASE)
    if match:
        query = query[match.start():].strip()

    query = query.rstrip(";").strip()
    if not re.match(r"^(MATCH|OPTIONAL MATCH|WITH)\b", query, re.IGNORECASE):
        return "MATCH (n:Entity) RETURN n LIMIT 25"
    return query

def add_to_vector_db(text: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # Use type ignore to silence the LiteralString warning from Pylance
    docs = text_splitter.create_documents([text]) # type: ignore
    vdb = get_vector_db()

    # A FAISS index is tied to the embedding dimension used to create it.
    # Rebuild an older index automatically when changing embedding providers/models.
    if vdb is not None and docs:
        current_dimension = len(embeddings.embed_query(docs[0].page_content))
        if vdb.index.d != current_dimension:
            vdb = None

    if vdb is None:
        vdb = FAISS.from_documents(docs, embeddings)
    else:
        vdb.add_documents(docs)
    vdb.save_local(VECTOR_DB_PATH)

def graph_enquiry(question: str):
    try:
        cypher_prompt = ChatPromptTemplate.from_template(
            "Return ONLY a valid read-only Cypher query, with no explanation or markdown. "
            "Nodes are labeled Entity. Use MATCH and RETURN only. "
            "Question: {question}"
        )
        cypher_query = (cypher_prompt | llm).invoke({"question": question}).content
        # Ensure cypher_query is treated as a string for .replace()
        query_str = to_safe_cypher(cypher_query)
        with driver.session() as session:
            result = session.run(Query(cast(Any, query_str)))
            context = [record.values() for record in result]
        if not context: return "No graph data found."
        final_prompt = ChatPromptTemplate.from_template(
            "Based on the following graph data: {context}, answer the user question: {question}"
        )
        return (final_prompt | llm).invoke({"context": context, "question": question}).content
    except Exception as e:
        return f"Graph error: {e}"

def vector_enquiry(question: str):
    vdb = get_vector_db()
    if vdb is None: return "Vector DB not initialized."
    docs = vdb.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in docs])
    final_prompt = ChatPromptTemplate.from_template(
        "Based on the following text chunks: {context}, answer the user question: {question}"
    )
    return (final_prompt | llm).invoke({"context": context, "question": question}).content

def hybrid_enquiry(question: str):
    v_context = ""
    vdb = get_vector_db()
    if vdb:
        docs = vdb.similarity_search(question, k=2)
        v_context = "\n".join([d.page_content for d in docs])
    
    g_context = ""
    try:
        cypher_prompt = ChatPromptTemplate.from_template(
            "Return ONLY a valid read-only Cypher query, with no explanation or markdown. "
            "Nodes are labeled Entity. Use MATCH and RETURN only. "
            "Question: {question}"
        )
        cypher_query = (cypher_prompt | llm).invoke({"question": question}).content
        query_str = to_safe_cypher(cypher_query)
        with driver.session() as session:
            result = session.run(Query(cast(Any, query_str)))
            g_context = str([record.values() for record in result])
    except:
        g_context = "No graph data found."

    final_prompt = ChatPromptTemplate.from_template(
        "You are a Hybrid RAG system. Use the Vector Context (semantic) and Graph Context (structural) to answer.\n\n"
        "Vector Context: {v_context}\n\nGraph Context: {g_context}\n\nQuestion: {question}"
    )
    return (final_prompt | llm).invoke({"v_context": v_context, "g_context": g_context, "question": question}).content

# 4. STREAMLIT UI
st.set_page_config(page_title="Knowledge Graph RAG", layout="wide")
st.title("🌐 Hybrid Knowledge Graph RAG")

tab1, tab2 = st.tabs(["📥 Ingestion", "🔍 Enquiry"])

with tab1:
    st.header("Data Ingestion")
    source = st.radio("Select Data Source:", ["Local JSON", "Government Portal", "Upload PDF"])
    
    if source == "Local JSON":
        files = ["data/raw/agrisnet_schemes.json", "data/raw/extraction_report.json"]
        if st.button("Start Local Ingestion"):
            with st.status("Ingesting Local Data...", expanded=True) as status:
                for file in files:
                    if os.path.exists(file):
                        status.write(f"Loading {file}...")
                        text = load_json_data(file)
                        chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
                        for idx, chunk in enumerate(chunks[:10]):
                            status.write(f"Processing {file} chunk {idx+1}/10...")
                            res = extract_graph_data(chunk)
                            store_in_graph(res)
                            add_to_vector_db(chunk)
                            time.sleep(2)
                status.update(label="Ingestion Complete!", state="complete", expanded=False)
            st.success("Local data integrated!")

    elif source == "Government Portal":
        url = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
        if st.button("Start Portal Ingestion"):
            with st.status("Scraping Portal...", expanded=True) as status:
                status.write("Fetching data...")
                text = scrape_portal_data(url)
                chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
                for idx, chunk in enumerate(chunks[:10]):
                    status.write(f"Processing chunk {idx+1}/10...")
                    res = extract_graph_data(chunk)
                    store_in_graph(res)
                    add_to_vector_db(chunk)
                    time.sleep(2)
                status.update(label="Portal Ingestion Complete!", state="complete", expanded=False)
            st.success("Portal data integrated!")

    else:
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file and st.button("Process PDF"):
            with st.status("Processing PDF...", expanded=True) as status:
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                status.write("Extracting text...")
                loader = PyPDFLoader("temp.pdf")
                pages = loader.load()
                text = "\n".join([p.page_content for p in pages])
                chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
                for idx, chunk in enumerate(chunks[:10]):
                    status.write(f"Processing PDF chunk {idx+1}/10...")
                    res = extract_graph_data(chunk)
                    store_in_graph(res)
                    add_to_vector_db(chunk)
                    time.sleep(2)
                status.update(label="PDF Ingestion Complete!", state="complete", expanded=False)
            st.success("PDF integrated!")

with tab2:
    st.header("Graph Enquiry & Comparison")
    user_q = st.text_input("Ask a question:")
    if user_q:
        with st.spinner("Searching..."):
            columns = st.columns(3)
            methods = [
                ("Vector-only", vector_enquiry),
                ("Graph-only", graph_enquiry),
                ("Hybrid", hybrid_enquiry),
            ]

            for column, (method_name, method_function) in zip(columns, methods):
                with column:
                    st.subheader(method_name)
                    try:
                        answer = method_function(user_q)
                        st.markdown(answer)
                    except Exception as e:
                        error_text = str(e)
                        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                            st.warning("Gemini API quota exhausted. Wait for reset or use another API key.")
                        else:
                            st.error(f"Request failed: {e}")

