
import os
import json
import streamlit as st

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --------------------------------
# CONFIG
# --------------------------------
load_dotenv()

DB_PATH = "./chroma_db"
PARENTS_JSONL = "./parents_store.jsonl"
COLLECTION_NAME = "pdk_summaries"
ID_KEY = "doc_id"

CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-5-mini-2025-08-07")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-large")

st.set_page_config(page_title="PDK Chatbot Demo", layout="wide")
st.title("🤖 GlobalFoundries PDK Chatbot")
st.caption("Multi-vector RAG: summaries in Chroma, full chunks from a tiny JSONL parent store.")


# --------------------------------
# CACHED LOADERS
# --------------------------------
@st.cache_resource(show_spinner=False)
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
        embedding_function=embeddings,
    )

# --- replace your load_parent_store() with this ---
@st.cache_resource(show_spinner=False)
def load_parent_store():
    from langchain.storage import InMemoryStore
    from langchain_core.documents import Document
    store = InMemoryStore()
    pairs = []

    if os.path.exists(PARENTS_JSONL):
        with open(PARENTS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    doc_id = rec["doc_id"]
                    content = rec.get("content", "")
                    metadata = rec.get("metadata", {}) or {}
                    kind = rec.get("kind", None)
                    if kind:
                        metadata = {**metadata, "kind": kind}
                    # Store a Document so retriever returns Documents later
                    pairs.append((doc_id, Document(page_content=content, metadata=metadata)))
                except Exception:
                    continue

    if pairs:
        store.mset(pairs)
    return store

@st.cache_resource(show_spinner=False)
def build_retriever():
    vectorstore = load_vectorstore()
    store = load_parent_store()
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=ID_KEY,
    )
    return retriever


# --------------------------------
# RAG PIPELINE
# --------------------------------
def format_docs(docs):
    # Turn retrieved Documents into a single context string
    return "\n\n".join([d.page_content for d in docs])

prompt = ChatPromptTemplate.from_template(
    "Answer the question using ONLY the context below. "
    "If you are unsure, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

def build_chain():
    retriever = build_retriever()
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=1)


    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

chain, retriever = build_chain()


# --------------------------------
# UI
# --------------------------------
col_q, col_k = st.columns([3, 1])
with col_q:
    question = st.text_input("Ask a question about the PDK:", value="what the name of  library of basic component reference designs?")
with col_k:
    k = st.slider("Top-k summaries", min_value=2, max_value=12, value=6, step=1)

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                # Run the chain
                answer = chain.invoke(question)
                st.subheader("Answer")
                st.markdown(answer)

                # Show retrieved summaries + their parent chunks for transparency
                vectorstore = load_vectorstore()
                retrieved_children = vectorstore.similarity_search(question, k=k)

                st.subheader("Retrieved (child summaries)")
                for i, d in enumerate(retrieved_children, 1):
                    st.markdown(f"**{i}.** *{d.metadata.get('source_document', 'unknown')}*, "
                                f"page {d.metadata.get('page_number', '—')}, "
                                f"type: {d.metadata.get('element_type', '—')}")
                    st.write(d.page_content)

                # Fetch parents from the docstore by ID and show them
                doc_ids = [d.metadata.get(ID_KEY) for d in retrieved_children if d.metadata.get(ID_KEY)]
                doc_ids = [x for x in doc_ids if x]

                if doc_ids:
                    store = load_parent_store()
                    parents = store.mget(doc_ids)

                    st.subheader("Parent chunks (full text/table)")
                    for i, content in enumerate(parents, 1):
                        if content:
                            st.markdown(f"**Parent {i}**")
                            st.write(content)

            except Exception as e:
                st.error(f"Error: {e}")

st.divider()
st.caption("Tip: run `python process_data.py` first to (re)build the index.")
