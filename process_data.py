# must be first
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass


import os
import uuid
import json
from typing import List, Tuple

from dotenv import load_dotenv
from unstructured.partition.auto import partition

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma

# -----------------------------
# CONFIG
# -----------------------------
load_dotenv()  # for OPENAI_API_KEY

DATA_PATH = "data"                       # folder with your PDFs etc.
DB_PATH = "./chroma_db"                  # Chroma persistence dir (created if missing)
PARENTS_JSONL = "./parents_store.jsonl"  # tiny file that stores full chunks
COLLECTION_NAME = "pdk_summaries"
ID_KEY = "doc_id"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".pptx", ".csv"}

CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-5-mini-2025-08-07")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-large")

# -----------------------------
# INIT
# -----------------------------
model = ChatOpenAI(model=CHAT_MODEL, temperature=1)
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_PATH,
    embedding_function=embeddings,
)

prompt = ChatPromptTemplate.from_template(
    "You are an assistant tasked with summarizing tables and text.\n"
    "Write a concise, retrieval-friendly summary (1–3 sentences).\n\n"
    "Element:\n{element}"
)
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()


# -----------------------------
# HELPERS
# -----------------------------
def is_supported_file(name: str) -> bool:
    name = name.lower()
    return any(name.endswith(ext) for ext in SUPPORTED_EXTENSIONS)

def elem_type_name(elem) -> str:
    return elem.__class__.__name__

def is_table(elem) -> bool:
    return elem_type_name(elem).lower() == "table"

def clean_text(s: str) -> str:
    if not s:
        return ""
    return " ".join(s.split()).strip()

def write_parents_jsonl(pairs: List[Tuple[str, dict]]):
    # pairs: [(doc_id, {"content": str, "metadata": {...}, "kind": "text|table"})]
    with open(PARENTS_JSONL, "a", encoding="utf-8") as f:
        for doc_id, payload in pairs:
            record = {"doc_id": doc_id, **payload}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# -----------------------------
# MAIN
# -----------------------------
def main():
    os.makedirs(DB_PATH, exist_ok=True)

    total_files = 0
    total_children = 0

    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            if not is_supported_file(file):
                continue

            total_files += 1
            file_path = os.path.join(root, file)
            print(f"\n--- Processing {file_path} ---")

            try:
                raw_elements = partition(
                    filename=file_path,
                    infer_table_structure=True,
                    strategy="fast",
                )

                if not raw_elements:
                    print("No elements found, skipping.")
                    continue

                texts, tables = [], []

                for elem in raw_elements:
                    txt = clean_text(getattr(elem, "text", None))
                    if not txt:
                        continue

                    md = getattr(elem, "metadata", None)
                    page = getattr(md, "page_number", None) if md else None

                    meta = {
                        "element_type": elem_type_name(elem),
                        "page_number": page,
                        "source_document": os.path.basename(file_path),
                    }

                    if is_table(elem):
                        tables.append((txt, meta))
                    else:
                        texts.append((txt, meta))

                print(f"Found {len(texts)} text chunks and {len(tables)} tables.")

                # Summarize and index TEXTS
                if texts:
                    text_ids = [str(uuid.uuid4()) for _ in texts]
                    text_summaries = summarize_chain.batch(
                        [t for t, _ in texts], {"max_concurrency": 5}
                    )

                    summary_docs = [
                        Document(
                            page_content=clean_text(text_summaries[i]) or "(empty summary)",
                            metadata={**texts[i][1], ID_KEY: text_ids[i]},
                        )
                        for i in range(len(texts))
                    ]
                    vectorstore.add_documents(summary_docs)
                    total_children += len(summary_docs)

                    parent_pairs = []
                    for i, (content, meta) in enumerate(texts):
                        parent_pairs.append(
                            (text_ids[i], {"content": content, "metadata": meta, "kind": "text"})
                        )
                    write_parents_jsonl(parent_pairs)

                # Summarize and index TABLES
                if tables:
                    table_ids = [str(uuid.uuid4()) for _ in tables]
                    table_summaries = summarize_chain.batch(
                        [t for t, _ in tables], {"max_concurrency": 5}
                    )

                    summary_docs = [
                        Document(
                            page_content=clean_text(table_summaries[i]) or "(empty summary)",
                            metadata={**tables[i][1], ID_KEY: table_ids[i]},
                        )
                        for i in range(len(tables))
                    ]
                    vectorstore.add_documents(summary_docs)
                    total_children += len(summary_docs)

                    parent_pairs = []
                    for i, (content, meta) in enumerate(tables):
                        parent_pairs.append(
                            (table_ids[i], {"content": content, "metadata": meta, "kind": "table"})
                        )
                    write_parents_jsonl(parent_pairs)

                vectorstore.persist()
                print(f"✔ Indexed parents with {total_children} total child summaries so far.")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print("\n✅ Done.")
    print(f"Files processed: {total_files}")
    print(f"Chroma DB: {os.path.abspath(DB_PATH)}")
    print(f"Parents JSONL: {os.path.abspath(PARENTS_JSONL)}")


if __name__ == "__main__":
    main()
