"""Build the two vector stores for Murshid.

Load -> Split -> Embed -> Store -> Retrieve, run as a standalone step so the
knowledge base can be rebuilt without opening the notebook.

    python -m src.ingest

The embedding model runs locally and needs no API key. It downloads once
(~1 GB) on first use and is cached afterwards.
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PERSIST_DIR = ROOT / "chroma"

# Multilingual, not the English-only all-mpnet-base-v2, because students ask in
# Arabic and English against an English knowledge base with Arabic summaries.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 80

SOURCES = {
    "academic": DATA_DIR / "academic",
    "campus": DATA_DIR / "campus",
}


# --------------------------------------------------------------------------
# 1. LOAD
# --------------------------------------------------------------------------
def load(folder: Path):
    """Read every markdown file in a folder into LangChain Documents."""
    if not folder.exists():
        raise FileNotFoundError(
            f"{folder} does not exist. Are you running from the repo root?"
        )
    loader = DirectoryLoader(
        str(folder),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    # Tag each chunk with a readable source name so citations are possible.
    for d in docs:
        d.metadata["source"] = Path(d.metadata.get("source", "unknown")).name
        d.metadata["collection"] = folder.name
    return docs


# --------------------------------------------------------------------------
# 2. SPLIT
# --------------------------------------------------------------------------
def split(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


# --------------------------------------------------------------------------
# 3-4. EMBED + STORE
# --------------------------------------------------------------------------
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_store(chunks, collection_name: str, embeddings, persist: bool = True):
    """Build ONE Chroma collection. Two separate collections is the whole point:
    if both retrievers query the same store, routing changes nothing."""
    kwargs = {"collection_name": collection_name}
    if persist:
        PERSIST_DIR.mkdir(exist_ok=True)
        kwargs["persist_directory"] = str(PERSIST_DIR)
    return Chroma.from_documents(chunks, embeddings, **kwargs)


def build_all(persist: bool = True):
    """Returns {"academic": retriever, "campus": retriever}."""
    embeddings = get_embeddings()
    retrievers = {}

    for name, folder in SOURCES.items():
        docs = load(folder)
        chunks = split(docs)
        print(f"[{name:9}] loaded {len(docs):2} documents -> {len(chunks):3} chunks")
        store = build_store(chunks, name, embeddings, persist=persist)
        retrievers[name] = store.as_retriever(search_kwargs={"k": 3})

    return retrievers


# --------------------------------------------------------------------------
# 5. RETRIEVE — and prove it works
# --------------------------------------------------------------------------
SMOKE_TESTS = [
    ("academic", "grade appeal deadline", "15"),
    ("academic", "minimum attendance percentage", "75"),
    ("academic", "course withdrawal deadline", "week 10"),
    ("campus", "library hours during finals week", "2:00 AM"),
    ("campus", "where is the IT helpdesk", "Building 4"),
    ("campus", "student portal account lockout", "30 minutes"),
]


def smoke_test(retrievers) -> bool:
    """Ask questions whose answers are verbatim in the documents.

    If a retriever returns nothing, or returns text that does not contain the
    expected fact, the pipeline is broken and nothing downstream is worth
    building. This is the single most common silent capstone failure.
    """
    print("\n--- retrieval smoke test ---")
    ok = True
    for collection, query, expected in SMOKE_TESTS:
        docs = retrievers[collection].invoke(query)
        if not docs:
            print(f"  FAIL  [{collection}] '{query}' -> retriever returned NOTHING")
            ok = False
            continue
        joined = " ".join(d.page_content for d in docs)
        hit = expected.lower() in joined.lower()
        print(f"  {'PASS' if hit else 'FAIL'}  [{collection}] '{query}' "
              f"-> expected {expected!r} in top-3 "
              f"(top hit: {docs[0].metadata.get('source')})")
        ok = ok and hit
    return ok


def isolation_test(retrievers) -> bool:
    """The academic store must NOT know about library hours, and vice versa.

    This is what proves the two stores are genuinely separate -- which is what
    makes the routing decision meaningful in the first place.
    """
    print("\n--- cross-store isolation test ---")
    ok = True

    a = retrievers["academic"].invoke("library opening hours during finals")
    leaked = any("library_hours" in d.metadata.get("source", "") for d in a)
    print(f"  {'PASS' if not leaked else 'FAIL'}  academic store queried for library "
          f"hours -> returned {[d.metadata.get('source') for d in a]}")
    ok = ok and not leaked

    c = retrievers["campus"].invoke("grade appeal deadline form AR-12")
    leaked = any("grade_appeals" in d.metadata.get("source", "") for d in c)
    print(f"  {'PASS' if not leaked else 'FAIL'}  campus store queried for grade "
          f"appeals -> returned {[d.metadata.get('source') for d in c]}")
    ok = ok and not leaked

    return ok


def main() -> int:
    print(f"embedding model: {EMBEDDING_MODEL}")
    print(f"data directory : {DATA_DIR}\n")

    retrievers = build_all(persist=True)

    passed = smoke_test(retrievers)
    passed = isolation_test(retrievers) and passed

    print("\n" + ("=" * 60))
    if passed:
        print("OK — both stores build and retrieve correctly.")
        return 0
    print("FAILED — fix retrieval before building anything on top of it.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
