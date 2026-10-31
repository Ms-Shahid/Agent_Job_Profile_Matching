"""
RAG Engine
==========
Vendored integration of Milestone 2 (RAG_Based_Job_Profile_matching).
Source: https://github.com/Ms-Shahid/RAG_Based_Job_Profile_matching

Provides a unified interface for loading all resume PDFs into a persistent
Chroma vector store and performing semantic similarity search.

Resume directory resolution order (first found wins):
  1. RESUMES_DIR environment variable
  2. <this_repo_root>/resumes/          (local copy / git-lfs / symlink)
  3. Sibling clone: ../RAG_Profile_matching/resumes/
  4. Sibling clone: ../RAG_Based_Job_Profile_matching/resumes/

Run `python setup_resumes.py` to automatically clone the resume data from
GitHub if none of the above locations exist.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# Load .env — look in this repo first, then fall back to find_dotenv()
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).parent.resolve()

_env_candidates = [
    _THIS_DIR / ".env",
    _THIS_DIR.parent / ".env",
]
for _p in _env_candidates:
    if _p.exists():
        load_dotenv(str(_p))
        break
else:
    load_dotenv(find_dotenv())


# ---------------------------------------------------------------------------
# Resume directory resolution
# ---------------------------------------------------------------------------

def _find_resumes_dir() -> Path:
    """
    Locate the resumes directory using a priority-ordered search.
    Raises FileNotFoundError with helpful instructions if none found.
    """
    # 1. Explicit env var
    env_dir = os.environ.get("RESUMES_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.exists():
            return p

    # 2. Local resumes/ folder inside this repo
    local = _THIS_DIR / "resumes"
    if local.exists() and any(local.glob("resume_*.pdf")):
        return local

    # 3. Sibling clone — RAG_Profile_matching (original local name)
    sibling1 = _THIS_DIR.parent / "RAG_Profile_matching" / "resumes"
    if sibling1.exists() and any(sibling1.glob("resume_*.pdf")):
        return sibling1

    # 4. Sibling clone — GitHub repo name
    sibling2 = _THIS_DIR.parent / "RAG_Based_Job_Profile_matching" / "resumes"
    if sibling2.exists() and any(sibling2.glob("resume_*.pdf")):
        return sibling2

    raise FileNotFoundError(
        "\n\n❌  Resume PDFs not found.\n\n"
        "Fix any ONE of these:\n"
        "  A) Run:  python setup_resumes.py          (auto-clones the data repo)\n"
        "  B) Set:  RESUMES_DIR=/path/to/resumes     in your .env file\n"
        "  C) Clone https://github.com/Ms-Shahid/RAG_Based_Job_Profile_matching\n"
        "     next to this repo so its resumes/ folder is a sibling directory.\n"
    )


# ---------------------------------------------------------------------------
# Singleton vector-store state
# ---------------------------------------------------------------------------
_docsearch: Optional[Chroma] = None
_resume_metadata: dict[str, dict] = {}


def build_vector_store(force_rebuild: bool = False) -> Chroma:
    """
    Load every resume PDF from the resolved resumes directory into a Chroma
    vector store.  Subsequent calls reuse the cached store unless
    force_rebuild=True.
    """
    global _docsearch, _resume_metadata

    if _docsearch is not None and not force_rebuild:
        return _docsearch

    resumes_dir = _find_resumes_dir()
    pdf_files = sorted(resumes_dir.glob("resume_*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No resume_*.pdf files found in {resumes_dir}")

    print(f"📂  Loading {len(pdf_files)} resumes from {resumes_dir} …")

    all_texts = []
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n")

    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = pdf_file.name
                doc.metadata["resume_id"] = pdf_file.stem
                doc.metadata["candidate_id"] = pdf_file.stem

            chunks = splitter.split_documents(docs)
            all_texts.extend(chunks)

            _resume_metadata[pdf_file.stem] = {
                "resume_id": pdf_file.stem,
                "filename": pdf_file.name,
                "path": str(pdf_file),
                # "resume_1_backend_senior" → "Backend Senior"
                "display_name": " ".join(pdf_file.stem.split("_")[2:]).title(),
            }
        except Exception as exc:
            print(f"  ⚠️  Failed to load {pdf_file.name}: {exc}")

    print(f"✅  Loaded {len(all_texts)} chunks from {len(_resume_metadata)} resumes")

    embeddings = HuggingFaceEmbeddings()
    _docsearch = Chroma.from_documents(all_texts, embeddings)
    print("✅  Vector store built\n")
    return _docsearch


def semantic_search(query: str, top_k: int = 10) -> list[tuple]:
    """
    Semantic similarity search across all resume chunks.

    Returns:
        List of (Document, score) tuples.
        score = cosine distance — lower means more similar.
    """
    global _docsearch
    if _docsearch is None:
        build_vector_store()

    return _docsearch.similarity_search_with_score(query, k=top_k)


def get_all_resume_metadata() -> dict[str, dict]:
    """Return the resume metadata index (triggers build_vector_store lazily)."""
    global _resume_metadata
    if not _resume_metadata:
        build_vector_store()
    return _resume_metadata


def get_resume_path(candidate_id: str) -> Optional[str]:
    """Return the PDF path for a given candidate_id / resume stem."""
    meta = get_all_resume_metadata()
    if candidate_id in meta:
        return meta[candidate_id]["path"]
    for key, val in meta.items():
        if candidate_id.lower() in key.lower():
            return val["path"]
    return None
