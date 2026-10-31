"""
Agent Tools
===========

All LangChain @tool functions available to the LangGraph agent, grouped by origin:

  A. Milestone 1 — File System Tools
     Source: https://github.com/Ms-Shahid/LLM-Powered-File-System-Assistant
     Vendored at: milestones/milestone1/fs_tools.py

  B. Milestone 2 — RAG Semantic Search
     Source: https://github.com/Ms-Shahid/RAG_Based_Job_Profile_matching
     Integrated via: rag_engine.py  (self-contained, no path coupling)

  C. New Tools — extract_requirements, compare_candidates,
                 generate_interview_questions
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Env loading — .env in this repo root
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).parent.resolve()
_env_file = _THIS_DIR / ".env"
if _env_file.exists():
    load_dotenv(str(_env_file))
else:
    load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# OpenRouter config
# All LLM calls are routed through OpenRouter using the standard OpenAI SDK
# interface that LangChain's ChatOpenAI supports via base_url + api_key.
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "openai/gpt-4o-mini"  # change to any OpenRouter model slug

# ---------------------------------------------------------------------------
# Milestone 1 import — vendored locally, no sys.path hacks needed
# ---------------------------------------------------------------------------
from milestones.milestone1 import read_file as _read_file
from milestones.milestone1 import list_files as _list_files
from milestones.milestone1 import write_file as _write_file
from milestones.milestone1 import search_in_file as _search_in_file

# ---------------------------------------------------------------------------
# Milestone 2 / RAG engine — path-agnostic wrapper
# ---------------------------------------------------------------------------
from rag_engine import semantic_search as _semantic_search, get_all_resume_metadata

# ---------------------------------------------------------------------------
# Shared LLM (lazy singleton)
# ---------------------------------------------------------------------------
_llm: Optional[ChatOpenAI] = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=OPENROUTER_MODEL,
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            temperature=0.3,
        )
    return _llm


# ===========================================================================
# A. FILE SYSTEM TOOLS  (Milestone 1)
# ===========================================================================

@tool
def read_resume_file(filepath: str) -> str:
    """
    Read a resume or any text file and return its content.
    Provide the full absolute path to the file.
    """
    result = _read_file(filepath)
    if result is None:
        return json.dumps({"error": f"Could not read file: {filepath}"})
    return json.dumps(result)


@tool
def list_resume_files(directory: str, extension: str = "") -> str:
    """
    List all files in a directory, optionally filtered by extension (e.g. '.pdf').
    Returns a JSON array of filenames.
    """
    ext = extension if extension else None
    files = _list_files(directory, ext)
    return json.dumps(files)


@tool
def write_report_file(filepath: str, content: str) -> str:
    """
    Write text content to a file (creates parent directories automatically).
    Returns a JSON status object with a 'success' boolean.
    """
    result = _write_file(filepath, content)
    return json.dumps(result)


@tool
def search_keyword_in_file(filepath: str, keyword: str) -> str:
    """
    Case-insensitive keyword search within a file.
    Returns JSON with 'found' (bool) and 'context' (surrounding text).
    """
    result = _search_in_file(filepath, keyword)
    return json.dumps(result)


# ===========================================================================
# B. RAG SEARCH TOOL  (Milestone 2)
# ===========================================================================

@tool
def search_resumes(query: str, top_k: int = 10) -> str:
    """
    Perform semantic search across ALL resume PDFs using the RAG vector store.

    Args:
        query:  Natural-language description of required skills / experience.
        top_k:  Number of top results to return (default 10).

    Returns:
        JSON list of matches sorted by match_score descending, each with:
        candidate_id, resume_name, match_score (0-100), content_preview.
    """
    raw = _semantic_search(query, top_k=top_k)

    matches: list[dict] = []
    for doc, score in raw:
        # Cosine distance → similarity percentage (lower distance = higher score)
        similarity = round(max(0.0, 1.0 - float(score)) * 100, 1)
        matches.append(
            {
                "candidate_id": doc.metadata.get("candidate_id", "unknown"),
                "resume_name": doc.metadata.get("source", "unknown"),
                "match_score": similarity,
                "content_preview": doc.page_content[:300],
            }
        )

    # Deduplicate — keep best score per candidate
    seen: dict[str, dict] = {}
    for m in matches:
        cid = m["candidate_id"]
        if cid not in seen or m["match_score"] > seen[cid]["match_score"]:
            seen[cid] = m

    ranked = sorted(seen.values(), key=lambda x: x["match_score"], reverse=True)
    return json.dumps(ranked, indent=2)


# ===========================================================================
# C. NEW TOOLS
# ===========================================================================

def _strip_fences(text: str) -> str:
    """Remove accidental ```json ... ``` markdown fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        # parts[1] is the content between first pair of fences
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


@tool
def extract_requirements(jd: str) -> str:
    """
    Parse a job description and return structured requirements as JSON.

    Returns a JSON object with:
      must_have        — list of essential requirements
      nice_to_have     — list of preferred requirements
      key_skills       — top 5-10 skill keywords for semantic search
      experience_years — minimum years required (int or null)
      role_summary     — one-sentence role description
    """
    llm = _get_llm()
    prompt = (
        "You are an expert technical recruiter. Analyse the job description below "
        "and return a JSON object (no markdown fences) with these exact keys:\n\n"
        '  "must_have"       : list of strings — essential, non-negotiable requirements\n'
        '  "nice_to_have"    : list of strings — preferred or bonus requirements\n'
        '  "key_skills"      : list of strings — top 5-10 skill keywords for semantic search\n'
        '  "experience_years": integer or null — minimum years of experience required\n'
        '  "role_summary"    : string — one sentence describing the role\n\n'
        f"Job Description:\n{jd}\n\nReturn ONLY valid JSON, no extra text."
    )
    response = llm.invoke(prompt)
    try:
        return _strip_fences(response.content)
    except Exception as exc:
        return json.dumps({"error": str(exc), "raw": response.content})


@tool
def compare_candidates(candidate_ids: str, job_description: str = "") -> str:
    """
    Head-to-head comparison of a list of candidates.

    Args:
        candidate_ids:   JSON-encoded list of candidate IDs, e.g.
                         '["resume_1_backend_senior", "resume_3_devops_expert"]'
        job_description: Optional JD text to compare against.

    Returns:
        JSON with:
          ranking        — ordered list of candidate IDs (best → worst)
          candidates     — dict keyed by ID, each with strengths, gaps, overall_fit
          recommendation — who to advance and why
    """
    llm = _get_llm()

    try:
        ids: list[str] = json.loads(candidate_ids)
    except json.JSONDecodeError:
        ids = [c.strip() for c in candidate_ids.strip("[]").split(",")]

    # Gather content previews per candidate via semantic search
    candidate_snippets: dict[str, str] = {}
    for cid in ids:
        query = cid.replace("_", " ")
        results = _semantic_search(query, top_k=5)
        snippets = [
            doc.page_content[:400]
            for doc, _ in results
            if doc.metadata.get("candidate_id") == cid
        ]
        if not snippets:
            snippets = [doc.page_content[:200] for doc, _ in results[:3]]
        candidate_snippets[cid] = "\n---\n".join(snippets)

    candidates_text = "\n\n".join(
        f"=== {cid} ===\n{snippet}"
        for cid, snippet in candidate_snippets.items()
    )
    jd_clause = f"\nJob Description:\n{job_description}\n" if job_description else ""

    prompt = (
        "You are a senior technical recruiter performing a head-to-head candidate comparison.\n"
        f"{jd_clause}\n"
        f"Candidate profiles (extracted from their resumes):\n{candidates_text}\n\n"
        "Return a JSON object (no markdown fences) with:\n"
        '  "ranking"       : ordered list of candidate IDs from best to worst fit\n'
        '  "candidates"    : dict keyed by candidate_id, each with:\n'
        '        "strengths"   : list of strength strings\n'
        '        "gaps"        : list of gap strings\n'
        '        "overall_fit" : short paragraph\n'
        '  "recommendation": string — who to advance and why\n\n'
        "Return ONLY valid JSON."
    )
    response = llm.invoke(prompt)
    return _strip_fences(response.content)


@tool
def generate_interview_questions(candidate_id: str, job_description: str = "") -> str:
    """
    Generate tailored interview questions for a specific candidate.

    Args:
        candidate_id:    Resume stem, e.g. 'resume_1_backend_senior'
        job_description: Optional JD text to align questions to.

    Returns:
        JSON with:
          technical_questions   — list of 5 technical depth questions
          behavioral_questions  — list of 3 STAR-format behavioral questions
          screening_questions   — list of 3 initial-screen questions
          areas_to_probe        — gaps / uncertainties worth exploring
    """
    llm = _get_llm()

    results = _semantic_search(candidate_id.replace("_", " "), top_k=5)
    candidate_content = "\n---\n".join(
        doc.page_content[:400]
        for doc, _ in results
        if doc.metadata.get("candidate_id") == candidate_id
    )
    if not candidate_content:
        candidate_content = "\n---\n".join(doc.page_content[:200] for doc, _ in results[:3])

    jd_clause = f"\nJob Description:\n{job_description}\n" if job_description else ""

    prompt = (
        "You are an expert technical interviewer. Based on the candidate profile and "
        "job description below, generate tailored interview questions.\n"
        f"{jd_clause}\n"
        f"Candidate ID: {candidate_id}\n"
        f"Candidate Profile Excerpt:\n{candidate_content}\n\n"
        "Return a JSON object (no markdown fences) with:\n"
        '  "technical_questions"  : list of 5 technical depth questions\n'
        '  "behavioral_questions" : list of 3 STAR-format behavioral questions\n'
        '  "screening_questions"  : list of 3 short initial-screen questions\n'
        '  "areas_to_probe"       : list of gaps or uncertainties worth exploring\n\n'
        "Return ONLY valid JSON."
    )
    response = llm.invoke(prompt)
    return _strip_fences(response.content)


# ---------------------------------------------------------------------------
# Exported tool list used by the agent
# ---------------------------------------------------------------------------
ALL_TOOLS = [
    # Milestone 1 — file system
    read_resume_file,
    list_resume_files,
    write_report_file,
    search_keyword_in_file,
    # Milestone 2 — RAG search
    search_resumes,
    # New tools
    extract_requirements,
    compare_candidates,
    generate_interview_questions,
]
