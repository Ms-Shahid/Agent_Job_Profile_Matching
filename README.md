# AI Profile Matching Agent

A LangGraph-powered agentic workflow that screens, ranks, and explains
candidate matches for any job description — with a Gradio web UI.

Integrates two upstream milestone projects:

| Milestone | Repo | Role |
|-----------|------|------|
| 1 – File System Tools | [LLM-Powered-File-System-Assistant](https://github.com/Ms-Shahid/LLM-Powered-File-System-Assistant) | Read / write / search files |
| 2 – RAG Search | [RAG_Based_Job_Profile_matching](https://github.com/Ms-Shahid/RAG_Based_Job_Profile_matching) | Semantic search over 45 resume PDFs |

---

## Architecture

```
START
  ↓
Parse JD
  ↓
Extract Requirements        ← extract_requirements() tool
  ↓
Search Resumes              ← search_resumes() RAG tool  (Milestone 2)
  ↓
Rank Candidates             ← compare_candidates() tool
  ↓
Generate Report             ← write_report_file() tool   (Milestone 1)
  ↓
Human Feedback Loop
  ↓  (loop if refinement requested)
Conversational Agent        ← all tools available
  ↓
END
```

### Tools

| Tool | Origin |
|------|--------|
| `read_resume_file` | Milestone 1 |
| `list_resume_files` | Milestone 1 |
| `write_report_file` | Milestone 1 |
| `search_keyword_in_file` | Milestone 1 |
| `search_resumes` | Milestone 2 (RAG) |
| `extract_requirements` | New |
| `compare_candidates` | New |
| `generate_interview_questions` | New |

---

## Quick Start

### 1 — Clone this repo

```bash
git clone https://github.com/Ms-Shahid/Agent_profile_matching.git
cd Agent_profile_matching
```

### 2 — Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3 — Add your OpenAI API key

```bash
cp .env.example .env
# Edit .env and set:
#   OPENAI_API_KEY=sk-...
```

### 4 — Download the resume dataset

```bash
python setup_resumes.py
```

This clones the 45 resume PDFs from the Milestone 2 repo into `resumes/`.
Only needs to be run once.

> **Already have the repos locally?**
> If `RAG_Profile_matching` or `RAG_Based_Job_Profile_matching` is cloned
> as a sibling directory next to `Agent_profile_matching`, the agent finds
> the resumes automatically — no setup needed.

### 5 — Launch the Gradio UI  *(recommended)*

```bash
python app.py
# Opens http://127.0.0.1:7860
```

### 5 — Or use the CLI

```bash
python main.py --demo          # built-in demo JD
python main.py                 # paste your own JD interactively
python main.py --jd-file jd.txt
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | ✅ yes | — | OpenRouter API key — get one at [openrouter.ai/keys](https://openrouter.ai/keys) |
| `RESUMES_DIR` | no | auto-resolved | Override path to resume PDFs |
| `PORT` | no | `7860` | Gradio server port |

---

## Project Structure

```
Agent_profile_matching/
├── app.py                  # Gradio web UI (3 tabs)
├── main.py                 # CLI entry point
├── matching_agent.py       # LangGraph StateGraph + all nodes
├── tools.py                # All 8 LangChain @tool functions
├── rag_engine.py           # RAG vector store wrapper (Milestone 2)
├── setup_resumes.py        # One-time resume data setup
├── requirements.txt
├── .env.example
├── milestones/
│   ├── milestone1/
│   │   ├── fs_tools.py     # Vendored from Milestone 1
│   │   └── __init__.py
│   └── milestone2/
│       └── __init__.py     # Resume data loaded via rag_engine.py
├── resumes/                # Created by setup_resumes.py (gitignored)
└── reports/                # Generated reports (gitignored)
```

---

## Gradio UI Tabs

### Tab 1 — Run Pipeline
Paste any job description → click **Run Matching Pipeline** → see:
- Live node-by-node progress log
- Final ranked shortlist table
- Full Markdown report (Hire / Maybe / No-Hire per candidate)

### Tab 2 — Chat & Refine
Natural-language follow-ups after the pipeline:
- *"Compare the top 3 candidates side by side"*
- *"Why did resume_1_backend_senior rank higher?"*
- *"Re-rank with a focus on React experience"*
- *"Generate interview questions for resume_4_data_scientist"*

### Tab 3 — Interview Questions
Select a candidate from the dropdown and get:
- 5 tailored technical questions
- 3 STAR behavioral questions
- 3 initial screening questions
- Areas to probe (identified gaps)

---

## Multi-Round Screening (Part C)

| Round | What happens |
|-------|-------------|
| **Round 1** | RAG semantic search → retrieves top 20 → advances top 10 |
| **Round 2** | `compare_candidates` deep analysis on top 10 → ranked shortlist |
| **Round 3** | Hire / No-Hire recommendation + full explainability report |

---

## Requirements

- Python 3.10+
- OpenAI API key
- ~2 GB RAM (for HuggingFace embeddings + Chroma)
- Internet access on first run (to download embedding model weights)
