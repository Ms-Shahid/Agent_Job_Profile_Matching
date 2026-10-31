"""
app.py — Gradio UI for the AI Profile Matching Agent
=====================================================

Launches a browser-based interface with three tabs:

  Tab 1 · Run Pipeline
        Paste a JD → click "Run Matching Pipeline" → see the ranked shortlist
        and the full generated report in real time.

  Tab 2 · Chat / Refine
        Natural-language follow-up questions once the pipeline has run:
          "Compare the top 3 matches"
          "Why did resume_1_backend_senior rank higher?"
          "Generate interview questions for resume_4_data_scientist"
          "Re-rank candidates prioritising React experience"

  Tab 3 · Interview Questions
        Pick a candidate from a dropdown and get tailored questions instantly.

Usage:
    python app.py
    # then open http://127.0.0.1:7860 in your browser
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import AIMessage, HumanMessage

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).parent.resolve()
_env_file = _THIS_DIR / ".env"
if _env_file.exists():
    load_dotenv(str(_env_file))
else:
    load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# OpenRouter — single place to update the model or key name
# ---------------------------------------------------------------------------
_API_KEY_VAR = "OPENROUTER_API_KEY"


def _api_key_missing() -> bool:
    return not os.environ.get(_API_KEY_VAR)

# ---------------------------------------------------------------------------
# Lazy imports (heavy deps load once the UI is actually used)
# ---------------------------------------------------------------------------
_graph = None
_rag_ready = False
_rag_lock = threading.Lock()

# Shared pipeline state across tabs (one session per server process)
_pipeline_state: dict = {}


def _ensure_rag():
    """Build the vector store once, thread-safely."""
    global _rag_ready
    with _rag_lock:
        if not _rag_ready:
            from rag_engine import build_vector_store
            build_vector_store()
            _rag_ready = True


def _get_graph():
    global _graph
    if _graph is None:
        from matching_agent import build_graph
        _graph = build_graph()
    return _graph


def _last_ai(state: dict) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage):
            return msg.content
    return ""


# ---------------------------------------------------------------------------
# Demo JD
# ---------------------------------------------------------------------------
DEMO_JD = """\
Senior Backend Engineer

We are looking for a Senior Backend Engineer with 5+ years of experience
building scalable microservices and distributed systems.

Required Skills:
- Python (5+ years)
- PostgreSQL or similar relational databases
- AWS (EC2, S3, RDS, Lambda)
- Docker and Kubernetes
- REST API design and CI/CD pipelines

Nice to Have:
- Kafka / message queues
- GraphQL, Terraform
- Team leadership or mentoring experience

Experience: 5+ years | Location: Remote
"""


# ===========================================================================
# Tab 1 helpers — Run Pipeline
# ===========================================================================

def run_pipeline(jd_text: str, progress=gr.Progress()):
    """Run the full matching pipeline and yield incremental status updates."""
    global _pipeline_state

    if not jd_text.strip():
        yield (
            "⚠️ Please paste a job description first.",
            "",
            gr.update(choices=[], value=None),
        )
        return

    if _api_key_missing():
        yield (
            "❌ OPENROUTER_API_KEY not set. Add it to your .env file and restart.",
            "",
            gr.update(choices=[], value=None),
        )
        return

    # ── Initialise RAG ──────────────────────────────────────────────────
    progress(0, desc="Initialising RAG vector store …")
    yield ("🔄 Initialising RAG vector store (first run may take ~60 s) …", "", gr.update())
    _ensure_rag()

    # ── Build initial state ─────────────────────────────────────────────
    from matching_agent import AgentState

    initial_state: AgentState = {
        "messages": [HumanMessage(content=jd_text)],
        "raw_jd": jd_text,
        "parsed_requirements": {},
        "round1_candidates": [],
        "round2_candidates": [],
        "shortlist": [],
        "comparison_report": {},
        "screening_round": 0,
        "final_report": "",
        "human_feedback": "",
        "needs_refinement": False,
        "current_node": "",
    }

    graph = _get_graph()
    config = {"recursion_limit": 60}

    log_lines: list[str] = []
    current_state = initial_state

    node_labels = {
        "parse_jd": "📋 Parsing job description",
        "extract_requirements": "🔍 Extracting requirements",
        "search_resumes": "🔎 Searching resumes (Round 1)",
        "rank_candidates": "📊 Ranking candidates (Round 2)",
        "generate_report": "📝 Generating report (Round 3)",
        "human_feedback": "💬 Entering human feedback mode",
    }

    step_count = len(node_labels)
    step = 0

    for event in graph.stream(current_state, config=config, stream_mode="values"):
        current_state = event
        node = current_state.get("current_node", "")
        if node and node in node_labels:
            step += 1
            progress(step / step_count, desc=node_labels[node])
            msg = _last_ai(current_state)
            if msg:
                log_lines.append(msg)
                yield (
                    "\n\n---\n\n".join(log_lines),
                    "",
                    gr.update(choices=[], value=None),
                )

    # ── Store state for chat tab ─────────────────────────────────────────
    _pipeline_state = current_state

    # ── Build shortlist markdown table ──────────────────────────────────
    shortlist = current_state.get("shortlist", [])
    table_lines = [
        "## 🏆 Final Shortlist\n",
        "| Rank | Candidate | Score | Decision |",
        "|------|-----------|-------|----------|",
    ]
    candidate_ids = []
    for c in shortlist:
        decision = "✅ HIRE" if c["rank"] <= 3 else ("🔄 MAYBE" if c["rank"] <= 6 else "❌ NO HIRE")
        table_lines.append(
            f"| {c['rank']} | `{c['candidate_id']}` | {c['match_score']}% | {decision} |"
        )
        candidate_ids.append(c["candidate_id"])

    report_md = current_state.get("final_report", "")
    shortlist_md = "\n".join(table_lines)

    progress(1.0, desc="Complete ✅")
    yield (
        "\n\n---\n\n".join(log_lines),
        shortlist_md + "\n\n---\n\n" + report_md,
        gr.update(choices=candidate_ids, value=candidate_ids[0] if candidate_ids else None),
    )


# ===========================================================================
# Tab 2 helpers — Chat / Refine
# ===========================================================================

def chat(user_message: str, history: list[list[str]]):
    """
    Handle a conversational follow-up question.
    Runs the conversational_agent node with the user's message injected.
    """
    global _pipeline_state

    if not _pipeline_state:
        history = history or []
        history.append([user_message, "⚠️ Run the matching pipeline first (Tab 1)."])
        return history, ""

    if not user_message.strip():
        return history, ""

    # Inject user message
    new_msg = HumanMessage(content=user_message)
    current_messages = list(_pipeline_state.get("messages", []))
    current_messages.append(new_msg)

    from matching_agent import AgentState

    refinement_state: AgentState = {
        **_pipeline_state,
        "messages": current_messages,
        "human_feedback": user_message,
        "needs_refinement": True,
    }

    graph = _get_graph()
    config = {"recursion_limit": 30}

    for event in graph.stream(refinement_state, config=config, stream_mode="values"):
        _pipeline_state = event

    agent_reply = _last_ai(_pipeline_state)
    history = history or []
    history.append([user_message, agent_reply or "No response generated."])
    return history, ""


# ===========================================================================
# Tab 3 helpers — Interview Questions
# ===========================================================================

def generate_questions(candidate_id: str, jd_text: str):
    """Generate interview questions for a selected candidate."""
    if not candidate_id:
        return "⚠️ Select a candidate from the dropdown."

    if _api_key_missing():
        return "❌ OPENROUTER_API_KEY not set."

    _ensure_rag()

    from tools import generate_interview_questions as _gen_q

    try:
        raw = _gen_q.invoke({"candidate_id": candidate_id, "job_description": jd_text or ""})
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw  # Return as-is if not valid JSON

    lines = [f"## 🎤 Interview Questions for `{candidate_id}`\n"]

    def _section(title: str, items: list[str]) -> list[str]:
        out = [f"### {title}", ""]
        for i, q in enumerate(items, 1):
            out.append(f"{i}. {q}")
        out.append("")
        return out

    lines += _section("🔧 Technical Questions", data.get("technical_questions", []))
    lines += _section("🧠 Behavioral Questions (STAR format)", data.get("behavioral_questions", []))
    lines += _section("📋 Screening Questions", data.get("screening_questions", []))

    probes = data.get("areas_to_probe", [])
    if probes:
        lines += ["### 🔍 Areas to Probe", ""]
        lines += [f"- {p}" for p in probes]

    return "\n".join(lines)


# ===========================================================================
# Gradio UI layout
# ===========================================================================

_CSS = """
.gr-button-primary { background: #4f46e5 !important; }
.output-log { font-family: monospace; font-size: 0.85rem; }
footer { display: none !important; }
"""

with gr.Blocks(title="AI Profile Matching Agent") as demo:

    gr.Markdown(
        """
# 🤖 AI Profile Matching Agent
### Powered by LangGraph · RAG · OpenAI · Milestone 1 & 2

> Built on [LLM-Powered File System Assistant](https://github.com/Ms-Shahid/LLM-Powered-File-System-Assistant)
> and [RAG-Based Job Profile Matching](https://github.com/Ms-Shahid/RAG_Based_Job_Profile_matching)
        """
    )

    # ── Shared state: candidate dropdown synced between Tab 1 & Tab 3 ──
    _candidate_state = gr.State([])

    with gr.Tabs():

        # ================================================================
        # TAB 1 — Run Pipeline
        # ================================================================
        with gr.Tab("🚀 Run Pipeline"):
            gr.Markdown(
                "Paste a job description and run the full 3-round screening pipeline."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    jd_input = gr.Textbox(
                        label="Job Description",
                        placeholder="Paste the full job description here …",
                        lines=18,
                        value=DEMO_JD,
                    )
                    with gr.Row():
                        run_btn = gr.Button("▶  Run Matching Pipeline", variant="primary")
                        clear_btn = gr.Button("🗑  Clear")

                with gr.Column(scale=1):
                    pipeline_log = gr.Markdown(
                        label="Pipeline Log",
                        value="*Pipeline output will appear here …*",
                        elem_classes=["output-log"],
                    )

            report_output = gr.Markdown(
                label="Candidate Report",
                value="*Shortlist & full report will appear here after the pipeline runs …*",
            )

            # Shared candidate dropdown (also used in Tab 3)
            candidate_dropdown_p1 = gr.Dropdown(
                label="Candidates found (use in Tab 3 → Interview Questions)",
                choices=[],
                interactive=True,
                visible=True,
            )

            run_btn.click(
                fn=run_pipeline,
                inputs=[jd_input],
                outputs=[pipeline_log, report_output, candidate_dropdown_p1],
            )
            clear_btn.click(
                fn=lambda: (DEMO_JD, "*Pipeline output will appear here …*", "*Report will appear here …*"),
                outputs=[jd_input, pipeline_log, report_output],
            )

        # ================================================================
        # TAB 2 — Chat / Refine
        # ================================================================
        with gr.Tab("💬 Chat & Refine"):
            gr.Markdown(
                """
Ask natural-language follow-up questions after the pipeline has run.

**Example queries:**
- *"Compare the top 3 candidates side by side"*
- *"Why did resume_1_backend_senior rank higher than resume_3_devops_expert?"*
- *"Re-rank with a focus on AWS experience"*
- *"Generate interview questions for resume_4_data_scientist"*
- *"Find candidates with React and 3+ years experience"*
                """
            )

            chatbot = gr.Chatbot(
                label="Agent Chat",
                height=450,
            )

            with gr.Row():
                chat_input = gr.Textbox(
                    label="",
                    placeholder="Ask a follow-up question …",
                    scale=9,
                    show_label=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            send_btn.click(
                fn=chat,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input],
            )
            chat_input.submit(
                fn=chat,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input],
            )

        # ================================================================
        # TAB 3 — Interview Questions
        # ================================================================
        with gr.Tab("🎤 Interview Questions"):
            gr.Markdown(
                "Select a candidate and optionally paste the JD to get tailored interview questions."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    candidate_dropdown_t3 = gr.Dropdown(
                        label="Select Candidate",
                        choices=[],
                        interactive=True,
                        info="Run the pipeline first to populate this list.",
                    )
                    jd_for_questions = gr.Textbox(
                        label="Job Description (optional — improves question relevance)",
                        lines=6,
                        placeholder="Paste JD here …",
                    )
                    gen_btn = gr.Button("🎤  Generate Questions", variant="primary")

                with gr.Column(scale=1):
                    questions_output = gr.Markdown(
                        value="*Interview questions will appear here …*"
                    )

            gen_btn.click(
                fn=generate_questions,
                inputs=[candidate_dropdown_t3, jd_for_questions],
                outputs=[questions_output],
            )

            # Sync candidate list from Tab 1 to Tab 3 when pipeline runs
            candidate_dropdown_p1.change(
                fn=lambda v: gr.update(choices=candidate_dropdown_p1.choices, value=v),
                inputs=[candidate_dropdown_p1],
                outputs=[candidate_dropdown_t3],
            )

    gr.Markdown(
        """
---
**Sources:**
[Milestone 1](https://github.com/Ms-Shahid/LLM-Powered-File-System-Assistant) ·
[Milestone 2](https://github.com/Ms-Shahid/RAG_Based_Job_Profile_matching) ·
[Agent (this repo)](https://github.com/Ms-Shahid/Agent_profile_matching)
        """
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,           # set True to get a public gradio.live link
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="indigo"),
        css=_CSS,
    )
