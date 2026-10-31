"""
Matching Agent  (Part A)
========================

LangGraph-based agentic workflow that implements the full candidate
screening pipeline:

    START
      ↓
    Parse JD
      ↓
    Extract Requirements
      ↓
    Search Resumes
      ↓
    Rank Candidates
      ↓
    Generate Report
      ↓
    Human Feedback Loop
      ↓
    END

The agent also supports:
  • Multi-round screening (Round 1 → top-10, Round 2 → deep analysis,
    Round 3 → Hire / No-Hire recommendation)          (Part C)
  • Conversational natural-language queries             (Part B)
  • Iterative refinement via the human-feedback node   (Part B)
  • Per-candidate explainability in the final report   (Part C)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any, Optional

from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from tools import ALL_TOOLS

# ---------------------------------------------------------------------------
# Load env — .env in this repo root only
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).parent.resolve()
_env_file = _THIS_DIR / ".env"
if _env_file.exists():
    load_dotenv(str(_env_file))
else:
    load_dotenv(find_dotenv())


# ===========================================================================
# Agent State
# ===========================================================================

class AgentState(TypedDict):
    # ── Core message history ─────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Job description understanding ────────────────────────────────────
    raw_jd: str                          # Original JD text supplied by the user
    parsed_requirements: dict            # Output of extract_requirements()

    # ── Candidate management ─────────────────────────────────────────────
    round1_candidates: list[dict]        # Top-10 from Round 1 screening
    round2_candidates: list[dict]        # Top-10 after deep analysis
    shortlist: list[dict]                # Final ranked shortlist w/ reasoning
    comparison_report: dict              # Head-to-head comparison data

    # ── Multi-round screening progress ───────────────────────────────────
    screening_round: int                 # 1, 2, or 3

    # ── Output ───────────────────────────────────────────────────────────
    final_report: str                    # Markdown report written to disk
    human_feedback: str                  # Latest feedback from the human node

    # ── Control ──────────────────────────────────────────────────────────
    needs_refinement: bool               # True → loop back from feedback
    current_node: str                    # Debug: last node executed


# ===========================================================================
# LLM + tool binding
# ===========================================================================

# ---------------------------------------------------------------------------
# OpenRouter config — same provider/model as tools.py
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "openai/gpt-4o-mini"

_llm = ChatOpenAI(
    model=OPENROUTER_MODEL,
    base_url=OPENROUTER_BASE_URL,
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    temperature=0.3,
    streaming=True,
)

_llm_with_tools = _llm.bind_tools(ALL_TOOLS)

SYSTEM_PROMPT = """You are an expert AI recruiting assistant powered by a LangGraph
agentic workflow. You have access to tools for:

  • Reading / writing files (Milestone 1 – file system tools)
  • Semantic resume search (Milestone 2 – RAG vector store with 45 resumes)
  • Extracting structured requirements from job descriptions
  • Comparing candidates head-to-head
  • Generating tailored interview questions

Your job is to help recruiters find the best candidates for a given role by:
  1. Parsing and understanding the job description
  2. Searching resumes semantically
  3. Ranking and shortlisting candidates with clear reasoning
  4. Generating a detailed explainability report
  5. Iterating based on human feedback

Always be specific, cite evidence from the resumes, and explain your reasoning."""


# ===========================================================================
# Node implementations
# ===========================================================================

def node_parse_jd(state: AgentState) -> dict:
    """
    Node 1 – Parse JD
    Extracts the job description from the latest human message if not
    already stored in state.raw_jd.
    """
    raw_jd = state.get("raw_jd", "")

    # If JD not yet stored, pull from the last human message
    if not raw_jd:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                raw_jd = msg.content
                break

    ai_msg = AIMessage(
        content=f"📋 **Parsing Job Description** …\n\nExtracted JD ({len(raw_jd)} chars). Proceeding to requirement extraction."
    )
    return {
        "raw_jd": raw_jd,
        "current_node": "parse_jd",
        "messages": [ai_msg],
    }


def node_extract_requirements(state: AgentState) -> dict:
    """
    Node 2 – Extract Requirements
    Calls the LLM with extract_requirements tool to get structured requirements.
    """
    raw_jd = state.get("raw_jd", "")
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Call the `extract_requirements` tool with the following job description "
                f"and return its result:\n\n{raw_jd}"
            )
        ),
    ]

    response = _llm_with_tools.invoke(messages)

    # Execute the tool call immediately if the model made one
    parsed_requirements: dict = {}
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_node = ToolNode(ALL_TOOLS)
        tool_messages = tool_node.invoke({"messages": [response]})
        tool_result_msg = tool_messages["messages"][-1]
        try:
            parsed_requirements = json.loads(tool_result_msg.content)
        except (json.JSONDecodeError, AttributeError):
            parsed_requirements = {"raw": str(tool_result_msg.content)}
    else:
        # LLM returned inline JSON
        try:
            parsed_requirements = json.loads(response.content)
        except (json.JSONDecodeError, AttributeError):
            parsed_requirements = {"raw": str(response.content)}

    summary_lines = []
    if "must_have" in parsed_requirements:
        summary_lines.append("**Must-have:** " + ", ".join(parsed_requirements["must_have"][:5]))
    if "nice_to_have" in parsed_requirements:
        summary_lines.append("**Nice-to-have:** " + ", ".join(parsed_requirements["nice_to_have"][:5]))
    if "experience_years" in parsed_requirements:
        summary_lines.append(f"**Experience:** {parsed_requirements['experience_years']}+ years")

    summary = "\n".join(summary_lines) if summary_lines else json.dumps(parsed_requirements, indent=2)

    ai_msg = AIMessage(
        content=f"✅ **Requirements Extracted**\n\n{summary}\n\nProceeding to resume search …"
    )
    return {
        "parsed_requirements": parsed_requirements,
        "current_node": "extract_requirements",
        "messages": [ai_msg],
    }


def node_search_resumes(state: AgentState) -> dict:
    """
    Node 3 – Search Resumes
    Uses the RAG semantic search tool to find matching candidates.
    Round 1: top-20 retrieved, top-10 selected.
    """
    req = state.get("parsed_requirements", {})
    raw_jd = state.get("raw_jd", "")

    # Build a rich search query from requirements
    key_skills = req.get("key_skills", [])
    must_have = req.get("must_have", [])
    search_query = " ".join(key_skills + must_have[:3]) or raw_jd[:300]

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Call `search_resumes` with the query:\n'{search_query}'\nand top_k=20."
            )
        ),
    ]

    response = _llm_with_tools.invoke(messages)

    round1_candidates: list[dict] = []

    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_node = ToolNode(ALL_TOOLS)
        tool_messages = tool_node.invoke({"messages": [response]})
        tool_result_msg = tool_messages["messages"][-1]
        try:
            results = json.loads(tool_result_msg.content)
            round1_candidates = results[:20]  # cap at 20
        except (json.JSONDecodeError, AttributeError):
            pass

    # Keep top-10 for Round 1
    top10 = round1_candidates[:10]

    ai_msg = AIMessage(
        content=(
            f"🔍 **Resume Search Complete (Round 1)**\n\n"
            f"Screened candidates — top 10 selected from {len(round1_candidates)} matches:\n"
            + "\n".join(
                f"  {i+1}. `{c['candidate_id']}` — score {c['match_score']}%"
                for i, c in enumerate(top10)
            )
        )
    )
    return {
        "round1_candidates": round1_candidates,
        "screening_round": 1,
        "current_node": "search_resumes",
        "messages": [ai_msg],
    }


def node_rank_candidates(state: AgentState) -> dict:
    """
    Node 4 - Rank Candidates
    Round 2: deep analysis of top-10 candidates from Round 1.
    Produces a ranked shortlist with per-candidate reasoning.
    """
    top10 = (state.get("round1_candidates") or [])[:10]
    req = state.get("parsed_requirements", {})
    raw_jd = state.get("raw_jd", "")

    if not top10:
        return {
            "shortlist": [],
            "screening_round": 2,
            "current_node": "rank_candidates",
            "messages": [AIMessage(content="⚠️ No candidates to rank.")],
        }

    candidate_ids = json.dumps([c["candidate_id"] for c in top10])

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Call `compare_candidates` with:\n"
                f"  candidate_ids: {candidate_ids}\n"
                f"  job_description: {raw_jd[:500]}\n\n"
                f"Then use the result to produce a ranked shortlist."
            )
        ),
    ]

    response = _llm_with_tools.invoke(messages)

    comparison_report: dict = {}
    shortlist: list[dict] = []

    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_node = ToolNode(ALL_TOOLS)
        tool_messages = tool_node.invoke({"messages": [response]})
        tool_result_msg = tool_messages["messages"][-1]
        try:
            comparison_report = json.loads(tool_result_msg.content)
        except (json.JSONDecodeError, AttributeError):
            comparison_report = {}

    # Build shortlist from comparison ranking + candidate scores
    ranking = comparison_report.get("ranking", [c["candidate_id"] for c in top10])
    candidates_detail = comparison_report.get("candidates", {})
    score_map = {c["candidate_id"]: c["match_score"] for c in top10}

    for i, cid in enumerate(ranking):
        detail = candidates_detail.get(cid, {})
        shortlist.append(
            {
                "rank": i + 1,
                "candidate_id": cid,
                "match_score": score_map.get(cid, 0),
                "strengths": detail.get("strengths", []),
                "gaps": detail.get("gaps", []),
                "overall_fit": detail.get("overall_fit", ""),
            }
        )

    # Round 2 – top-10 deep analysis done; round2_candidates = shortlist
    ai_msg = AIMessage(
        content=(
            "📊 **Candidate Ranking Complete (Round 2 – Deep Analysis)**\n\n"
            + "\n".join(
                f"  {c['rank']}. `{c['candidate_id']}` — {c['match_score']}%"
                + (f" | Strengths: {', '.join(c['strengths'][:2])}" if c["strengths"] else "")
                for c in shortlist[:5]
            )
            + ("\n  …" if len(shortlist) > 5 else "")
        )
    )
    return {
        "round2_candidates": shortlist,
        "shortlist": shortlist,
        "comparison_report": comparison_report,
        "screening_round": 2,
        "current_node": "rank_candidates",
        "messages": [ai_msg],
    }


def node_generate_report(state: AgentState) -> dict:
    """
    Node 5 – Generate Report
    Round 3: produces Hire / No-Hire recommendations, per-candidate
    explainability, improvement suggestions, and writes a Markdown report
    to disk via the write_report_file tool.
    """
    shortlist = state.get("shortlist", [])
    req = state.get("parsed_requirements", {})
    raw_jd = state.get("raw_jd", "")
    comparison = state.get("comparison_report", {})

    overall_recommendation = comparison.get("recommendation", "See candidate details below.")

    # Build Markdown report
    report_lines = [
        "# Candidate Matching Report",
        "",
        "## Job Description Summary",
        f"> {req.get('role_summary', raw_jd[:200])}",
        "",
        f"**Must-have skills:** {', '.join(req.get('must_have', []))}",
        f"**Nice-to-have:** {', '.join(req.get('nice_to_have', []))}",
        f"**Experience required:** {req.get('experience_years', 'N/A')}+ years",
        "",
        "---",
        "",
        "## Round 1 Screening Summary",
        f"Screened {len(state.get('round1_candidates', []))} candidates → top 10 advanced.",
        "",
        "## Round 2 Deep Analysis",
        f"Analysed top 10 candidates → ranked shortlist produced.",
        "",
        "## Round 3 – Hire / No-Hire Recommendations",
        "",
        f"**Overall Recommendation:** {overall_recommendation}",
        "",
        "---",
        "",
        "## Candidate Details",
        "",
    ]

    for candidate in shortlist:
        cid = candidate["candidate_id"]
        rank = candidate["rank"]
        score = candidate["match_score"]
        strengths = candidate.get("strengths", [])
        gaps = candidate.get("gaps", [])
        fit = candidate.get("overall_fit", "")

        hire_rec = "✅ **HIRE**" if rank <= 3 else ("🔄 **MAYBE**" if rank <= 6 else "❌ **NO HIRE**")

        report_lines += [
            f"### {rank}. `{cid}`  —  Match Score: {score}%  {hire_rec}",
            "",
            f"**Overall Fit:** {fit}" if fit else "",
            "",
            "**Strengths:**",
            *[f"  - {s}" for s in strengths],
            "",
            "**Gaps:**",
            *[f"  - {g}" for g in gaps],
            "",
            "**Improvement Suggestions (for borderline candidates):**",
        ]

        if rank in (4, 5, 6):
            report_lines.append(
                f"  - Candidate `{cid}` is borderline. "
                "Address the identified gaps with additional training or targeted interviews."
            )
        report_lines += ["", "---", ""]

    report_md = "\n".join(report_lines)

    # Write to disk using Milestone 1 file-system tool
    report_path = str(_THIS_DIR / "reports" / "candidate_report.md")
    from milestones.milestone1 import write_file as _wf
    _wf(report_path, report_md)

    ai_msg = AIMessage(
        content=(
            f"📝 **Final Report Generated (Round 3)**\n\n"
            f"Report written to `{report_path}`\n\n"
            f"**Recommendation:** {overall_recommendation}\n\n"
            "**Top candidates:**\n"
            + "\n".join(
                f"  {c['rank']}. `{c['candidate_id']}` — {c['match_score']}%"
                for c in shortlist[:5]
            )
            + "\n\nEntering human feedback loop — do you want to refine the results?"
        )
    )
    return {
        "final_report": report_md,
        "screening_round": 3,
        "current_node": "generate_report",
        "messages": [ai_msg],
    }


def node_human_feedback(state: AgentState) -> dict:
    """
    Node 6 – Human Feedback Loop
    In non-interactive mode the agent inspects the latest human message.
    The CLI (main.py) injects the user's reply as a new HumanMessage before
    resuming the graph.
    """
    last_human = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human = msg.content.strip()
            break

    # Detect refinement intent keywords
    refinement_keywords = [
        "refine", "adjust", "change", "update", "modify", "different",
        "re-rank", "rerank", "re rank", "add", "remove", "filter",
        "why", "explain", "compare", "questions", "more", "focus",
    ]
    needs_refinement = any(kw in last_human.lower() for kw in refinement_keywords)

    # If the user asked for a comparison or interview questions, handle inline
    if "compare" in last_human.lower() or "questions" in last_human.lower():
        # Let the conversational agent node handle it
        needs_refinement = True

    if not last_human or last_human.lower() in {"no", "done", "exit", "quit", "finish", "end"}:
        ai_msg = AIMessage(content="✅ Workflow complete. Goodbye!")
        return {
            "needs_refinement": False,
            "human_feedback": last_human,
            "current_node": "human_feedback",
            "messages": [ai_msg],
        }

    ai_msg = AIMessage(
        content=(
            f"💬 Received your feedback: *{last_human}*\n\n"
            + (
                "Refining results based on your input …"
                if needs_refinement
                else "No further refinements requested. Finishing workflow."
            )
        )
    )
    return {
        "needs_refinement": needs_refinement,
        "human_feedback": last_human,
        "current_node": "human_feedback",
        "messages": [ai_msg],
    }


def node_conversational_agent(state: AgentState) -> dict:
    """
    Conversational node that handles natural-language queries, refinements,
    and iterative requests using the full tool suite.  Invoked when
    needs_refinement is True.
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    # Inject context about current shortlist into the system prompt
    shortlist = state.get("shortlist", [])
    req = state.get("parsed_requirements", {})

    context_injection = ""
    if shortlist:
        context_injection += "\n\nCurrent shortlist:\n" + "\n".join(
            f"  {c['rank']}. {c['candidate_id']} ({c['match_score']}%)"
            for c in shortlist[:10]
        )
    if req:
        context_injection += f"\n\nParsed JD requirements: {json.dumps(req, indent=2)}"

    if context_injection:
        messages[0] = SystemMessage(content=SYSTEM_PROMPT + context_injection)

    response = _llm_with_tools.invoke(messages)
    tool_messages_list = [response]

    # Process any tool calls
    while hasattr(response, "tool_calls") and response.tool_calls:
        tool_node = ToolNode(ALL_TOOLS)
        tool_output = tool_node.invoke({"messages": [response]})
        tool_result_msgs = tool_output["messages"]
        tool_messages_list.extend(tool_result_msgs)

        # Continue if LLM wants another tool call
        follow_up_messages = messages + tool_messages_list
        response = _llm_with_tools.invoke(follow_up_messages)
        tool_messages_list.append(response)

        if not (hasattr(response, "tool_calls") and response.tool_calls):
            break

    return {
        "needs_refinement": False,
        "current_node": "conversational_agent",
        "messages": tool_messages_list,
    }


# ===========================================================================
# Routing / edges
# ===========================================================================

def route_after_feedback(state: AgentState) -> str:
    """After human feedback, either loop to conversational agent or end."""
    if state.get("needs_refinement", False):
        return "conversational_agent"
    return END


# ===========================================================================
# Graph construction
# ===========================================================================

def build_graph() -> Any:
    """Build and compile the LangGraph StateGraph."""
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("parse_jd", node_parse_jd)
    builder.add_node("extract_requirements", node_extract_requirements)
    builder.add_node("search_resumes", node_search_resumes)
    builder.add_node("rank_candidates", node_rank_candidates)
    builder.add_node("generate_report", node_generate_report)
    builder.add_node("human_feedback", node_human_feedback)
    builder.add_node("conversational_agent", node_conversational_agent)

    # Fixed edges (main pipeline)
    builder.add_edge(START, "parse_jd")
    builder.add_edge("parse_jd", "extract_requirements")
    builder.add_edge("extract_requirements", "search_resumes")
    builder.add_edge("search_resumes", "rank_candidates")
    builder.add_edge("rank_candidates", "generate_report")
    builder.add_edge("generate_report", "human_feedback")

    # Conditional: feedback → refinement loop or END
    builder.add_conditional_edges(
        "human_feedback",
        route_after_feedback,
        {"conversational_agent": "conversational_agent", END: END},
    )

    # After conversational agent, go back to human feedback (re-prompt user)
    builder.add_edge("conversational_agent", "human_feedback")

    return builder.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_agent(
    job_description: str,
    *,
    initial_feedback: str = "",
    verbose: bool = True,
) -> AgentState:
    """
    Run the full matching agent pipeline for a given job description.

    Args:
        job_description:  The raw JD text.
        initial_feedback: Optional first feedback (empty = stop after report).
        verbose:          Print node summaries to stdout.

    Returns:
        Final AgentState.
    """
    # Ensure reports directory exists
    (_THIS_DIR / "reports").mkdir(exist_ok=True)

    graph = build_graph()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=job_description)],
        "raw_jd": job_description,
        "parsed_requirements": {},
        "round1_candidates": [],
        "round2_candidates": [],
        "shortlist": [],
        "comparison_report": {},
        "screening_round": 0,
        "final_report": "",
        "human_feedback": initial_feedback,
        "needs_refinement": False,
        "current_node": "",
    }

    config = {"recursion_limit": 50}

    final_state = graph.invoke(initial_state, config=config)

    if verbose:
        _print_state_summary(final_state)

    return final_state


def _print_state_summary(state: AgentState) -> None:
    """Print a brief summary of the final state."""
    print("\n" + "=" * 60)
    print("AGENT PIPELINE COMPLETE")
    print("=" * 60)
    shortlist = state.get("shortlist", [])
    if shortlist:
        print(f"\nTop {min(5, len(shortlist))} Candidates:")
        for c in shortlist[:5]:
            print(f"  {c['rank']}. {c['candidate_id']} — {c['match_score']}%")
    if state.get("final_report"):
        print(f"\n📄 Report saved to: {_THIS_DIR}/reports/candidate_report.md")
    print()
