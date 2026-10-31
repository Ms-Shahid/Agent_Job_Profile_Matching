"""
Interactive CLI Entry Point
===========================

Run the full matching agent pipeline with a conversational interface.

Usage:
    python main.py                    # interactive mode (prompts for JD)
    python main.py --jd-file <path>   # load JD from a text file
    python main.py --demo             # use a built-in demo JD

Once the pipeline finishes its first pass, you can iterate:
  • "Find me candidates with React and 3+ years experience"
  • "Compare the top 3 matches side by side"
  • "Why did resume_1_backend_senior rank higher than resume_3_devops_expert?"
  • "Generate interview questions for resume_1_backend_senior"
  • "no" / "done" / "exit" to finish
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.messages import AIMessage, HumanMessage

from matching_agent import build_graph, AgentState
from rag_engine import build_vector_store


# ---------------------------------------------------------------------------
# Demo JD
# ---------------------------------------------------------------------------
DEMO_JD = """
Senior Backend Engineer

We are looking for a Senior Backend Engineer with 5+ years of experience
building scalable microservices and distributed systems.

Required Skills:
- Python (5+ years)
- PostgreSQL or similar relational databases
- AWS (EC2, S3, RDS, Lambda)
- Docker and Kubernetes
- REST API design
- CI/CD pipelines

Nice to Have:
- Kafka / message queues
- GraphQL
- Terraform
- Team leadership or mentoring experience

Experience: 5+ years
Location: Remote (US timezones preferred)
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_ai_message(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            return msg.content
    return ""


def _print_banner() -> None:
    print("\n" + "=" * 65)
    print("  🤖  AI Profile Matching Agent  (LangGraph + RAG)")
    print("=" * 65)
    print("  Milestone 1: File System Tools  ✓")
    print("  Milestone 2: RAG Semantic Search ✓")
    print("  New Tools:   extract_requirements, compare_candidates,")
    print("               generate_interview_questions          ✓")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AI Profile Matching Agent")
    parser.add_argument("--jd-file", type=str, help="Path to a plain-text JD file")
    parser.add_argument("--demo", action="store_true", help="Run with built-in demo JD")
    args = parser.parse_args()

    _print_banner()

    # ── Step 1: Pre-build the vector store ───────────────────────────────
    print("🔄  Initialising RAG vector store (this may take ~60 s on first run) …\n")
    build_vector_store()

    # ── Step 2: Get job description ──────────────────────────────────────
    if args.demo:
        jd = DEMO_JD
        print("📋  Using demo job description:\n")
        print(jd[:400] + " …\n")
    elif args.jd_file:
        jd_path = Path(args.jd_file)
        if not jd_path.exists():
            print(f"❌  File not found: {jd_path}")
            sys.exit(1)
        jd = jd_path.read_text()
        print(f"📋  Loaded JD from: {jd_path}\n")
    else:
        print("📋  Paste your job description below.")
        print("    Type END on a new line when done.\n")
        lines = []
        while True:
            try:
                line = input()
                if line.strip().upper() == "END":
                    break
                lines.append(line)
            except EOFError:
                break
        jd = "\n".join(lines)
        if not jd.strip():
            print("No JD provided — using demo JD.")
            jd = DEMO_JD

    # ── Step 3: Build graph & run first pipeline pass ────────────────────
    graph = build_graph()
    _REPORTS_DIR = Path(__file__).parent.resolve() / "reports"
    _REPORTS_DIR.mkdir(exist_ok=True)

    initial_state: AgentState = {
        "messages": [HumanMessage(content=jd)],
        "raw_jd": jd,
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

    print("\n🚀  Running matching pipeline …\n")
    config = {"recursion_limit": 60}

    # Stream node outputs
    current_state = initial_state
    for event in graph.stream(current_state, config=config, stream_mode="values"):
        current_state = event
        last_msg = _last_ai_message(current_state)
        if last_msg:
            # Only print newly generated messages (simple deduplicate)
            print(last_msg)
            print()

    # ── Step 4: Interactive refinement loop ──────────────────────────────
    print("\n" + "─" * 65)
    print("💬  Human Feedback Mode")
    print("    Ask follow-up questions or type 'done' / 'exit' to finish.")
    print("    Examples:")
    print('      "Compare the top 3 matches side by side"')
    print('      "Why did candidate X rank higher than Y?"')
    print('      "Generate interview questions for resume_1_backend_senior"')
    print('      "Adjust requirements: need 7+ years experience in Node.js"')
    print("─" * 65 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋  Session terminated.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"done", "exit", "quit", "no", "finish"}:
            print("\n✅  Workflow complete. Goodbye!")
            break

        # Inject user message and re-invoke the human_feedback → conversational_agent path
        new_human_msg = HumanMessage(content=user_input)
        current_messages = list(current_state.get("messages", []))
        current_messages.append(new_human_msg)

        # Patch state and resume
        refinement_state: AgentState = {
            **current_state,
            "messages": current_messages,
            "human_feedback": user_input,
            "needs_refinement": True,
        }

        for event in graph.stream(
            refinement_state,
            config=config,
            stream_mode="values",
        ):
            current_state = event

        last_msg = _last_ai_message(current_state)
        if last_msg:
            print(f"\nAgent: {last_msg}\n")

    # ── Step 5: Print final summary ───────────────────────────────────────
    shortlist = current_state.get("shortlist", [])
    if shortlist:
        print("\n" + "=" * 65)
        print("📊  FINAL SHORTLIST")
        print("=" * 65)
        for c in shortlist[:10]:
            hire = "HIRE" if c["rank"] <= 3 else ("MAYBE" if c["rank"] <= 6 else "NO HIRE")
            print(f"  {c['rank']:2d}. {c['candidate_id']:<40s} {c['match_score']:5.1f}%  [{hire}]")

    report_path = Path(__file__).parent / "reports" / "candidate_report.md"
    if report_path.exists():
        print(f"\n📄  Full report: {report_path}")

    print()


if __name__ == "__main__":
    main()
