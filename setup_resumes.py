"""
setup_resumes.py
================
One-time setup: clones the Milestone 2 resume dataset into this repo's
resumes/ directory so the agent can find it without needing a sibling clone.

Usage:
    python setup_resumes.py

What it does:
    1. Checks whether resumes/ already exists and has PDF files.
    2. If not, shallow-clones https://github.com/Ms-Shahid/RAG_Based_Job_Profile_matching
       into a temp directory and copies the resumes/ folder here.
    3. Prints instructions for next steps.

This only needs to be run ONCE.  After that, resumes/ is present locally
and the agent finds it automatically.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_URL = "https://github.com/Ms-Shahid/RAG_Based_Job_Profile_matching.git"
THIS_DIR = Path(__file__).parent.resolve()
DEST_DIR = THIS_DIR / "resumes"


def already_have_resumes() -> bool:
    return DEST_DIR.exists() and len(list(DEST_DIR.glob("resume_*.pdf"))) > 0


def clone_and_copy() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "m2"
        print(f"⬇️   Shallow-cloning {REPO_URL} …")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(tmp_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("❌  git clone failed:\n", result.stderr)
            sys.exit(1)

        src_resumes = tmp_path / "resumes"
        if not src_resumes.exists():
            print(f"❌  resumes/ not found in cloned repo at {src_resumes}")
            sys.exit(1)

        print(f"📂  Copying resumes → {DEST_DIR} …")
        shutil.copytree(str(src_resumes), str(DEST_DIR))

    count = len(list(DEST_DIR.glob("resume_*.pdf")))
    print(f"✅  {count} resume PDFs copied to {DEST_DIR}")


def main() -> None:
    print("=" * 60)
    print("  Agent Profile Matching — Resume Data Setup")
    print("=" * 60)

    if already_have_resumes():
        count = len(list(DEST_DIR.glob("resume_*.pdf")))
        print(f"✅  resumes/ already contains {count} PDFs. Nothing to do.")
        return

    # Prefer a sibling clone if it exists (avoids re-downloading)
    for sibling_name in ("RAG_Profile_matching", "RAG_Based_Job_Profile_matching"):
        sibling = THIS_DIR.parent / sibling_name / "resumes"
        if sibling.exists() and any(sibling.glob("resume_*.pdf")):
            print(f"📁  Found sibling clone at {sibling}")
            count = len(list(sibling.glob("resume_*.pdf")))
            print(f"    Copying {count} PDFs to {DEST_DIR} …")
            shutil.copytree(str(sibling), str(DEST_DIR))
            print(f"✅  Done.  {count} PDFs copied.")
            _print_next_steps()
            return

    # No sibling → clone from GitHub
    clone_and_copy()
    _print_next_steps()


def _print_next_steps() -> None:
    print()
    print("Next steps:")
    print("  1. Copy .env.example → .env  and add your OPENROUTER_API_KEY")
    print("  2. pip install -r requirements.txt")
    print("  3. python main.py --demo          # CLI mode")
    print("  4. python app.py                  # Gradio UI")
    print()


if __name__ == "__main__":
    main()
