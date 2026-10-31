# Assignment Requirements

## Part A: Agent Architecture — 40%

### 1. Create `matching_agent.py`

Build the agent using **LangGraph**.

### 2. Agent State Design

The agent state should track:

- Conversation history
- Job requirements understanding
- Candidate shortlist and reasoning

### 3. Agent Workflow

Implement the following graph structure:

```text
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
```

### 4. Tools Available to Agent

#### Existing Tools

- All file system tools from **Milestone 1** 
  - Milestone-1 project Path : /Users/mohammedshahid/Developer/code/Airtribe/projects/LLMPoweredFileSystem
- RAG search tool from **Milestone 2**
  - Milestone-2 project path : /Users/mohammedshahid/Developer/code/Airtribe/projects/RAG_Profile_matching

#### Additional Tools

```python
extract_requirements(jd: str)
```

Parse the job description and identify:

- Must-have requirements
- Nice-to-have requirements

```python
compare_candidates(candidate_ids: list)
```

Perform a head-to-head comparison between candidates.

```python
generate_interview_questions(candidate_id: str)
```

Generate screening/interview questions for a specific candidate.

---

# Part B: Interactive Features — 30%

## 1. Conversational Interface

The agent should accept natural-language queries such as:

> "Find me candidates with React and 3+ years experience"

> "Compare the top 3 matches side by side"

> "Why did John rank higher than Jane?"

## 2. Iterative Refinement

The agent should allow users to:

- Adjust requirements during the conversation
- Re-rank candidates based on new criteria
- Explain why the rankings changed

---

# Part C: Advanced Capabilities — 30%

## 1. Multi-Round Screening

Implement a multi-stage candidate screening process.

### Round 1: Initial Screening

- Screen **100 resumes**
- Select the **top 10 candidates**

### Round 2: Deep Analysis

- Perform detailed analysis of the top 10 candidates

### Round 3: Final Recommendation

Generate a final:

- **Hire** recommendation
- **No-Hire** recommendation

## 2. Explainability

The agent should generate detailed match reports that:

- Highlight strengths for each candidate
- Highlight gaps for each candidate
- Explain the candidate's ranking
- Provide improvement suggestions for borderline candidates