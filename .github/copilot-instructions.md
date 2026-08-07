# 🤖 AI Agent Instructions — RAG Engineer (Person 2)

> **Role**: RAG Engineer — Retrieval System + Evaluation
> **Repository**: `Omarmahmoud984/hospital-A-memory-rag`
> **Context**: This repo already contains a working MCP server for the Meridian Hospital Network
> (`agent/`, `mcp_server/`, `db/`) built by a previous teammate. A second teammate (Person 1) is
> adding a **Memory system** in `memory/` and `context_eval/`. Your job is to add the **RAG
> (Retrieval-Augmented Generation) system** in `rag/` and `retrieval_eval/`, and integrate it with
> the existing `agent/` and `mcp_server/` code — **without breaking anything that already works**.

---

## 🎯 Your Goals & Deliverables

Build a document ingestion + retrieval pipeline over the hospital's policy/protocol documents
(e.g. triage guidelines, operating-room rules — currently exposed as static MCP resources at
`triage://protocols/guidelines` and `hospital://operating-rooms/rules`), implement **three RAG
architectures** (Naive, Hybrid, Agentic) with **Self-RAG verification**, and produce an
**evaluation framework** comparing them on accuracy, token usage, and latency.

Deliverables:
- Vector DB + retrieval pipeline working end-to-end
- 3 RAG architectures implemented and runnable independently
- Self-RAG verification layer (checks retrieved chunks actually support the generated answer)
- A domain-specific test question set for this hospital project (not generic RAG demo questions)
- An evaluation script producing a comparison table (accuracy, tokens, latency) across the 3 architectures
- Integration point in `agent/agent.py` so the agent can call RAG when memory/tools alone aren't enough
- All work committed on a feature branch with a PR opened against `main`

---

## ⚠️ Ground Rules for the Agent (Copilot)

1. **Read before writing.** Before creating any file, open and read `agent/agent.py`,
   `agent/mcp_protocol.py`, `mcp_server/MCP.py`, `mcp_server/schemas.py`, `mcp_server/db_helpers.py`,
   `db/schema.sql`, and `README.md`. Match existing code style, naming conventions, and how MCP
   resources/tools are currently defined. Do not invent a parallel framework.
2. **Do not modify** `db/schema.sql`, `db/seed.sql`, or anything under `memory/` /
   `context_eval/` — those are owned by the other teammate. If the RAG layer needs a new table
   (e.g. for storing document chunks/embeddings), propose it as a new file
   `db/rag_extensions.sql` instead of editing the existing schema.
3. **Do not break existing tests.** Run `python agent/test_e2e.py` after your changes and make
   sure the existing `PASS` output is unchanged.
4. **No hardcoded secrets.** Any API key (e.g. `ANTHROPIC_API_KEY`) must be read from environment
   variables, matching how `agent/agent.py` already does it (offline deterministic fallback if no
   key is present).
5. **Explain, don't just generate.** For each file, add a short module-level docstring explaining
   what it does and why, so a human reviewer (me) can understand and defend every design decision.

---

## 🛠️ Step-by-Step Instructions

### Step 1: Git Branch Setup
```bash
git checkout main
git pull origin main
git checkout -b feat/rag-engineer
git config user.name "<your name>"
git config user.email "<your GitHub email>"
```

### Step 2: Implement Files

#### 1. `rag/chunking.py`
- Load source documents (start with the two existing policy resources: triage guidelines and
  operating-room rules; structure this so more hospital documents can be added later, e.g. from a
  `rag/documents/` folder).
- Implement chunking with configurable chunk size and overlap. Preserve section/heading metadata
  per chunk (e.g. `source`, `section_title`, `chunk_index`) so retrieved chunks can be cited.

#### 2. `rag/embedding.py`
- Generate embeddings for each chunk (use a single configurable embedding function so the
  embedding model can be swapped later; document which model/provider is used and why).
- Batch embedding calls; handle rate limits/errors gracefully.

#### 3. `rag/vectordb.py`
- Build a vector index (HNSW) over the chunk embeddings.
- Maintain a metadata store (chunk text, source document, section) alongside the index so results
  can be mapped back to readable citations.
- Expose a simple `search(query, k)` interface returning ranked chunks with scores.

#### 4. `rag/naive_rag.py`
- Baseline: embed the query, retrieve top-k chunks via vector similarity, stuff them into the
  prompt, generate an answer. No re-ranking, no verification.

#### 5. `rag/hybrid_rag.py`
- Combine vector search with keyword search (BM25) over the same chunk set, merge/re-rank
  results (e.g. reciprocal rank fusion), then generate.

#### 6. `rag/agentic_rag.py`
- Multi-step retrieval loop: the model can issue follow-up retrieval queries if the first
  retrieval is insufficient (e.g. query decomposition or iterative refinement), up to a bounded
  number of steps, before producing a final answer.

#### 7. `rag/self_rag.py`
- Verification layer usable by any of the three architectures above: given a generated answer and
  the chunks it was based on, check whether the answer is actually supported by those chunks
  (e.g. an entailment/verification prompt or heuristic overlap check). Flag unsupported claims
  instead of returning them silently.

#### 8. `retrieval_eval/questions.py`
- A set of domain-specific test questions grounded in the actual hospital documents/resources in
  this repo (triage guidelines, OR rules) — include easy factual questions, multi-hop questions,
  and at least a few "unanswerable from the documents" questions to test that the system doesn't
  hallucinate.

#### 9. `retrieval_eval/evaluate.py`
- Run all test questions through Naive, Hybrid, and Agentic (with Self-RAG) pipelines.
- Measure: answer accuracy (against expected answers you define), token usage, and latency per
  architecture.
- Output a comparison table (e.g. Markdown or CSV) summarizing results, and print a short
  recommendation of which architecture performed best and why.

#### 10. Integration
- In `agent/agent.py` (or a small new adapter), add a code path where the agent can invoke the RAG
  pipeline (default to the best-performing architecture from your evaluation) when a user question
  requires policy/protocol knowledge that isn't in the structured database or memory.
- Follow the existing MCP resource/tool pattern already used in `mcp_server/MCP.py` rather than
  bypassing it.

### Step 3: Commit (Staged Commits)
```bash
git add rag/
git commit -m "feat(rag): implement chunking, embedding, vector DB, and 3 RAG architectures with self-RAG verification"

git add retrieval_eval/
git commit -m "test(rag): add domain-specific evaluation question set and comparison framework"

git add agent/
git commit -m "feat(agent): integrate RAG pipeline into agent for policy/protocol queries"

git push origin feat/rag-engineer
```

### Step 4: Open Pull Request
Open a PR from `feat/rag-engineer` to `main` titled:
`feat(rag): retrieval system, 3 architectures, self-RAG verification, and evaluation`

In the PR description, include the comparison table produced by `retrieval_eval/evaluate.py` and
a short explanation of trade-offs between the three architectures.

---

## ✅ After Copilot Finishes — Your Review Checklist

Before this counts as done, go through the generated code yourself and confirm you can answer:
- What does each of the 3 architectures actually do differently, in your own words?
- Why did the evaluation results come out the way they did?
- What would break if the underlying documents changed?
- Could you explain `self_rag.py`'s verification logic to someone else without looking at it?

If you can't answer these, that's a sign to go back and read/adjust the code with the agent rather
than submitting it as-is — since you'll likely be asked about it directly.
