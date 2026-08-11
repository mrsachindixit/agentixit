# 🤖 AgenticAI — Hands-On Code Companion

<p align="left">
	<img src="https://img.shields.io/badge/Local--First-Ollama-10b981?style=for-the-badge" alt="Local First" />
	<img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
	<img src="https://img.shields.io/badge/Frameworks-LangChain%20%7C%20LangGraph%20%7C%20DSPy%20%7C%20LlamaIndex-8b5cf6?style=for-the-badge" alt="Frameworks" />
</p>

Build real AI agents from scratch: tool-calling, RAG, memory, multi-agent systems, and production hardening — all running locally with Ollama.

This is the code half of the repo. The text lives in the root **[README.md](../README.md)** — read a section, then run the matching module.

| Book chapter | Code |
|---|---|
| [Basics of LLMs, Prompts and Tool Calls](../README.md#basics-of-llms-prompts-and-tool-calls) | `module01_raw/`, `module02_basics/` |
| [Foundation of Agentic AI](../README.md#foundation-of-agentic-ai) | `module01_raw/1.10_rag_basic/`, `module03_langchain/` |
| [Agentic Execution and Patterns](../README.md#agentic-execution-and-patterns) | `module03_langchain/`, `module08_frameworks/` |
| [Production Aspects](../README.md#production-aspects) | `module04_production/`, `module05_security/`, `module07_evaluation/` |
| [Enterprise Suites and Protocols](../README.md#enterprise-suites-and-protocols) | `module06_enterprise/` |

---

## 🎯 Quick Navigation
- [Setup](#setup)
- [🗺️ Learning Path](#-learning-path)
- [⚙️ Configuration](#️-configuration)
- [🏁 Capstone Assignments](#-capstone-assignments)
- [🎮 Playground](#-playground)
- [🧪 Testing and Sanity Checks](#-testing-and-sanity-checks)
- [🛠️ Troubleshooting](#️-troubleshooting)

---

## Setup

Environment setup (Python, virtualenv, dependencies, Ollama models) lives in one place:
**[../docs/setup.md](../docs/setup.md)**.

All commands below assume you are in this `code/` directory with the virtualenv active.

```bash
cd code
python capstones/capstone1_sql_agent/cap1_app.py "List engineering employees with salary > 2000000"
```
---

## 🗺️ Learning Path

Follow the modules in order — each one builds on the previous. The arc moves from a single raw LLM call all the way to enterprise-grade protocols and alternative frameworks.

---

### Module 01 · Raw LLM Interactions
*Start here. No frameworks — just Python + Ollama. You learn what an LLM actually does at the API level.*

| File | What it teaches |
|---|---|
| `1.1_hello_llm.py` | Your first LLM call — send a prompt, read a response |
| `1.2_chat_io.py` | Interactive chat loop with streaming and turn history |
| `1.3_tool_single.py` | Give the LLM one tool and let it decide when to use it |
| `1.4_memory_sim.py` | Simulate short-term memory by managing the message list yourself |
| `1.5_code_gen.py` | Prompt the LLM to write Python, then execute the output safely |
| `1.6_regex_bot.py` | Deterministic response routing via regex — no LLM for intent |
| `1.7_nlp_bot.py` | Lightweight NLP intent detection as a fallback before the LLM |
| `1.8_tools_multi.py` | Multiple tools in one agent loop — LLM picks which to call |
| `1.9_db_agent_sqlite.py` | Full agent that translates natural language to SQL and executes it |
| `1.10_rag_basic/` | Retrieval-Augmented Generation from scratch: embed, store, retrieve, answer |
| `1.11_ota_loop_from_scratch.py` | The bare Observe-Think-Act loop every agent reduces to — no framework |
| `1.12_ralph_loop.py` | The Ralph Wiggum loop: dumbest agent that still works, terminated by real tests |
| `1.13_rag_hybrid_bm25.py` | Hybrid retrieval: BM25 + dense + Reciprocal Rank Fusion + metadata filter |
| `1.14_tool_failure_recovery.py` | Five tool failure modes and their recovery paths, including idempotent replay |

---

### Module 02 · LangChain Primitives
*Introduce LangChain's chat model objects. Same concepts as Module 01 but now using the framework's building blocks — notice how the pattern maps 1:1.*

| File | What it teaches |
|---|---|
| `2.1_chat_obj_llm.py` | `ChatOllama` object replaces raw `requests` — same call, cleaner interface |
| `2.2_image_analysis.py` | Multi-modal input: pass an image to a vision-capable LLM |
| `2.3_chat_obj_tool_single.py` | `model.bind_tools([...])` — single tool, same loop as `1.3` but framework-native |
| `2.4_chat_obj_tool_multi.py` | Multiple tools bound to the model; LangChain handles the dispatch loop |
| `2.7_cot_fewshot_self_consistency.py` | Zero-shot CoT, few-shot CoT, and self-consistency voting on the same problem |
| `2.8_structured_output_response.py` | Enforce a Pydantic schema on the LLM **response** — three approaches, including retry-on-format-error |
| `2.5_reasoning_models.py` | Standard model vs reasoning model on the same puzzle, side by side |
| `2.6_responses_api.py` | Stateless Chat Completions vs the stateful Responses API |

---

### Module 03 · LangChain Agents & LangGraph
*The framework takes over orchestration. You declare agents, graphs, and memory — the framework runs the loop.*

| File | What it teaches |
|---|---|
| `3.1_tool_call.py` | LangChain `@tool` decorator and `.invoke()` — the declarative equivalent of `1.3` |
| `3.2_agent_simple.py` | `create_react_agent` — hand the LLM a toolset and let the ReAct loop run |
| `3.3_memory_checkpoint_langchain.py` | Persist conversation state with `MemorySaver` checkpoints |
| `3.4_multi_agent.py` | Two agents collaborating: one plans, one executes |
| `3.5_lg_basic.py` | First LangGraph: nodes, edges, and `StateGraph` — graphs replace imperative loops |
| `3.6_agent_middleware_langchain.py` | Intercept agent steps with middleware: logging, auth, rate-limit |
| `3.7_multi_tool_call.py` | LLM makes multiple tool calls in a single turn |
| `3.8_agent_multi_tool.py` | Agent with a full tool registry; LangChain resolves calls automatically |
| `3.9_agent_tools_seq_basic.py` | Force sequential tool execution — output of one feeds into the next |
| `3.10_memory_checkpoint_langchain_1.py` | Persistent memory across sessions using thread IDs |
| `3.11_multi_agent.py` | Supervisor pattern: one agent routes tasks to specialist sub-agents |
| `3.12_rag_langchain.py` | Full RAG pipeline using LangChain retriever + chat chain |
| `3.13_agent_tools_with_tool_seq.py` | Tools that themselves trigger further tool calls (tool chaining) |
| `3.14_multi_tool_orchestration.py` | Parallel and conditional tool fan-out in a single agent run |
| `3.15_langgraph_coordinator.py` | LangGraph coordinator node that routes to worker subgraphs |
| `3.20_advanced_rag_query_rewrite_rerank.py` | Advanced RAG: query rewrite, rerank (heuristic), and grounded answer synthesis |
| `3.21_rag_cross_encoder_rerank.py` | Two-stage retrieval: bi-encoder for recall + real cross-encoder for precision |
| `3.22_rag_hyde_multiquery.py` | Query rewriting techniques: HyDE, multi-query expansion, decomposition, step-back |
| `3.23_mixture_of_agents.py` | Mixture-of-Agents: parallel proposers + aggregator synthesises a robust answer |
| `3.24_multi_agent_deadlock_safety.py` | Hop limits, cycle detection, watchdog — prevent multi-agent runaway |
| `3.25_critic_refiner_loop.py` | Critic-refiner **loop** with rubric scores and an iteration budget (vs. one-shot review in `3.4`) |
| `3.16_lg_pause_resume_hitl.py` | LangGraph `interrupt`: pause mid-graph for human approval, then resume |
| `3.17_lg_functional_api.py` | The same graph written twice — StateGraph vs the functional `@task` API |
| `3.18_deep_agents.py` | Deep agent with a durable plan file it can re-read across steps |
| `3.19_agentic_rag.py` | RAG where the agent decides whether and what to retrieve |
| `3.26_handoff_envelope_validation.py` | Validating inter-agent handoff envelopes before acting on them |

---

### Module 04 · Production — Performance & Reliability
*Your agent works in a notebook — now make it fast, cheap, observable, and resilient. Each file adds one production dimension.*

| File | Category | What it teaches |
|---|---|---|
| `4.1_prompt_structure_and_params.py` | Prompting | System/user prompt structuring and LLM parameter tuning for performance |
| `4.2_prompt_perf_basics.py` | Prompting | Prompt focus, role design, and memory cleanup effects on latency/tokens |
| `4.3_output_format_perf_impact.py` | Prompting | Output formatting impact on latency, size, and parseability |
| `4.4_prompt_caching_anthropic.py` | Prompting | Prompt caching with cache-on/cache-off comparison and usage metrics |
| `4.5_context_compaction.py` | Context & Cost | Summarize old history and keep recent turns to shrink context window |
| `4.6_token_budget_controls.py` | Context & Cost | Input budget policy with deterministic truncation and output reserve |
| `4.7_request_dedup_coalescing.py` | Context & Cost | Coalesce identical in-flight requests to avoid duplicate model work |
| `4.8_cost_attribution_tenant.py` | Context & Cost | Per-tenant / per-feature cost attribution and chargeback aggregation |
| `4.9_parallel_tools_timeout.py` | Latency & Resilience | Parallel tool fan-out with timeout and partial-result handling |
| `4.10_timeout_fallback_ladder.py` | Latency & Resilience | Timeout + fallback + degraded response strategy |
| `4.11_multi_llm_performance_routing.py` | Latency & Resilience | Complexity-based routing: fast vs deep model selection and latency comparison |
| `4.12_streaming_events_langgraph.py` | Latency & Resilience | LangGraph state-event streaming, token streaming, and cancellation |
| `4.13_monitoring_basic.py` | Observability | Structured logging of every LLM call: latency, tokens, outcome |
| `4.14_monitoring_opentelemetry.py` | Observability | OpenTelemetry traces and spans — agent calls become distributed traces |

---

### Module 05 · Production — Security & Safety
*The other half of going to production: defend against attacks, lock down tools, protect data, and check your readiness.*

| File | Category | What it teaches |
|---|---|---|
| `5.1_security_attacks_diy.py` | Attacks | DIY attack lab: fire injection / exfiltration / SQL+tool-abuse at a live agent and watch the guard |
| `5.2_security_basic.py` | Defences | Input validation and prompt injection defence without a library |
| `5.3_multi_llm_security_crosscheck.py` | Defences | Two-model safety pattern: generator model + guard/reviewer model |
| `5.4_tool_auth_basic.py` | Authorization | Basic tool authorization via scopes (allow/deny behavior) |
| `5.5_mcp_tool_auth_basic.py` | Authorization | MCP-style tool authorization pattern with scope checks |
| `5.6_pii_basic.py` | Data Protection | Detect and redact PII with regex before data reaches the LLM |
| `5.7_pii_presidio.py` | Data Protection | Production PII detection with Microsoft Presidio NER |
| `5.8_pii_langchain.py` | Data Protection | Presidio wired into a LangChain chain as a preprocessing step |
| `5.9_bias_guardrails.py` | Safety & Policy | Detect biased or harmful outputs and block or flag the response |
| `5.10_agentic_12_factors.py` | Reliability | The 12 factors of a production agent — theory + a quick self-check |

---

---

### Module 06 · Enterprise Integrations
*Deploy and integrate agents using enterprise protocols such as MCP and A2A.*

| File | What it teaches |
|---|---|
| `6.1_mcp_server.py` | Build an MCP server from scratch using FastAPI — understand the envelope protocol |
| `6.2_mcp_client.py` | HTTP client that calls the `6.1` server's tools and reads its resources |
| `6.3_mcp_sdk_server.py` | Same server rebuilt with the official `mcp` SDK: `@mcp.tool`, `@mcp.resource`, `@mcp.prompt`, and a real LLM call via `Context` |
| `6.4_mcp_sdk_client.py` | Async `ClientSession` that invokes tools, reads resources, and fetches prompts from `6.3` |
| `6.5_a2a_demo.py` | Agent-to-Agent in one file: two Ollama-backed agents passing messages to each other |
| `6.6_a2a_server.py` | A2A server — registers agents, routes messages, maintains inboxes |
| `6.7_a2a_client.py` | A2A client — registers, sends messages, polls inbox |
| `6.4_mcp_sdk_client_langchain_adapter.py` | Reaching an MCP server through the LangChain MCP adapter |

---

### Module 07 · Evaluation & Testing for LLM/Agent Systems
*Learn how to measure quality, safety, cost, and regressions in LLM and agent workflows.*

| File | What it teaches |
|---|---|
| `7.0_eval_metrics_basics.py` | Core eval metrics: exact match, groundedness proxy, latency/cost signals |
| `7.1_eval_golden_dataset.py` | Golden dataset with clean, dirty, bias, and edge cases — must-include/must-not-include scoring |
| `7.2_eval_prompt_regression.py` | Prompt regression checks and format drift detection |
| `7.3_eval_llm_judge_rubric.py` | Rubric-based judging pattern (relevance/correctness/clarity) |
| `7.4_eval_agent_trajectory.py` | Agent path evaluation: tool choice, step budget, success |
| `7.5_eval_safety_attacks.py` | Safety evals for attack blocking behavior |
| `7.6_eval_rag_groundedness.py` | Groundedness checks using retrieved-context overlap |
| `7.7_eval_cost_latency_tradeoff.py` | Two trade-offs side by side: accuracy-per-second (latency) and accuracy-per-cost |
| `7.8_eval_ci_gate.py` | Release gating with thresholds for quality/safety/latency |
| `7.9_eval_report_card.py` | Consolidated eval report and action-oriented summary |
| `7.10_eval_planning_quality.py` | Planning eval: step coverage, dependency validity, and order quality |
| `7.11_eval_replanning_recovery.py` | Replanning eval: failure recovery via retry/fallback/degraded modes |
| `7.12_eval_multi_agent_handoff.py` | Multi-agent eval: delegation correctness, loop control, and termination |
| `7.13_eval_multi_llm_jury.py` | Multi-LLM eval panel: dual-judge scoring with agreement/disagreement signal |
| `7.14_eval_advanced_rag.py` | Advanced RAG eval: baseline vs improved grounding outcomes |
| `7.15_eval_factfulness.py` | Factfulness eval: supported-claim ratio under context constraints |
| `7.16_eval_react_multi_agent.py` | ReAct discipline eval in a multi-agent run: reason-before-act, valid delegation, termination |

---

### Module 08 · Framework Integrations
*Compare framework-specific patterns across Python, Java, and JavaScript runtimes.*

| File | What it teaches |
|---|---|
| `8.1_llamaindex_rag.py` | LlamaIndex RAG with Ollama embeddings — compare with `1.10` and `3.12` |
| `8.2_dspy_optimized_agent.py` | DSPy: declare signatures and let `BootstrapFewShot` optimise prompts automatically |
| `8.3_embabel_goal_agent.java` | Embabel (Java/Spring): `@Action` + `@Goal` — declarative planning, framework picks execution order |
| `8.4_langchain_js_tool_call.mjs` | LangChain in JavaScript — same tool-call pattern as `3.1`, different runtime |



---

## ⚙️ Configuration

All configuration is via environment variables. Defaults work out of the box.

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Chat/completion model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |

Override in PowerShell:
```powershell
$env:OLLAMA_MODEL = "llama3.2"
$env:OLLAMA_EMBED_MODEL = "nomic-embed-text"
```

Override in Bash/zsh:
```bash
export OLLAMA_MODEL=llama3.2
export OLLAMA_EMBED_MODEL=nomic-embed-text
```

---

## 🏁 Capstone Assignments

Treat each capstone as a real-world project, not a toy demo.

The intended flow is:
1. Read the problem statement carefully.
2. Design and implement your own solution first.
3. Compare with the reference implementation in `capstones/` afterward.

---

### Assignment 1 — SQLite Analyst Agent

Build an end-to-end agent that lets non-technical business users ask plain-English questions against a **multi-table relational SQLite database** and receive safe, accurate, human-readable answers. Your agent should automatically discover the live schema (tables, columns, types, relationships, plus sample rows) so the LLM never hallucinates column names. It must generate SQL through a tightly prompted LLM call that returns structured JSON — not free text — and it should be capable of producing JOINs, aggregations, CTEs, and grouped queries, not just flat `SELECT *` statements. Before executing anything, the agent must enforce a read-only safety guard (allowlist + blocklist) and auto-append a `LIMIT` when missing. After execution, expose the query plan (`EXPLAIN QUERY PLAN`) for observability, and make a second LLM call to summarize the raw result into a concise business narrative. Seed your database with at least five interrelated tables (departments, employees with self-referencing managers, projects, a many-to-many junction, salary history) — a single flat table makes every query trivial and defeats the purpose. Deliver a CLI that accepts a natural-language question and prints the generated SQL, the intent explanation, the query plan, a result preview, and the business summary.

**Advanced requirements — what makes this capstone yours:**
- **Self-correcting SQL loop.** When a generated query throws a SQLite error (bad column, ambiguous join, syntax), do **not** fail — feed the exact error message plus the schema back to the model and let it repair the query, bounded to a small retry budget (e.g. 3). Print every repair attempt so the student can see the agent reasoning its way to a working query.
- **Risk scoring + human approval gate.** Score each query for blast radius (joins, CTEs, missing `LIMIT`, length) and **refuse to auto-execute** anything above a threshold until a human approves it. The refusal must explain *why* it scored high. (A reference scoring scheme lives in the implementation — design your own first.)
- **Ambiguity → clarifying question.** If the question maps plausibly to more than one column/table (e.g. "revenue" when there are three money columns), the agent must ask **one** targeted clarifying question instead of silently guessing.
- **Cost-aware plan reading.** Parse `EXPLAIN QUERY PLAN` and translate it into a plain-English warning when the query will do a full table scan vs. use an index — so the business user understands *why* a question is slow.

---

### Assignment 2 — Research Agent

Build a **planner/executor research assistant** that takes a high-level research prompt (e.g., *"Survey methods for low-resource named-entity recognition"*), decomposes it into a structured plan, executes each step using specialised tools, and produces an evidence-backed synthesis. The planner should be an LLM call that outputs a structured JSON plan with ordered steps (search, ingest, extract, synthesize) — include a deterministic fallback when the model returns invalid JSON. The executor should iterate over the plan, dispatch each step to the right tool, and **persist every intermediate result** to disk as timestamped JSON so the research trail is auditable. You will need at least three tools: a web/paper search tool (a placeholder with a production-shaped interface is fine), a PDF/text ingestion tool that walks a folder and extracts full text with structured metadata, and an LLM summarizer with a heuristic fallback when the model is unreachable. The crown jewel is a full **RAG pipeline** — chunk ingested documents, embed them into a Chroma vector store, load a similarity retriever, and answer follow-up questions grounded in document context with source citations. Design the embedding and chat functions as injectable parameters (not hard-coded) so you can write unit tests with deterministic fakes instead of depending on a live Ollama server. Deliver a CLI that prints the plan, executes every step, and writes all notes plus a final synthesis to `data/notes/`.

**Advanced requirements — what makes this capstone yours:**
- **Plan quality gate before execution.** Before running a single step, validate the plan: every step's dependency must reference an earlier step, there must be no cycles, and the step types must cover the goal (you can't synthesize before you ingest). Reject and **re-plan** if the gate fails — never execute an incoherent plan.
- **Dependency-aware execution with recovery.** Execute steps in dependency order, passing prior outputs forward. When a step fails, retry with a degraded strategy (e.g. summarizer falls back to a heuristic) rather than aborting the whole run — and record the degradation in the audit trail.
- **Citation-faithful synthesis (no unsupported claims).** Every sentence in the final synthesis must map to at least one retrieved chunk. Sentences with no supporting evidence must be dropped or explicitly flagged as `[unverified]`. The agent should report a *grounding ratio* (supported sentences / total).
- **Contradiction surfacing.** When two sources disagree on a fact, the synthesis must **say so** ("Source A reports X; Source B reports Y") instead of silently picking one or averaging them.

---

### Assignment 3 — Standalone RAG Agent

Build a **conversational RAG agent** that ingests a local collection of PDF documents, constructs a persistent vector index, and then enters an interactive chat loop where every answer is grounded in retrieved context and aware of prior conversation turns. The ingestion module should recursively walk a data directory, extract text page-by-page (handling empty pages gracefully), and carry source/title metadata through the entire pipeline. The index builder should split documents into configurable chunks, embed them with Ollama, store in Chroma, and persist to disk — expose CLI flags for chunk size, overlap, and directories so students can experiment with retrieval quality. Conversation memory is critical: maintain a rolling history so the agent can resolve follow-ups like *"Tell me more about that"* — without it every turn is independent and the agent feels broken. Each prompt should be composed of three clear blocks: system instructions (answer from context, cite sources), the concatenated retrieved chunks, and the conversation history. Use a `.env` file for all Ollama configuration and let CLI flags override env vars. Deliver three independently runnable entry points: (a) ingest PDFs, (b) build/update the index, (c) launch the interactive conversational agent.

**Advanced requirements — what makes this capstone yours:**
- **History-aware query rewriting before retrieval.** A follow-up like *"tell me more about that"* retrieves garbage if sent to the vector store as-is. Before retrieving, the agent must rewrite the follow-up into a **standalone query** using the conversation history ("more about *the 2023 refund policy change*"), then retrieve on the rewritten query. Print both the raw and rewritten query so the effect is visible.
- **Answerability / abstain gate with confidence.** Score how well the retrieved chunks actually cover the question. If coverage is below a threshold, the agent must **abstain** ("I don't have enough in these documents to answer that") rather than hallucinate a confident-sounding answer. Report a confidence value with each answer.
- **Inline citations that can be checked.** Every answer must cite the specific source + page it drew from, and a `--verify` mode should print the exact retrieved chunk behind each citation so a reader can confirm the answer wasn't invented.
- **Retrieval self-check + one re-try.** After retrieving, the agent grades whether the chunks are relevant; if they're weak, it rewrites the query once more and re-retrieves before answering — a single, bounded self-correction loop.

---

> **Note:** Reference implementations are in `capstones/` — consult them **after** you attempt each assignment independently.

---

## 🎮 Playground

Use the interactive Streamlit app to experiment without writing code:

```bash
streamlit run playground/app.py
```

---

## 🧪 Testing and Sanity Checks

### Run all tests
```bash
pytest -q
```

### Run module suite
```bash
pytest tests -q
```

`tests/` is the canonical suite for module files (`module01_raw` -> `module08_frameworks`) with one self-contained test per source file.

---

## 🛠️ Troubleshooting

| Symptom | Fix |
|---|---|
| `Connection refused` | Run `ollama serve` |
| `Model not found` | Run `ollama pull llama3.2` then `ollama list` |
| Import errors | Run `pip install --upgrade -r requirements.txt` |
| Missing advanced model | Pull additional models as needed: `llama3.1:latest`, `lfm2.5-thinking:latest`, `llava` |

---

## 📝 Notes

- All code runs fully locally — no OpenAI API key or cloud account required.
