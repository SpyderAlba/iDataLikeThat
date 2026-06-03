# AgentOps with MLflow — Workshop Project

## Project Overview

A 60-minute workshop for Women Who Code 2026 covering MLflow-based observability for agent systems. Two notebook tracks (Databricks managed MLflow and OSS MLflow) walk through the six pillars of AgentOps.

**Speaker:** Stephanie Alba · [@SpyderAlba](https://github.com/SpyderAlba)

## Repository Structure

```
WomenWhoCode2026/
├── AgentOps with Databricks Managed MLflow.ipynb   # Track A notebook
├── AgentOps with OSS MLflow.ipynb                  # Track B notebook
├── README.md                                        # Workshop README
└── Screenshot 2026-06-02 at 4.33.49 PM.png         # Next Steps image
```

## Six Pillars of AgentOps

1. **Tracing & Observability** — MLflow Tracing captures every step: inputs, outputs, tool calls, latency, token usage
2. **Evaluation** — Automated quality scoring with built-in and custom scorers
3. **Human Feedback** — Grounding automated scores with real user signals
4. **Production Monitoring** — Live dashboards, trace sampling, drift detection, and alerting
5. **Governance & Lifecycle** — Model versioning, eval gates in CI/CD, audit trails, and access control
6. **Prompt Optimization** — Data-driven prompt improvement

## Key Differences Between Notebooks

| Feature | Databricks (Track A) | OSS (Track B) |
|---------|----------------------|----------------|
| Evaluation scorers | Built-in LLM judges (`Correctness()`, `Safety()`, `Guidelines()`) | Code-based `@scorer` functions |
| Multi-turn simulation | `ConversationSimulator` | Markdown reference only |
| Prompt optimization | `GepaPromptOptimizer` + `mlflow.genai.optimize_prompts()` | Manual prompt iteration with version comparison |
| Trace storage | UC-backed Delta tables | Standard MLflow tracking server |
| LLM endpoint | `databricks-claude-sonnet-4` | `gpt-4o-mini` (for external use) |
| MLflow version | `>=3.13.0` | `>=3.10` |

## Prerequisites

**Databricks Track**
- Databricks workspace with a running cluster
- Unity Catalog enabled
- Access to Foundation Model APIs

**OSS Track (self-hosted)**
- Python 3.9+
- `mlflow>=3.10`, `openai`
- A self-hosted MLflow tracking server
- OpenAI API key or another LLM provider

## Conventions

- **"AgentOps" is one word** — not "Agent Ops" or "Agent-Ops"
- Both notebooks use dynamic experiment names derived from the notebook path
- The Databricks notebook stores traces in Unity Catalog Delta tables
- Built-in LLM-judge scorers are currently Databricks-only; OSS support is coming soon
