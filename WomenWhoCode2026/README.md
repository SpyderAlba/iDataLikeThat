# AgentOps with MLflow
### Practical MLflow Workflows for Monitoring, Evaluation & Continuous Improvement

**Speaker:** Stephanie Alba · [@SpyderAlba](https://github.com/SpyderAlba)
**Format:** 60-Minute Hands-On Workshop

---

## What You'll Build

By the end of this session you'll have a working observability stack for agent systems, covering all six pillars of Agent Ops:

1. **Tracing & Observability** — MLflow Tracing captures every step: inputs, outputs, tool calls, latency, token usage
2. **Evaluation** — Automated quality scoring using `mlflow.genai.evaluate()` with built-in and custom scorers
3. **Human Feedback** — Grounding automated scores with real user signals
4. **Production Monitoring** — Live dashboards, trace sampling, drift detection, and alerting
5. **Governance & Lifecycle** — Model versioning, eval gates in CI/CD, audit trails, and access control
6. **Prompt Optimization** — Data-driven prompt improvement via `mlflow.genai.optimize_prompts()`

---

## Notebooks

Two tracks — same six pillars, same concepts, different MLflow capabilities:

| Track | Notebook | When to Use |
|---|---|---|
| **Track A — Databricks** | `AgentOps with MLflow.py` | On Databricks; uses MLflow 3 GenAI APIs (built-in scorers, UC-backed traces, multi-turn simulation, prompt optimization) |
| **Track B — OSS** | `AgentOps with OSS MLflow.py` | OSS MLflow APIs only; uses code-based `@scorer` functions for evaluation |

> **Note:** Built-in LLM-judge scorers (`Correctness`, `Safety`, `Guidelines`, etc.), `ConversationSimulator`, and `optimize_prompts()` are currently available only in Managed MLflow on Databricks. OSS support is coming soon. The OSS notebook uses code-based `@scorer` functions for deterministic evaluation. Tracing (Pillar 1) and metric logging (Pillars 3–4) work fully on self-hosted MLflow.

---

## Key Differences Between Tracks

| Feature | Databricks Track | OSS Track |
|---|---|---|
| Evaluation scorers | Built-in LLM judges (`Correctness()`, `Safety()`, `Guidelines()`, etc.) | Code-based `@scorer` functions (`safety_check`, `answer_relevance`, etc.) |
| Multi-turn simulation | `ConversationSimulator` with `ConversationCompleteness()`, `UserFrustration()` | Markdown reference (Databricks only) |
| Prompt optimization | `GepaPromptOptimizer` with `mlflow.genai.optimize_prompts()` | Manual prompt iteration with version comparison |
| Trace storage | UC-backed Delta tables (`mfg_mc_se_sa.agent_ops`) | Standard MLflow tracking server |
| LLM endpoint | Databricks Foundation Models (`databricks-claude-sonnet-4`) | Databricks Foundation Models (or OpenAI `gpt-4o-mini` for external use) |

---

## Prerequisites

**Databricks Track**
- Databricks workspace with a running cluster
- Unity Catalog enabled; catalog/schema: `mfg_mc_se_sa.agent_ops`
- Access to Foundation Model APIs (`databricks-claude-sonnet-4`)

**OSS Track (on Databricks)**
- Same as above — the OSS notebook runs on Databricks by default using only OSS-compatible APIs

**OSS Track (self-hosted)**
- Python 3.9+
- `mlflow>=3.10`, `openai`
- A self-hosted MLflow tracking server (`mlflow server --host 0.0.0.0 --port 5000`)
- OpenAI API key (`OPENAI_API_KEY` env var) or another LLM provider
- See the "Running Outside Databricks" section in the notebook for the 5 code changes needed

---

## Setup

Both notebooks install their own dependencies in cell 2:

```python
# Databricks track
%pip install "mlflow[databricks]>=3.13.0" openai "gepa>=0.0.26" litellm --quiet

# OSS track
%pip install mlflow>=3.10 openai --quiet
```

Import the notebooks into your Databricks workspace and attach to a running cluster.

---

## Resources

| Resource | Link |
|---|---|
| MLflow Releases | [mlflow.org/releases](https://mlflow.org/releases/3.13.0/) |
| MLflow Tracing Docs | [mlflow.org/docs/latest/tracing](https://mlflow.org/docs/latest/tracing.html) |
| MLflow GenAI Evaluation | [docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/) |
| Prompt Optimization | [mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/](https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/) |
| Migrate Traces to UC | [docs.databricks.com/aws/en/mlflow3/genai/tracing/migrate-traces-to-uc](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/migrate-traces-to-uc) |
| Databricks MLflow 3 | [docs.databricks.com/aws/en/mlflow3/](https://docs.databricks.com/aws/en/mlflow3/) |
| MLflow Auth Plugin | [mlflow.org/docs/latest/auth/](https://mlflow.org/docs/latest/auth/index.html) |

---

If you find this useful, feel free to use it. If you extend it, a pull request is always welcome!
