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

Two tracks — same six pillars, same concepts, different MLflow runtime:

| Track | Notebook | When to Use |
|---|---|---|
| **Track A — Databricks** | `Agent Ops at Scale (Databricks).ipynb` | On Databricks; uses MLflow 3 GenAI APIs (`mlflow.genai.evaluate()`, built-in scorers, UC-backed traces) |
| **Track B — OSS** | `Agent Ops at Scale (OSS MLflow).ipynb` | Local / self-hosted MLflow; uses classic `mlflow.evaluate()` API |

> **Note:** `mlflow.genai.evaluate()`, built-in scorers (including `Safety()`), `ConversationSimulator`, and `optimize_prompts()` are currently available only in Managed MLflow on Databricks. OSS support is coming soon. Tracing (Pillar 1) and metric logging (Pillars 3–4) work fully on self-hosted MLflow.

---

## Key Differences Between Tracks

| Feature | Databricks Track | OSS Track |
|---|---|---|
| Evaluation API | `mlflow.genai.evaluate()` | `mlflow.genai.evaluate()` |
| Safety scorer | `Safety()` (built-in) | `Guidelines(name="safety", guidelines=[...])` |
| Trace storage | UC-backed Delta tables | Standard MLflow tracking server |
| Agent return type | `{"response": answer}` | Plain string |
| Prompt loading | `mlflow.genai.load_prompt()` | `mlflow.genai.load_prompt()` |

---

## Prerequisites

**Databricks Track**
- Databricks workspace with a running cluster
- Unity Catalog enabled; catalog/schema: `mfg_mc_se_sa.agent_observability`
- MLflow 3.x (included in Databricks Runtime 15.x+)

**OSS Track**
- Python 3.9+
- `mlflow>=2.14`, `openai`, `pandas`
- OpenAI API key (`OPENAI_API_KEY` env var)

---

## Setup

```bash
# OSS track only
pip install mlflow>=2.14 openai pandas
export OPENAI_API_KEY="your-key-here"
jupyter notebook "Agent Ops at Scale (OSS MLflow).ipynb"
```

For the Databricks track, import the notebook into your workspace and attach it to a running cluster.

---

## Resources

| Resource | Link |
|---|---|
| MLflow Tracing Docs | [mlflow.org/docs/latest/tracing](https://mlflow.org/docs/latest/tracing) |
| MLflow GenAI Evaluation (Databricks) | [docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/) |
| MLflow Evaluate (OSS) | [mlflow.org/docs/latest/llms/llm-evaluate](https://mlflow.org/docs/latest/llms/llm-evaluate) |
| Prompt Optimization | [mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/](https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/) |
| Migrate Traces to UC | [docs.databricks.com/aws/en/mlflow3/genai/tracing/migrate-traces-to-uc](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/migrate-traces-to-uc) |

---

## Files to Delete Before Using This Repo

The following files from the original repo are no longer relevant and should be removed:

- `kmeans_viz_template.Rmd` — old R clustering template from a previous project
- `Documentation/ProjectPlan.md` — internal planning doc, not relevant to this workshop

---

If you find this useful, feel free to use it. If you extend it, a pull request is always welcome!
