# iDataLikeThat

A collection of data science projects, notebooks, and templates.  
If you find something useful, use it. If you expand on it, pull requests are welcome!

---

## Projects

### [AgentOps with Managed MLflow on Databricks](./AIBuilders2026/)
**AI Builders 2026 · 20-minute walkthrough · with Sonakshi Pandey**

One customer-support agent, taken through the AgentOps loop end to end:

- **Trace:** a "lazy" prompt ships, nothing errors, and the trace shows exactly what the agent did
- **Evaluate:** LLM judges plus a plain-English helpfulness rule catch what correctness alone would ship
- **Improve:** GEPA rewrites the prompt from the judges' feedback (`mlflow.genai.optimize_prompts()`)
- **Govern:** an eval gate blocks or approves the change, and the `@production` prompt alias moves after review

Built on MLflow 3.17 with Unity Catalog-backed traces and the Prompt Registry. To run it in your workspace, set `UC_CATALOG`, `UC_SCHEMA`, and the SQL warehouse ID in the **Configure MLflow** cell.

### [AgentOps with MLflow](./WomenWhoCode2026/)
**Practical MLflow Workflows for Monitoring, Evaluation & Continuous Improvement**

Hands-on workshop notebooks covering the six pillars of Agent Ops: tracing, evaluation, human feedback, production monitoring, governance, and prompt optimization — using MLflow on Databricks and open-source.

Two tracks available:
- **Track A (Databricks):** MLflow 3 GenAI APIs with Unity Catalog-backed traces
- **Track B (OSS):** Classic `mlflow.evaluate()` on self-hosted MLflow

---

*More projects coming someday.*
