# Databricks notebook source
# MAGIC %md
# MAGIC # AgentOps with MLflow
# MAGIC **Practical MLflow Workflows for Monitoring, Evaluation & Continuous Improvement**
# MAGIC
# MAGIC *Stephanie Alba | Databricks*
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Databricks workspace with access to Foundation Model APIs
# MAGIC - MLflow 3.10+ (`mlflow[databricks]>=3.10`)
# MAGIC - Basic familiarity with Python and LLMs
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this notebook you will be able to:
# MAGIC 1. **Instrument** an agent with MLflow tracing and structured spans
# MAGIC 2. **Evaluate** agent quality using built-in and custom LLM-as-judge metrics
# MAGIC 3. **Integrate** human feedback signals into your evaluation workflow
# MAGIC 4. **Monitor** production quality, latency, and cost metrics
# MAGIC 5. **Govern** agent lifecycle with evaluation gates and run comparisons
# MAGIC 6. **Optimize** prompts systematically using MLflow's prompt optimization framework
# MAGIC
# MAGIC > **Workshop flow:** The companion slide deck frames the *why*; this notebook is the hands-on *how*.

# COMMAND ----------

# MAGIC %pip install "mlflow[databricks]>=3.13.0" openai "gepa>=0.0.26" litellm --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Why Agent Observability Matters
# MAGIC
# MAGIC ### Core AgentOps Challenges
# MAGIC
# MAGIC Building production agents is fundamentally harder than building traditional ML models. Here are the six challenges every team faces:
# MAGIC
# MAGIC | # | Challenge | Description |
# MAGIC |---|-----------|-------------|
# MAGIC | 1 | **Leveraging Enterprise Context** | Agents must ground responses in proprietary data -- knowledge bases, internal docs, databases -- without leaking or hallucinating. |
# MAGIC | 2 | **Reliability & Consistency** | Non-deterministic outputs mean the same question can yield different answers. You need guardrails to ensure consistent quality. |
# MAGIC | 3 | **End-to-End Observability** | Multi-step chains (retrieval -> reasoning -> tool use -> generation) create opaque pipelines. You need trace-level visibility. |
# MAGIC | 4 | **Safety & Guardrails** | Agents can generate harmful, biased, or off-topic content. Automated safety checks must run on every invocation. |
# MAGIC | 5 | **Scalability** | What works for 10 queries/day breaks at 10,000. Latency, cost, and quality all degrade differently under load. |
# MAGIC | 6 | **Maintainability** | Models change, data drifts, user expectations evolve. Without versioning and eval gates, regressions ship silently. |
# MAGIC
# MAGIC > These challenges are why "just log to a database" isn't enough. You need a structured observability framework.

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLflow Core Concepts for Agent Tracking
# MAGIC *The building blocks you'll use throughout this workshop*
# MAGIC
# MAGIC | Concept | What It Is | Example |
# MAGIC |---------|------------|---------|
# MAGIC | **Experiment** | A named container for related runs | `"customer-support-agent-v2"` |
# MAGIC | **Run** | One execution instance capturing params, metrics, artifacts & traces | A single evaluation pass over 50 test questions |
# MAGIC | **Metric** | Numeric signal logged over time | `latency_ms`, `quality_score`, `token_count` |
# MAGIC | **Artifact** | Any file: traces, eval results, prompt templates, datasets | `eval_results.json`, `trace.json` |
# MAGIC | **Trace** | A structured record of an agent's execution with nested spans | Retrieval span -> Generation span -> Tool span |
# MAGIC | **Tag** | Key-value metadata for organizing and filtering runs | `{"env": "prod", "version": "1.2.0"}` |
# MAGIC | **Prompt** | A versioned prompt template stored in the Prompt Registry | `mlflow.genai.register_prompt("support_agent", template=...)` |
# MAGIC
# MAGIC > An **Experiment** contains many **Runs**. Each **Run** captures **Metrics**, **Artifacts**, and **Traces**. **Prompts** are versioned independently and can be optimized over time.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pillar 1: Instrument and Trace an Agent
# MAGIC ---
# MAGIC
# MAGIC In this section we'll build a simple RAG-style support agent, instrument it with MLflow tracing, and explore the trace UI.
# MAGIC
# MAGIC ### Unity Catalog-Backed Traces
# MAGIC
# MAGIC ### Advanced: Unity Catalog-Backed Traces
# MAGIC
# MAGIC For production use, you can store traces in **Unity Catalog** Delta tables. See the [migration guide](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/migrate-traces-to-uc) for setup details.
# MAGIC
# MAGIC | Benefit | Description |
# MAGIC |---------|-------------|
# MAGIC | **SQL queryability** | Query traces with SQL — join with other Delta tables, build dashboards |
# MAGIC | **Governance** | UC permissions, column masking, audit logs apply to trace data |
# MAGIC | **Durability** | Traces stored as Delta tables with full ACID guarantees |
# MAGIC | **Cross-workspace access** | Share trace data across workspaces via Unity Catalog |

# COMMAND ----------

import mlflow
import os
from mlflow.entities.trace_location import UnityCatalog

# ── Configure MLflow experiment with UC-backed traces ─────────
mlflow.set_tracking_uri("databricks")
os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = "bce0a02b2be86f1b"

UC_CATALOG = "mfg_mc_se_sa"
UC_SCHEMA = "agent_ops"

experiment = mlflow.set_experiment(
    experiment_name="/Users/stephanie.rivera@databricks.com/WWC - AgentOps/AgentOps with MLflow",
    trace_location=UnityCatalog(
        catalog_name=UC_CATALOG,
        schema_name=UC_SCHEMA,
        table_prefix="agentops_workshop",
    ),
)

mlflow.openai.autolog()

print("MLflow tracking configured!")
print(f"  Experiment: {experiment.name}")
print(f"  Traces stored in UC: {UC_CATALOG}.{UC_SCHEMA}")
print(f"  Tracking URI: databricks")
print(f"  Experiment: AgentOps with MLflow")

# COMMAND ----------

import mlflow.genai

# ── Register prompt in the Prompt Registry ────────────────
# Prompts are UC-backed on Databricks — use catalog.schema.name format
support_prompt = mlflow.genai.register_prompt(
    name="mfg_mc_se_sa.agent_ops.support_agent_prompt",
    template=(
        "You are a helpful customer support agent. Use ONLY the following "
        "context to answer questions. If the context doesn't contain the "
        "answer, say so.\n\n"
        "Context:\n{{context}}\n\n"
        "Question: {{question}}"
    ),
)

print(f"Prompt registered: {support_prompt.name}")
print(f"  Version: {support_prompt.version}")
print(f"  URI: {support_prompt.uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Define a Knowledge Base
# MAGIC We'll use a simple in-memory knowledge base to simulate retrieval. In production, this would be a vector store, search index, or database.

# COMMAND ----------

from openai import OpenAI

# ── Simple knowledge base ─────────────────────────────────
KNOWLEDGE_BASE = {
    "returns": "Our return policy allows returns within 30 days of purchase. Items must be in original condition with receipt. Refunds are processed within 5-7 business days to the original payment method.",
    "shipping": "Standard shipping takes 5-7 business days. Express shipping (2-day) is available for $12.99. Free shipping on orders over $50. International shipping available to 40+ countries.",
    "warranty": "All electronics come with a 2-year manufacturer warranty. Extended warranty available for purchase. Warranty covers defects in materials and workmanship, not accidental damage.",
    "pricing": "We offer price matching within 14 days of purchase. Show us a competitor's lower price and we'll match it. Excludes marketplace sellers and clearance items.",
    "account": "You can manage your account at account.example.com. Reset passwords via email verification. Two-factor authentication is available and recommended for all accounts.",
}

@mlflow.trace(name="retrieve_context", span_type="RETRIEVER")
def retrieve_context(query: str) -> str:
    """Keyword-based retrieval from knowledge base."""
    query_lower = query.lower()
    results = []
    for topic, content in KNOWLEDGE_BASE.items():
        if topic in query_lower or any(word in query_lower for word in topic.split()):
            results.append(content)
    # Default fallback
    if not results:
        results = [KNOWLEDGE_BASE["returns"]]
    return "\n\n".join(results)

print("Knowledge base loaded with topics:", list(KNOWLEDGE_BASE.keys()))

# COMMAND ----------

import os

@mlflow.trace
def run_support_agent(question: str) -> dict:
    """RAG support agent with traced retrieval and generation spans."""
    # Step 1: Retrieve context
    context = retrieve_context(question)

    # Step 2: Load the versioned prompt from the registry
    prompt = mlflow.genai.load_prompt(support_prompt.uri)
    system_message = prompt.format(question=question, context=context)

    # Step 3: Generate response using Databricks Foundation Model
    client = OpenAI(
        api_key=os.environ.get("DATABRICKS_TOKEN", dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()),
        base_url=f"{mlflow.utils.databricks_utils.get_workspace_url()}/serving-endpoints"
    )

    with mlflow.start_span(name="llm_generation", span_type="LLM") as span:
        span.set_inputs({"question": question, "context": context})

        response = client.chat.completions.create(
            model="databricks-claude-sonnet-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": question}
            ],
            max_tokens=300,
            temperature=0.1
        )

        answer = response.choices[0].message.content
        span.set_outputs({"answer": answer})

    return {"response": answer}

print("Support agent defined and ready!")

# COMMAND ----------

# ── Run the agent on test questions ─────────────────────────
test_questions = [
    "What is your return policy?",
    "How long does shipping take?",
    "Does my laptop come with a warranty?",
    "Can you match a competitor's price?",
    "How do I reset my password?",
]

print("Running agent on 5 test questions...\n")
for i, question in enumerate(test_questions, 1):
    result = run_support_agent(question)
    print(f"Q{i}: {question}")
    print(f"A{i}: {result['response'][:150]}...")
    print()

print("All traces logged to MLflow! Check the Traces tab in the experiment UI.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checkpoint: Explore the Trace UI
# MAGIC
# MAGIC Now that we've run the agent, let's explore what MLflow captured.
# MAGIC
# MAGIC 1. Click **Experiments** in the left sidebar
# MAGIC 2. Open the **"AgentOps with MLflow"** experiment
# MAGIC 3. Click the **Traces** tab
# MAGIC 4. Click on any trace to see the waterfall view:
# MAGIC    - **support_agent** (root span)
# MAGIC      - **retrieve_context** (retriever span) -- inputs, outputs, latency
# MAGIC      - **llm_generation** (LLM span) -- prompt, response, token counts
# MAGIC
# MAGIC > **Key insight:** Every span captures inputs, outputs, and timing. This is how you debug multi-step agent failures in production.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pillar 2: Evaluation
# MAGIC ---
# MAGIC
# MAGIC Tracing tells you *what happened*. Evaluation tells you *how well it went*. This 6-step loop drives continuous improvement:
# MAGIC
# MAGIC | Step | Action | Details |
# MAGIC |------|--------|---------|
# MAGIC | 1. **Define** | Identify quality signals | What matters? Correctness, helpfulness, safety, groundedness |
# MAGIC | 2. **Collect** | Curate a golden dataset | Input/output pairs representative of production traffic |
# MAGIC | 3. **Evaluate** | Run `mlflow.genai.evaluate()` | Scorers produce Feedback on every trace |
# MAGIC | 4. **Analyze** | Compare runs in MLflow UI | Identify regressions, improvements, and edge cases |
# MAGIC | 5. **Improve** | Adjust prompts, retrieval, or model | Re-run evaluation. Ship only when metrics improve. |
# MAGIC | 6. **Repeat** | Add failing cases to golden dataset | The loop compounds quality over time |
# MAGIC
# MAGIC > **Start small:** Even 50 well-curated examples are transformative. You don't need thousands to begin.

# COMMAND ----------

# ── Prepare evaluation dataset ──────────────────────────────
eval_data = [
    {
        "inputs": {"question": "What is your return policy?"},
        "expectations": {"expected_response": "Returns are allowed within 30 days of purchase. Items must be in original condition with receipt. Refunds processed in 5-7 business days."},
    },
    {
        "inputs": {"question": "How long does standard shipping take?"},
        "expectations": {"expected_response": "Standard shipping takes 5-7 business days. Express 2-day shipping is $12.99. Free shipping on orders over $50."},
    },
    {
        "inputs": {"question": "What does the warranty cover?"},
        "expectations": {"expected_response": "Electronics have a 2-year manufacturer warranty covering defects in materials and workmanship. Extended warranty is available. Accidental damage is not covered."},
    },
    {
        "inputs": {"question": "Do you offer price matching?"},
        "expectations": {"expected_response": "Price matching is available within 14 days of purchase for competitor prices, excluding marketplace sellers and clearance items."},
    },
    {
        "inputs": {"question": "How do I set up two-factor authentication?"},
        "expectations": {"expected_response": "Manage your account at account.example.com. Two-factor authentication is available and recommended for all accounts."},
    },
]

print(f"Evaluation dataset: {len(eval_data)} examples")
for i, row in enumerate(eval_data, 1):
    print(f"  {i}. {row['inputs']['question']}")

# COMMAND ----------

from mlflow.genai.scorers import Correctness, RelevanceToQuery, Safety, Guidelines

# Suppress internal MLflow/pandas warnings we can't fix from user code
import warnings
warnings.filterwarnings("ignore", message=".*Inferred schema contains integer column.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*default dtype for empty Series.*")

# ── Run automated evaluation ────────────────────────────────
results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=run_support_agent,
    scorers=[
        Correctness(),
        RelevanceToQuery(),
        Safety(),
    ],
)

print("Evaluation complete! Check the MLflow Traces tab for per-row results.")
print(f"  Run ID: {results.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scorers Deep Dive
# MAGIC *Built-in scorers vs. custom judges*
# MAGIC
# MAGIC MLflow 3 provides a unified **scorer** interface. Built-in scorers cover common quality dimensions; custom judges and code-based scorers handle domain-specific needs:
# MAGIC
# MAGIC | Scenario | Approach | Example |
# MAGIC |----------|----------|---------|
# MAGIC | Factual correctness | `Correctness()` | Compares output to expected response |
# MAGIC | Relevant to the user's query | `RelevanceToQuery()` | Checks output addresses the question |
# MAGIC | Grounded in retrieved context | `RetrievalGroundedness()` | Verifies output uses retrieval results |
# MAGIC | Safety / harmful content | `Safety()` | Detects unsafe or harmful outputs |
# MAGIC | Custom natural-language rules | `Guidelines(name=..., guidelines=...)` | Pass/fail against your own rules |
# MAGIC | Fully custom LLM judge | `make_judge(name=..., instructions=...)` | Numerical, categorical, or boolean scores |
# MAGIC | Deterministic / code-based | `@scorer` decorator | Exact match, format validation, latency checks |
# MAGIC
# MAGIC > **Key principle:** Scorers receive a **Trace** and return **Feedback** attached to that trace. The same scorers work in both development evaluation and production monitoring.

# COMMAND ----------

from mlflow.genai.scorers import Guidelines, scorer
from mlflow.entities import Feedback

# ── Define custom scorers ─────────────────────────────────────

# Option 1: Guidelines scorer — pass/fail against natural-language rules
helpfulness_judge = Guidelines(
    name="helpfulness",
    guidelines=[
        "The response must directly address the user's question",
        "The response must provide specific, actionable information",
        "The response must not redirect the user without answering",
    ],
)

# Option 2: Code-based scorer — deterministic checks using @scorer
@scorer
def response_length_check(outputs: str) -> Feedback:
    """Check that the response is substantive (not too short or too long)."""
    word_count = len(outputs.split()) if outputs else 0
    if word_count < 10:
        return Feedback(value="no", rationale=f"Response too short ({word_count} words). Expected at least 10.")
    elif word_count > 500:
        return Feedback(value="no", rationale=f"Response too long ({word_count} words). Expected under 500.")
    else:
        return Feedback(value="yes", rationale=f"Response length is appropriate ({word_count} words).")

print("Custom scorers defined:")
print("  1. helpfulness (Guidelines judge — pass/fail against 3 rules)")
print("  2. response_length_check (code-based scorer — deterministic)")

# COMMAND ----------

# ── Run evaluation with custom scorers ─────────────────────
custom_results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=run_support_agent,
    scorers=[
        Correctness(),
        helpfulness_judge,
        response_length_check,
    ],
)

print("Custom evaluation complete! Check the MLflow Traces tab for per-row feedback.")
print(f"  Run ID: {custom_results.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Multi-Turn Conversation Evaluation
# MAGIC
# MAGIC Single-turn evaluation scores individual question-answer pairs. But real agents handle **multi-turn conversations** where quality emerges over several exchanges — follow-up questions, clarifications, and context that builds across turns.
# MAGIC
# MAGIC MLflow provides two approaches:
# MAGIC
# MAGIC | Approach | How It Works | Best For |
# MAGIC |----------|-------------|----------|
# MAGIC | **Evaluate existing conversations** | Pass traced sessions to `mlflow.genai.evaluate()` | Production conversations you've already captured |
# MAGIC | **Simulate conversations** | `ConversationSimulator` generates multi-turn dialogues with a simulated user | Testing new agents before deployment |
# MAGIC
# MAGIC **Built-in multi-turn scorers:**
# MAGIC - `ConversationCompleteness()` — Did the agent fully resolve the user's goal?
# MAGIC - `UserFrustration()` — Did the user become frustrated? Did the agent make it worse?

# COMMAND ----------

from mlflow.genai.simulators import ConversationSimulator
from mlflow.genai.scorers import ConversationCompleteness, UserFrustration

# ── Ensure DATABRICKS_HOST is set for the simulator's internal LLM calls ──
workspace_url = mlflow.utils.databricks_utils.get_workspace_url()
if not workspace_url.startswith("http"):
    workspace_url = f"https://{workspace_url}"
os.environ["DATABRICKS_HOST"] = workspace_url
if "DATABRICKS_TOKEN" not in os.environ:
    os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# ── Define a multi-turn predict function ──────────────────────
# ConversationSimulator passes chat messages (list of dicts) to predict_fn
def multi_turn_predict(input: list[dict], **kwargs) -> str:
    """Agent that handles multi-turn conversations."""
    all_context = "\n\n".join(KNOWLEDGE_BASE.values())
    messages = [
        {"role": "system", "content": f"You are a helpful customer support agent. Use the following context to answer questions. If the context doesn't contain the answer, say so.\n\n{all_context}"}
    ] + input

    client = OpenAI(
        api_key=os.environ["DATABRICKS_TOKEN"],
        base_url=f"{os.environ['DATABRICKS_HOST']}/serving-endpoints"
    )
    response = client.chat.completions.create(
        model="databricks-claude-sonnet-4",
        messages=messages,
        max_tokens=300,
        temperature=0.1,
    )
    return response.choices[0].message.content

print("Multi-turn predict function defined!")

# COMMAND ----------

# ── Simulate and evaluate multi-turn conversations ────────
simulator = ConversationSimulator(
    test_cases=[
        {
            "goal": "Get a full understanding of the return policy including timeframes and conditions",
            "persona": "You are a new customer with a straightforward question",
        },
        {
            "goal": "Compare shipping options and decide which is best for an urgent order",
            "persona": "You are a busy professional who needs a quick, clear answer",
        },
        {
            "goal": "Understand warranty coverage and whether accidental damage is included",
            "persona": "You are a skeptical customer who asks detailed follow-up questions",
        },
    ],
    max_turns=4,
    user_model="databricks:/databricks-claude-sonnet-4",
)

multi_turn_results = mlflow.genai.evaluate(
    data=simulator,
    predict_fn=multi_turn_predict,
    scorers=[ConversationCompleteness(), UserFrustration()],
)

print("Multi-turn evaluation complete!")
print(f"  Run ID: {multi_turn_results.run_id}")
print("\nCheck the MLflow Traces tab — each simulated conversation appears as a session.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pillar 3: Human Feedback
# MAGIC ---
# MAGIC *Align automated scores with real-world quality signals*
# MAGIC
# MAGIC Automated evaluation is necessary but not sufficient. Human feedback grounds your metrics in reality.
# MAGIC
# MAGIC | Signal Type | Source | How to Capture |
# MAGIC |-------------|--------|----------------|
# MAGIC | **Explicit** | User clicks thumbs up/down | Log as `mlflow.log_metric("user_rating", 1)` tied to trace |
# MAGIC | **Corrections** | User rephrases or edits response | Capture before/after pair as artifact |
# MAGIC | **Annotations** | Internal reviewers label quality | Batch-annotate traces, log labels |
# MAGIC | **Implicit** | User follows up, asks for clarification, or abandons | Infer satisfaction from session behavior |
# MAGIC
# MAGIC **Feedback loop steps:**
# MAGIC 1. Capture explicit signals (thumbs up/down, star ratings) in your agent UI
# MAGIC 2. Collect corrections and annotations -- these become your highest-quality training data
# MAGIC 3. Validate your LLM judges -- compare automated scores against human ratings on 50+ examples
# MAGIC 4. Feed back into your golden dataset -- add labeled production examples weekly

# COMMAND ----------

import random
import numpy as np

# ── Simulate human feedback and judge agreement ─────────────
with mlflow.start_run(run_name="human-feedback-analysis"):
    # Simulate 50 rated examples
    simulated_feedback = []
    for i in range(50):
        human_score = random.choice([1, 2, 3, 4, 5])
        # Simulate a reasonably-aligned judge (correlation ~0.75)
        judge_score = max(1, min(5, human_score + random.choice([-1, 0, 0, 0, 1])))
        simulated_feedback.append({"human": human_score, "judge": judge_score})

        mlflow.log_metric("human_rating", human_score, step=i)
        mlflow.log_metric("judge_rating", judge_score, step=i)

    # Compute agreement rate
    human_scores = [f["human"] for f in simulated_feedback]
    judge_scores = [f["judge"] for f in simulated_feedback]
    exact_match = sum(1 for h, j in zip(human_scores, judge_scores) if h == j) / len(human_scores)
    within_one = sum(1 for h, j in zip(human_scores, judge_scores) if abs(h - j) <= 1) / len(human_scores)
    correlation = np.corrcoef(human_scores, judge_scores)[0, 1]

    mlflow.log_metrics({
        "agreement_exact": exact_match,
        "agreement_within_one": within_one,
        "human_judge_correlation": correlation,
    })

    print(f"Human-Judge Agreement Analysis (n=50):")
    print(f"  Exact match:  {exact_match:.1%}")
    print(f"  Within +/- 1: {within_one:.1%}")
    print(f"  Correlation:  {correlation:.3f}")
    print(f"\n  Recommendation: {'Judge is well-calibrated!' if correlation > 0.7 else 'Consider refining your grading prompt.'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pillar 4: Production Monitoring
# MAGIC *Track quality, latency, and cost in production*
# MAGIC
# MAGIC Development evaluation catches regressions before deployment. Production monitoring catches issues that only appear under real-world conditions.
# MAGIC
# MAGIC | Capability | What to Track | Databricks Tool |
# MAGIC |------------|---------------|-----------------|
# MAGIC | **Quality dashboards** | Automated eval scores aggregated hourly/daily | Lakehouse Monitoring + MLflow metrics |
# MAGIC | **Trace sampling** | Random sample of production traces for manual review | MLflow Traces + scheduled jobs |
# MAGIC | **Latency monitoring** | p50/p95/p99 latency per span, overall request latency | Lakehouse Monitoring |
# MAGIC | **Cost tracking** | Token usage, model calls, and $ cost per request | MLflow metrics + billing API |
# MAGIC | **Drift detection** | Input distribution shifts, topic drift, quality degradation | Lakehouse Monitoring drift profiles |
# MAGIC | **Alerting** | Threshold-based or anomaly-based alerts on any metric | Databricks Alerts + SQL Alerts |
# MAGIC
# MAGIC **Architecture pattern:**
# MAGIC ```
# MAGIC Agent Request -> MLflow Trace -> Delta Table -> Lakehouse Monitor -> Alert
# MAGIC                                              -> Dashboard
# MAGIC                                              -> Scheduled Eval Job
# MAGIC ```

# COMMAND ----------

import time

# ── Simulate production monitoring batch ────────────────────
with mlflow.start_run(run_name="prod-monitoring-batch"):
    # Simulate a batch of production metrics
    for step in range(20):
        # Simulate realistic production metrics
        latency = random.gauss(450, 120)  # ms, normally distributed
        quality = random.uniform(0.6, 1.0)
        tokens = random.randint(150, 800)
        cost = tokens * 0.000003  # ~$3 per million tokens

        mlflow.log_metrics({
            "prod/latency_ms": max(50, latency),
            "prod/quality_score": quality,
            "prod/token_count": tokens,
            "prod/cost_usd": cost,
        }, step=step)

    # Log aggregate metrics
    mlflow.log_metrics({
        "prod/latency_p50": 430,
        "prod/latency_p95": 680,
        "prod/latency_p99": 890,
        "prod/avg_quality": 0.82,
        "prod/total_cost_usd": 0.045,
        "prod/request_count": 20,
    })

    mlflow.set_tags({
        "monitoring.batch": "true",
        "monitoring.window": "1h",
        "env": "production",
    })

    print("Production monitoring batch logged!")
    print("  20 requests simulated with latency, quality, cost metrics")
    print("  Aggregate percentiles and totals computed")
    print("\n  Check the MLflow UI: Metrics tab -> prod/* namespace")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pillar 5: Governance & Lifecycle
# MAGIC *Ship responsibly at enterprise scale*
# MAGIC
# MAGIC | Practice | Description | Databricks Tool |
# MAGIC |----------|-------------|-----------------|
# MAGIC | **Model Registry** | Version every model, prompt template, and retrieval config | Unity Catalog Model Registry |
# MAGIC | **Eval gates in CI/CD** | Run `mlflow.evaluate()` on every PR that touches agent behavior | Databricks Jobs + GitHub Actions |
# MAGIC | **Deployment stages** | Promote through dev -> staging -> prod with full lineage | UC model aliases + deployment tags |
# MAGIC | **Audit trails** | Record who changed what, when, and why | Unity Catalog lineage + MLflow tags |
# MAGIC | **Access control** | Team-level experiment permissions, RBAC on model endpoints | Workspace permissions + UC grants |
# MAGIC | **Data governance** | Govern traces containing sensitive information | UC-backed traces + column masking |
# MAGIC
# MAGIC **CI/CD Evaluation Gate (pseudocode):**
# MAGIC ```python
# MAGIC # In your CI/CD pipeline:
# MAGIC from mlflow.genai.scorers import Correctness, Safety, Guidelines
# MAGIC
# MAGIC results = mlflow.genai.evaluate(
# MAGIC     data=golden_dataset,
# MAGIC     predict_fn=new_agent,
# MAGIC     scorers=[Correctness(), Safety(), Guidelines(name="quality", guidelines=...)],
# MAGIC )
# MAGIC traces = mlflow.search_traces(run_id=results.run_id)
# MAGIC # Check assessments for failures and block deployment if quality regresses
# MAGIC ```

# COMMAND ----------

# ── Eval gate simulation: review traces with feedback ─────
eval_traces = mlflow.search_traces(
    run_id=results.run_id,
)

if len(eval_traces) > 0:
    print("=== Evaluation Traces ===\n")
    display(eval_traces[["trace_id", "request", "response", "assessments"]].head())

    print("\nEval gate check: Review assessments (scorer feedback) on each trace.")
    print("If any key scorer fails, block the deployment.")
else:
    print("No evaluation traces found yet. Run the evaluation cells above first!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pillar 6: Prompt Optimization
# MAGIC ---
# MAGIC
# MAGIC We registered our prompt in the Prompt Registry at the start of this notebook. Now we can systematically improve it using evaluation data and LLM-driven reflection:
# MAGIC
# MAGIC 1. **Define** a predict function that loads the registered prompt
# MAGIC 2. **Run** `mlflow.genai.optimize_prompts()` with training data and scorers
# MAGIC 3. **Get back** an improved prompt version, automatically registered as a new version
# MAGIC
# MAGIC | Optimizer | How It Works | Best For |
# MAGIC |-----------|-------------|----------|
# MAGIC | **GepaPromptOptimizer** | Genetic-Pareto optimization with natural language reflection. Up to 35x fewer iterations than brute force. | High-stakes tasks with substantial eval datasets |
# MAGIC | **MetaPromptOptimizer** | Zero-shot or few-shot metaprompting. Fast, low cost. | Quick improvements, small datasets or no data at all |
# MAGIC
# MAGIC > **Docs:** [mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts](https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/)

# COMMAND ----------

from openai import OpenAI

# ── Step 1: Define a predict function that loads the registered prompt ──
def predict_with_prompt(question: str, context: str) -> str:
    """Predict function that loads the prompt from the registry."""
    prompt = mlflow.genai.load_prompt(f"prompts:/{support_prompt.name}/{support_prompt.version}")

    client = OpenAI(
        api_key=os.environ.get("DATABRICKS_TOKEN", dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()),
        base_url=f"{mlflow.utils.databricks_utils.get_workspace_url()}/serving-endpoints"
    )
    response = client.chat.completions.create(
        model="databricks-claude-sonnet-4",
        messages=[{"role": "user", "content": prompt.format(question=question, context=context)}],
        max_tokens=300,
        temperature=0.1,
    )
    return response.choices[0].message.content

# Quick test
test_result = predict_with_prompt("What is your return policy?", KNOWLEDGE_BASE["returns"])
print(f"Test response: {test_result[:150]}...")

# COMMAND ----------

from mlflow.genai.optimize import GepaPromptOptimizer
import litellm

# ── Step 2: Prepare training data for optimization ──────────
train_data = [
    {
        "inputs": {"question": "What is your return policy?", "context": KNOWLEDGE_BASE["returns"]},
        "expectations": {"expected_response": "Returns within 30 days with receipt. Refunds in 5-7 business days."},
    },
    {
        "inputs": {"question": "How long does shipping take?", "context": KNOWLEDGE_BASE["shipping"]},
        "expectations": {"expected_response": "Standard shipping 5-7 business days. Express 2-day for $12.99. Free over $50."},
    },
    {
        "inputs": {"question": "What does the warranty cover?", "context": KNOWLEDGE_BASE["warranty"]},
        "expectations": {"expected_response": "2-year warranty on electronics for defects. Extended warranty available. No accidental damage."},
    },
    {
        "inputs": {"question": "Do you price match?", "context": KNOWLEDGE_BASE["pricing"]},
        "expectations": {"expected_response": "Price matching within 14 days, excluding marketplace sellers and clearance."},
    },
    {
        "inputs": {"question": "How do I enable 2FA?", "context": KNOWLEDGE_BASE["account"]},
        "expectations": {"expected_response": "Go to account.example.com. Two-factor authentication is available for all accounts."},
    },
]

# ── Step 3: Run prompt optimization ─────────────────────────
optimizer = GepaPromptOptimizer(
    reflection_model="endpoints:/databricks-claude-sonnet-4",
    max_metric_calls=50,  # Keep cost low for workshop demo
)

result = mlflow.genai.optimize_prompts(
    predict_fn=predict_with_prompt,
    train_data=train_data,
    prompt_uris=[support_prompt.uri],
    optimizer=optimizer,
    scorers=[Correctness(model="endpoints:/databricks-claude-sonnet-4")],
)

print("\n=== Prompt Optimization Results ===")
print(f"  Initial score: {result.initial_eval_score}")
print(f"  Final score:   {result.final_eval_score}")
for prompt in result.optimized_prompts:
    print(f"\n  Optimized prompt '{prompt.name}' v{prompt.version}:")
    print(f"  {prompt.template[:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Practical Tips for Production Deployments
# MAGIC ---
# MAGIC *Lessons from teams who've done this at scale*
# MAGIC
# MAGIC | # | Tip | Why It Matters |
# MAGIC |---|-----|---------------|
# MAGIC | 1 | **Start with a golden dataset of 50-100 examples** | Even a small, well-curated eval set is transformative. Prioritize coverage of known failure modes and edge cases. |
# MAGIC | 2 | **Log costs alongside quality** | Track `token_count` and `model_id` alongside `quality_score`. Cost/quality tradeoffs are your most important business metric. |
# MAGIC | 3 | **Tag runs with environment and version** | Use `tags: {"env": "prod", "version": "1.2.0", "git_sha": "abc123"}`. This makes debugging regressions 10x faster. |
# MAGIC | 4 | **Validate your LLM judge before trusting it** | Run 50 examples through both your judge and human raters. If correlation < 0.7, revisit your `grading_prompt`. |
# MAGIC | 5 | **Sample production traces for manual review** | Don't rely solely on automated scores. Review 10-20 traces weekly to catch issues metrics miss. |
# MAGIC | 6 | **Version your eval datasets** | Log eval datasets as MLflow artifacts. When you add new examples, you can compare performance across dataset versions. |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC ---
# MAGIC
# MAGIC | Pillar | What We Covered | Key MLflow API |
# MAGIC |--------|----------------|----------------|
# MAGIC | **1. Tracing & Observability** | Instrumented an agent with nested spans for retrieval and generation | `@mlflow.trace`, `mlflow.start_span()` |
# MAGIC | **2. Evaluation** | Ran automated evaluation with built-in scorers and custom judges | `mlflow.genai.evaluate()`, `Guidelines`, `@scorer` |
# MAGIC | **3. Human Feedback** | Simulated human feedback collection and computed judge agreement rates | `mlflow.log_metric()`, correlation analysis |
# MAGIC | **4. Production Monitoring** | Logged production-style metrics (latency, quality, cost) at scale | `mlflow.log_metrics()`, tags, namespaced metrics |
# MAGIC | **5. Governance & Lifecycle** | Reviewed trace assessments and simulated CI/CD eval gates | `mlflow.search_traces()`, assessments |
# MAGIC | **6. Prompt Optimization** | Registered prompts and ran automated prompt optimization | `mlflow.genai.optimize_prompts()`, `GepaPromptOptimizer` |
# MAGIC
# MAGIC ### Next Steps
# MAGIC - **Expand your golden dataset** with real production examples
# MAGIC - **Set up Lakehouse Monitoring** for continuous quality tracking
# MAGIC - **Integrate eval gates** into your CI/CD pipeline
# MAGIC - **Explore the OSS notebook** if you work with self-hosted MLflow
# MAGIC
# MAGIC ### Resources
# MAGIC - [MLflow Tracing Docs](https://mlflow.org/docs/latest/tracing.html)
# MAGIC - [MLflow GenAI Evaluation](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/)
# MAGIC - [MLflow Prompt Optimization](https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/)
# MAGIC - [Migrate Traces to Unity Catalog](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/migrate-traces-to-uc)
# MAGIC - [Databricks MLflow 3 GenAI](https://docs.databricks.com/aws/en/mlflow3/genai/)

# COMMAND ----------

# ── Cleanup & experiment URL ────────────────────────────────
experiment = mlflow.get_experiment_by_name("/Users/stephanie.rivera@databricks.com/WWC - AgentOps/AgentOps with MLflow")
if experiment:
    workspace_url = mlflow.utils.databricks_utils.get_workspace_url()
    print(f"Experiment URL: {workspace_url}/#mlflow/experiments/{experiment.experiment_id}")
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"\nTotal runs in this experiment can be viewed in the MLflow UI.")
else:
    print("Experiment not found. It will be created when you run the configuration cell.")