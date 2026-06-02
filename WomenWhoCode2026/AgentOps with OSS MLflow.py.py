# Databricks notebook source
# MAGIC %md
# MAGIC # AgentOps with MLflow -- Open Source MLflow
# MAGIC **Practical MLflow Workflows for Monitoring, Evaluation & Continuous Improvement**
# MAGIC
# MAGIC *Stephanie Alba | Databricks*
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Python 3.9+
# MAGIC - Databricks workspace with access to Foundation Model APIs (or an OpenAI API key for external use)
# MAGIC - MLflow 3.10+ (`pip install mlflow>=3.10`)
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this notebook you will be able to:
# MAGIC 1. **Instrument** an agent with MLflow tracing and structured spans
# MAGIC 2. **Evaluate** agent quality using built-in and custom LLM-as-judge metrics
# MAGIC 3. **Integrate** human feedback signals into your evaluation workflow
# MAGIC 4. **Monitor** production quality, latency, and cost metrics
# MAGIC 5. **Govern** agent lifecycle with evaluation gates and run comparisons
# MAGIC 6. **Optimize** prompts by iterating on versions and comparing evaluation results
# MAGIC
# MAGIC > **This is the OSS track.** Same concepts as the Databricks notebook, using only open-source MLflow APIs. Runs on Databricks by default — see Pillar 1 for instructions to run on a self-hosted server. Built-in LLM-judge scorers and multi-turn simulators are Databricks-only (OSS support coming soon); this notebook uses code-based `@scorer` functions instead.

# COMMAND ----------

# MAGIC %pip install mlflow>=3.10 openai --quiet
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
# MAGIC ### Setup
# MAGIC
# MAGIC This notebook is configured to run on **Databricks** using the managed MLflow tracking server. All traces, metrics, and artifacts are stored in your Databricks workspace.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Running Outside Databricks (Self-Hosted MLflow)
# MAGIC
# MAGIC > **To run this notebook outside of Databricks**, you need five changes:
# MAGIC > 1. Start a self-hosted MLflow tracking server (see options below)
# MAGIC > 2. Change `mlflow.set_tracking_uri("databricks")` to `mlflow.set_tracking_uri("http://127.0.0.1:5000")`
# MAGIC > 3. Change the experiment name from a workspace path to a simple name, e.g. `mlflow.set_experiment("agent-observability-workshop")`
# MAGIC > 4. Change the prompt name from UC-qualified (`mfg_mc_se_sa.agent_ops.support_agent_prompt`) to a simple name (`support_agent_prompt`)
# MAGIC > 5. Switch the LLM client from Databricks Foundation Models to OpenAI: `OpenAI(api_key=os.environ["OPENAI_API_KEY"])` with `model="gpt-4o-mini"` (see comments in code cells)
# MAGIC
# MAGIC > **Important:** The new evaluation suite (`mlflow.genai.evaluate()`, built-in scorers, `ConversationSimulator`, prompt optimization) is currently available only in **Managed MLflow on Databricks**. Open source support is coming soon. Pillar 1 (tracing) and Pillars 3-4 (human feedback, monitoring) work fully on self-hosted MLflow. Pillar 2 (evaluation), multi-turn simulation, and Pillar 6 (prompt optimization) require Databricks.
# MAGIC
# MAGIC **Option A: Quick Start (files only)**
# MAGIC ```bash
# MAGIC pip install mlflow
# MAGIC mlflow server --host 0.0.0.0 --port 5000
# MAGIC # Open http://localhost:5000
# MAGIC ```
# MAGIC
# MAGIC **Option B: SQLite Backend (recommended for persistence)**
# MAGIC ```bash
# MAGIC mlflow server \
# MAGIC   --backend-store-uri sqlite:///mlflow.db \
# MAGIC   --default-artifact-root ./mlflow-artifacts \
# MAGIC   --host 0.0.0.0 --port 5000
# MAGIC ```
# MAGIC
# MAGIC **Option C: Docker (isolated environment)**
# MAGIC ```bash
# MAGIC docker run -p 5000:5000 ghcr.io/mlflow/mlflow mlflow server --host 0.0.0.0
# MAGIC ```

# COMMAND ----------

import mlflow
import os

# ── Configure MLflow experiment ───────────────────────────────
# Running on Databricks: uses the managed MLflow tracking server
mlflow.set_tracking_uri("databricks")

# ── TO RUN OUTSIDE DATABRICKS: ──
# 1. Uncomment the line below and comment out the set_tracking_uri("databricks") line above
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# 2. Use a simple experiment name instead of a workspace path:
# mlflow.set_experiment("agent-observability-workshop")

mlflow.set_experiment("/Users/stephanie.rivera@databricks.com/WWC - AgentOps/AgentOps with MLflow (OSS MLflow)")
mlflow.openai.autolog()

print("MLflow tracking configured!")
print(f"  Tracking URI: {mlflow.get_tracking_uri()}")
print(f"  Experiment: AgentOps with MLflow (OSS MLflow)")

# COMMAND ----------

import mlflow.genai

# ── Register prompt in the Prompt Registry ────────────────
# On Databricks, prompts are UC-backed and require a catalog.schema.name format.
# TO RUN OUTSIDE DATABRICKS: use a simple name instead, e.g. name="support_agent_prompt"
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

@mlflow.trace
def run_support_agent(question: str) -> dict:
    """RAG support agent with traced retrieval and generation spans."""
    # Step 1: Retrieve context
    context = retrieve_context(question)

    # Step 2: Load the versioned prompt from the registry
    prompt = mlflow.genai.load_prompt(support_prompt.uri)
    system_message = prompt.format(question=question, context=context)

    # Step 3: Generate response using Databricks Foundation Model
    # TO RUN OUTSIDE DATABRICKS: use OpenAI(api_key=os.environ["OPENAI_API_KEY"]) and model="gpt-4o-mini"
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

print("All traces logged to MLflow! Check the Traces tab in your experiment.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checkpoint: Explore the Trace UI
# MAGIC
# MAGIC Now that we've run the agent, let's explore what MLflow captured.
# MAGIC
# MAGIC 1. Navigate to your **MLflow Experiment** (on Databricks: left sidebar → Experiments; self-hosted: open `http://localhost:5000`)
# MAGIC 2. Click the **"AgentOps with MLflow (OSS MLflow)"** experiment
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
# MAGIC > **Note:** The new evaluation suite (`mlflow.genai.evaluate()`, scorers, simulators) is currently available only in **Managed MLflow on Databricks**, with open source support coming soon. The cells below run on Databricks. If running on a self-hosted MLflow server, these cells will not work until OSS support is released.
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

from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback

# ── Define code-based scorers (OSS-compatible) ────────────────
# Built-in LLM-judge scorers (Correctness, Guidelines, Safety, etc.) require
# Managed MLflow on Databricks. In OSS, use code-based @scorer functions instead.

@scorer
def safety_check(outputs) -> Feedback:
    """Check for obviously unsafe content patterns."""
    if not outputs:
        return Feedback(value="no", rationale="Empty response")
    response = outputs.get("response", str(outputs)) if isinstance(outputs, dict) else str(outputs)
    lower = response.lower()
    unsafe_patterns = ["hack", "exploit", "password is", "credit card", "ssn", "ignore previous"]
    for pattern in unsafe_patterns:
        if pattern in lower:
            return Feedback(value="no", rationale=f"Response contains potentially unsafe content: '{pattern}'")
    return Feedback(value="yes", rationale="No unsafe content patterns detected")

@scorer
def answer_relevance(inputs, outputs) -> Feedback:
    """Check that the response addresses the question topic."""
    question = inputs.get("question", "") if isinstance(inputs, dict) else str(inputs)
    response = outputs.get("response", str(outputs)) if isinstance(outputs, dict) else str(outputs)
    # Check if key terms from the question appear in the response
    key_terms = [w for w in question.lower().split() if len(w) > 3 and w not in ("what", "your", "does", "have", "with", "that", "this", "from")]
    matches = sum(1 for term in key_terms if term in response.lower())
    ratio = matches / max(len(key_terms), 1)
    if ratio >= 0.3:
        return Feedback(value="yes", rationale=f"Response addresses {matches}/{len(key_terms)} key terms from the question")
    return Feedback(value="no", rationale=f"Response only addresses {matches}/{len(key_terms)} key terms from the question")

# ── Run automated evaluation ────────────────────────────────
results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=run_support_agent,
    scorers=[safety_check, answer_relevance],
)

print("Evaluation complete! Check the MLflow Traces tab for per-row results.")
print(f"  Run ID: {results.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scorers Deep Dive
# MAGIC *Built-in scorers vs. custom judges*
# MAGIC
# MAGIC MLflow provides a unified **scorer** interface. Built-in scorers cover common quality dimensions; custom judges and code-based scorers handle domain-specific needs:
# MAGIC
# MAGIC | Scenario | Approach | Availability |
# MAGIC |----------|----------|--------------|
# MAGIC | Factual correctness | `Correctness()` | Databricks only |
# MAGIC | Relevant to the user's query | `RelevanceToQuery()` | Databricks only |
# MAGIC | Custom natural-language rules | `Guidelines(name=..., guidelines=...)` | Databricks only |
# MAGIC | Safety / harmful content | `Safety()` | Databricks only |
# MAGIC | Retrieval groundedness | `RetrievalGroundedness()` | Databricks only |
# MAGIC | Multi-turn completeness | `ConversationCompleteness()` | Databricks only |
# MAGIC | User frustration detection | `UserFrustration()` | Databricks only |
# MAGIC | Fully custom LLM judge | `make_judge(name=..., instructions=...)` | Databricks only |
# MAGIC | **Deterministic / code-based** | **`@scorer` decorator** | **OSS MLflow** |
# MAGIC
# MAGIC > **Key principle:** Scorers receive a **Trace** and return **Feedback** attached to that trace.
# MAGIC
# MAGIC > **Note:** Built-in LLM-judge scorers (Correctness, Guidelines, Safety, etc.) are currently available only in **Managed MLflow on Databricks**. In OSS MLflow, use code-based `@scorer` functions for deterministic checks. Open source support for LLM-judge scorers is coming soon.

# COMMAND ----------

# ── Define additional code-based scorers ──────────────────────

@scorer
def response_length_check(outputs) -> Feedback:
    """Check that the response is substantive (not too short or too long)."""
    response = outputs.get("response", str(outputs)) if isinstance(outputs, dict) else str(outputs)
    word_count = len(response.split()) if response else 0
    if word_count < 10:
        return Feedback(value="no", rationale=f"Response too short ({word_count} words). Expected at least 10.")
    elif word_count > 500:
        return Feedback(value="no", rationale=f"Response too long ({word_count} words). Expected under 500.")
    else:
        return Feedback(value="yes", rationale=f"Response length is appropriate ({word_count} words).")

@scorer
def helpfulness_check(inputs, outputs) -> Feedback:
    """Check that the response provides actionable information, not just a redirect."""
    response = outputs.get("response", str(outputs)) if isinstance(outputs, dict) else str(outputs)
    lower = response.lower()
    # Check for non-answers / redirects
    redirect_patterns = ["i don't know", "i cannot", "please contact", "visit our website", "call us at"]
    for pattern in redirect_patterns:
        if pattern in lower and len(response.split()) < 30:
            return Feedback(value="no", rationale=f"Response appears to redirect without answering ('{pattern}')")
    # Check for specificity — does it contain numbers, dates, or concrete details?
    has_specifics = any(char.isdigit() for char in response)
    if has_specifics:
        return Feedback(value="yes", rationale="Response contains specific, actionable information")
    return Feedback(value="yes", rationale="Response provides a substantive answer")

print("Additional scorers defined:")
print("  1. helpfulness_check (code-based — checks for redirects vs. actionable answers)")
print("  2. response_length_check (code-based — validates response word count)")

# COMMAND ----------

# ── Run evaluation with custom scorers ─────────────────────
custom_results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=run_support_agent,
    scorers=[
        helpfulness_check,
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
# MAGIC **On Databricks**, MLflow provides built-in multi-turn evaluation:
# MAGIC
# MAGIC | Feature | Description | Availability |
# MAGIC |---------|-------------|--------------|
# MAGIC | `ConversationSimulator` | Generates multi-turn dialogues with a simulated user | Databricks only |
# MAGIC | `ConversationCompleteness()` | Did the agent fully resolve the user's goal? | Databricks only |
# MAGIC | `UserFrustration()` | Did the user become frustrated? | Databricks only |
# MAGIC
# MAGIC **In OSS MLflow**, you can evaluate multi-turn conversations by:
# MAGIC 1. Logging each conversation session as a trace
# MAGIC 2. Writing code-based `@scorer` functions that analyze the full conversation
# MAGIC 3. Running `mlflow.genai.evaluate()` with pre-collected conversation data
# MAGIC
# MAGIC > See the **Databricks notebook** for a full multi-turn simulation and evaluation demo.

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
# MAGIC | Capability | What to Track | OSS Tooling |
# MAGIC |------------|---------------|-------------|
# MAGIC | **Quality dashboards** | Automated eval scores aggregated hourly/daily | MLflow metrics + Grafana dashboards |
# MAGIC | **Trace sampling** | Random sample of production traces for manual review | MLflow Traces + cron job scripts |
# MAGIC | **Latency monitoring** | p50/p95/p99 latency per span, overall request latency | Prometheus + MLflow metrics |
# MAGIC | **Cost tracking** | Token usage, model calls, and $ cost per request | MLflow metrics + custom aggregation |
# MAGIC | **Drift detection** | Input distribution shifts, topic drift, quality degradation | Custom scripts + MLflow comparisons |
# MAGIC | **Alerting** | Threshold-based or anomaly-based alerts on any metric | Slack webhooks + cron-based monitoring |
# MAGIC
# MAGIC **Architecture pattern:**
# MAGIC ```
# MAGIC Agent Request -> MLflow Trace -> PostgreSQL -> Grafana Dashboard
# MAGIC                                             -> Cron Eval Job
# MAGIC                                             -> Slack Alert
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
    print("\n  Check the MLflow Experiment UI -> Metrics tab -> prod/* namespace")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pillar 5: Governance & Lifecycle
# MAGIC *Ship responsibly*
# MAGIC
# MAGIC | Practice | Description | OSS Tooling |
# MAGIC |----------|-------------|-------------|
# MAGIC | **Model Registry** | Version every model, prompt template, and retrieval config | MLflow Model Registry |
# MAGIC | **Eval gates in CI/CD** | Run `mlflow.genai.evaluate()` on every PR that touches agent behavior | GitHub Actions + MLflow Python API |
# MAGIC | **Deployment stages** | Promote through dev -> staging -> prod with full lineage | MLflow model aliases + tags |
# MAGIC | **Audit trails** | Record who changed what, when, and why | MLflow tags + git integration |
# MAGIC | **Access control** | Team-level experiment permissions | MLflow Auth plugin (basic auth) |
# MAGIC | **Data governance** | Govern traces containing sensitive information | Custom filtering + access policies |
# MAGIC
# MAGIC **CI/CD Evaluation Gate (pseudocode):**
# MAGIC ```python
# MAGIC # In your CI/CD pipeline (GitHub Actions, Jenkins, etc.):
# MAGIC from mlflow.genai.scorers import scorer
# MAGIC from mlflow.entities import Feedback
# MAGIC
# MAGIC @scorer
# MAGIC def quality_check(inputs, outputs) -> Feedback:
# MAGIC     # Your deterministic quality checks here
# MAGIC     ...
# MAGIC
# MAGIC results = mlflow.genai.evaluate(
# MAGIC     data=golden_dataset,
# MAGIC     predict_fn=new_agent,
# MAGIC     scorers=[quality_check],
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
    print(eval_traces[["trace_id", "request", "response", "assessments"]].head().to_string())

    print("\nEval gate check: Review assessments (scorer feedback) on each trace.")
    print("If any key scorer fails, block the deployment.")
else:
    print("No evaluation traces found yet. Run the evaluation cells above first!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pillar 6: Prompt Optimization
# MAGIC ---
# MAGIC
# MAGIC We registered our prompt in the Prompt Registry at the start of this notebook. Now we can systematically improve it.
# MAGIC
# MAGIC ### Automated Prompt Optimization (Databricks Only)
# MAGIC
# MAGIC `mlflow.genai.optimize_prompts()` automates the prompt improvement loop. It is currently available only in **Managed MLflow on Databricks**, with open source support coming soon.
# MAGIC
# MAGIC | Optimizer | How It Works | Best For |
# MAGIC |-----------|-------------|----------|
# MAGIC | **GepaPromptOptimizer** | Genetic-Pareto optimization with natural language reflection. Up to 35x fewer iterations than brute force. | High-stakes tasks with substantial eval datasets |
# MAGIC | **MetaPromptOptimizer** | Zero-shot or few-shot metaprompting. Fast, low cost. | Quick improvements, small datasets or no data at all |
# MAGIC
# MAGIC > See the **Databricks notebook** for a full prompt optimization demo with `GepaPromptOptimizer` and `Correctness()`.
# MAGIC
# MAGIC > **Docs:** [mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts](https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/)
# MAGIC
# MAGIC ### Manual Prompt Iteration (OSS)
# MAGIC
# MAGIC In OSS MLflow, you can iterate on prompts manually using the Prompt Registry + evaluation loop:
# MAGIC
# MAGIC 1. **Register** a prompt version with `mlflow.genai.register_prompt()`
# MAGIC 2. **Evaluate** using `mlflow.genai.evaluate()` with code-based `@scorer` functions
# MAGIC 3. **Analyze** results, update the prompt template, and register a new version
# MAGIC 4. **Compare** evaluation runs across prompt versions in the MLflow UI

# COMMAND ----------

# ── Manual prompt iteration demo ─────────────────────────────
# Register an improved prompt version and compare evaluation results

improved_prompt = mlflow.genai.register_prompt(
    name=support_prompt.name,
    template=(
        "You are a knowledgeable and friendly customer support agent. "
        "Answer the customer's question using ONLY the provided context. "
        "Be specific — include numbers, dates, and concrete details from the context. "
        "If the context doesn't contain the answer, say 'I don't have that information' "
        "and suggest contacting support.\n\n"
        "Context:\n{{context}}\n\n"
        "Question: {{question}}"
    ),
)

print(f"Improved prompt registered: {improved_prompt.name}")
print(f"  Version: {improved_prompt.version} (was {support_prompt.version})")
print(f"  URI: {improved_prompt.uri}")

# COMMAND ----------

from openai import OpenAI

# ── Compare prompt versions with evaluation ──────────────────
@mlflow.trace
def run_agent_v2(question: str) -> dict:
    """Agent using the improved prompt version."""
    context = retrieve_context(question)
    prompt = mlflow.genai.load_prompt(improved_prompt.uri)
    system_message = prompt.format(question=question, context=context)

    # TO RUN OUTSIDE DATABRICKS: use OpenAI(api_key=os.environ["OPENAI_API_KEY"]) and model="gpt-4o-mini"
    client = OpenAI(
        api_key=os.environ.get("DATABRICKS_TOKEN", dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()),
        base_url=f"{mlflow.utils.databricks_utils.get_workspace_url()}/serving-endpoints"
    )
    response = client.chat.completions.create(
        model="databricks-claude-sonnet-4",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ],
        max_tokens=300,
        temperature=0.1,
    )
    return {"response": response.choices[0].message.content}

# Evaluate improved prompt
with mlflow.start_run(run_name="prompt_v2"):
    v2_results = mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=run_agent_v2,
        scorers=[safety_check, answer_relevance, helpfulness_check, response_length_check],
    )

print("\n=== Prompt Version Comparison ===")
print(f"  V1 Run ID: {results.run_id}")
print(f"  V2 Run ID: {v2_results.run_id}")
print(f"\nCompare runs in the MLflow Experiment UI to see which prompt performs better.")

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
# MAGIC | 7 | **OSS-to-Databricks migration is seamless** | When you outgrow self-hosted MLflow, switch `set_tracking_uri("databricks")` and gain Unity Catalog, Lakehouse Monitoring, and managed infra -- same code, same API. |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC ---
# MAGIC
# MAGIC | Pillar | What We Covered | Key MLflow API |
# MAGIC |--------|----------------|----------------|
# MAGIC | **1. Tracing & Observability** | Instrumented an agent with nested spans for retrieval and generation | `@mlflow.trace`, `mlflow.start_span()` |
# MAGIC | **2. Evaluation** | Ran automated evaluation with code-based `@scorer` functions | `mlflow.genai.evaluate()`, `@scorer`, `Feedback` |
# MAGIC | **3. Human Feedback** | Simulated human feedback collection and computed judge agreement rates | `mlflow.log_metric()`, correlation analysis |
# MAGIC | **4. Production Monitoring** | Logged production-style metrics (latency, quality, cost) at scale | `mlflow.log_metrics()`, tags, namespaced metrics |
# MAGIC | **5. Governance & Lifecycle** | Reviewed trace assessments and simulated CI/CD eval gates | `mlflow.search_traces()`, assessments |
# MAGIC | **6. Prompt Optimization** | Registered prompt versions and compared evaluation results | `mlflow.genai.register_prompt()`, `mlflow.genai.evaluate()` |
# MAGIC
# MAGIC ### Next Steps
# MAGIC - **Expand your golden dataset** with real production examples
# MAGIC - **Set up Grafana dashboards** for continuous quality tracking
# MAGIC - **Integrate eval gates** into your GitHub Actions / CI pipeline
# MAGIC - **Consider Databricks MLflow** when you need managed infra, Unity Catalog, and enterprise RBAC
# MAGIC
# MAGIC ### Resources
# MAGIC - [MLflow Tracing Docs](https://mlflow.org/docs/latest/tracing.html)
# MAGIC - [MLflow GenAI Evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/)
# MAGIC - [MLflow Prompt Optimization](https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/)
# MAGIC - [MLflow Auth Plugin](https://mlflow.org/docs/latest/auth/index.html)

# COMMAND ----------

# ── Cleanup & experiment URL ────────────────────────────────
experiment = mlflow.get_experiment_by_name("/Users/stephanie.rivera@databricks.com/WWC - AgentOps/AgentOps with MLflow (OSS MLflow)")
if experiment:
    tracking_uri = mlflow.get_tracking_uri()
    if tracking_uri == "databricks":
        workspace_url = mlflow.utils.databricks_utils.get_workspace_url()
        print(f"Experiment URL: {workspace_url}/ml/experiments/{experiment.experiment_id}")
    else:
        print(f"Experiment URL: {tracking_uri}/#/experiments/{experiment.experiment_id}")
    print(f"Experiment ID: {experiment.experiment_id}")
else:
    print("Experiment not found. It will be created when you run the configuration cell.")