# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

POC for customer **Arm** validating multi-agent LangGraph architecture with **human-in-the-loop interrupts** on Databricks Model Serving. Two separate Model Serving endpoints (supervisor + sub-agent) communicate via HTTP, with interrupt state checkpointed to **Lakebase Provisioned** (Postgres).

See `PLAN.md` for the full architecture diagram, test plan, and design rationale.

## Key Architecture: "Interrupts All The Way Up"

Sub-agent hits `interrupt()` → checkpoints to Lakebase → returns `custom_outputs: {status: "interrupted", thread_id, interrupt}` → Supervisor stores sub-agent thread_id in its own state → calls `interrupt()` itself → Caller only knows supervisor's thread_id → On resume, flows back down through both graphs.

## Deployment Commands

```bash
# Authenticate (profile: brickcon)
databricks auth login --host https://fevm-brickcon-landparcels-classic.cloud.databricks.com --profile brickcon

# Deploy notebooks to workspace
databricks bundle deploy --profile brickcon

# Run a specific job (deploy sub-agent first, then supervisor)
databricks bundle run deploy_sub_agent --profile brickcon
databricks bundle run deploy_supervisor --profile brickcon

# Check endpoint status
databricks serving-endpoints get coa-sub-agent --profile brickcon
databricks serving-endpoints get coa-supervisor --profile brickcon
```

## Notebook Conventions

All notebooks are **Databricks source format** (`# MAGIC` prefixed lines, `# COMMAND ----------` separators). They run as serverless jobs via DABs — no cluster definition needed in `databricks.yml`.

### Writing agent code in notebooks

Use `%%writefile agent.py` cell magic to write agent code to a file, then import from it. **Never** use `dbutils.fs.put()` or string variables to create agent files.

```python
# MAGIC %%writefile agent.py
# MAGIC import mlflow
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC ...
# MAGIC LAKEBASE_INSTANCE_NAME = "coa-checkpoint"
# MAGIC ...
# MAGIC AGENT = SupervisorModel()
# MAGIC mlflow.models.set_model(AGENT)
```

Then in the logging cell, **import constants from agent.py** to build the resources list (follow the documented pattern):

```python
from agent import LAKEBASE_INSTANCE_NAME, SUB_AGENT_ENDPOINT
from mlflow.models.resources import DatabricksLakebase, DatabricksServingEndpoint

resources = [
    DatabricksLakebase(database_instance_name=LAKEBASE_INSTANCE_NAME),
    DatabricksServingEndpoint(endpoint_name=SUB_AGENT_ENDPOINT),
]
```

### Model logging pattern

Follow the reference notebook at https://docs.databricks.com/aws/en/notebooks/source/generative-ai/short-term-memory-agent-lakebase.html exactly:

1. `%%writefile agent.py` — write agent code, export an `AGENT` instance
2. **Local validation** — call `AGENT.predict()` directly to verify the agent works
3. **Log model** — `mlflow.pyfunc.log_model(name="agent", python_model="agent.py", input_example=..., resources=..., registered_model_name=...)`
4. **Pre-deployment validation** — `mlflow.models.predict(model_uri=..., env_manager="uv")` (skip if agent requires live endpoint access like the supervisor)
5. **Deploy** — `agents.deploy(model_name=..., model_version=..., endpoint_name=..., scale_to_zero=True)`

### Resource declarations are critical

`DatabricksLakebase` in the `resources` parameter of `log_model()` enables **automatic authentication passthrough** for Model Serving endpoints connecting to Lakebase. Without it, the serving endpoint's service principal won't have Lakebase access.

## Infrastructure Details

| Resource | Value |
|----------|-------|
| Workspace | `https://fevm-brickcon-landparcels-classic.cloud.databricks.com` |
| CLI Profile | `brickcon` |
| Catalog | `brickcon_landparcels_classic_catalog` |
| Schema | `coa_interrupt_poc` |
| Lakebase Instance | `coa-checkpoint` (Provisioned tier, NOT Autoscaling) |
| Lakebase Database | `coa_checkpoints` |
| Sub-agent endpoint | `coa-sub-agent` |
| Supervisor endpoint | `coa-supervisor` |
| Sub-agent model | `brickcon_landparcels_classic_catalog.coa_interrupt_poc.coa_sub_agent` |
| Supervisor model | `brickcon_landparcels_classic_catalog.coa_interrupt_poc.coa_supervisor` |

## Key Libraries and Patterns

- **`databricks_langchain.CheckpointSaver`** — sync Lakebase checkpoint saver for use in Model Serving. Takes `instance_name` parameter. Handles auth automatically when `DatabricksLakebase` resource is declared.
- **`langgraph.types.interrupt` + `Command(resume=...)`** — LangGraph v0.2.57+ human-in-the-loop pattern
- **`mlflow.pyfunc.ResponsesAgent`** — agent wrapper compatible with OpenAI Responses API
- **`custom_inputs`/`custom_outputs`** — pass `action`, `thread_id`, `resume_value` through the serving endpoint protocol

## Deployment Order

Sub-agent **must** be deployed and READY before supervisor, because:
1. Supervisor's `DatabricksServingEndpoint` resource declaration causes a pre-deployment check for the sub-agent endpoint
2. Supervisor's local validation calls the sub-agent endpoint via HTTP

## Notebook Execution Order

| Phase | Notebook | Purpose |
|-------|----------|---------|
| 0 | `01_setup_lakebase.py` | Create Lakebase database, verify connectivity |
| 1a | `02_test_single_agent.py` | Single agent interrupt + checkpoint |
| 1b | `03_test_supervisor_local.py` | In-process subgraph interrupt propagation |
| 1c | `04_test_protocol_local.py` | Simulate cross-endpoint protocol locally |
| 2 | `05_deploy_sub_agent.py` | Deploy sub-agent to Model Serving |
| 3 | `06_deploy_supervisor.py` | Deploy supervisor to Model Serving |
