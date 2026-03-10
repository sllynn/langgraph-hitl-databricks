# Multi-Agent Interrupt POC

Validates multi-agent LangGraph architecture with human-in-the-loop interrupts on Databricks Model Serving.

## Architecture

```
Caller
  |
  v
coa-supervisor (Model Serving endpoint)
  |
  v
coa-sub-agent (Model Serving endpoint)
  |
  v
Lakebase (shared checkpoint store)
```

- **Supervisor** orchestrates sub-agents, each running on separate Model Serving endpoints
- **Sub-agents** can raise interrupts (human-in-the-loop) which propagate back through the supervisor to the caller
- **All graph state** is checkpointed in Lakebase — the supervisor's state includes which sub-agent raised the interrupt and the sub-agent's thread ID
- **On resume**, the caller only passes back the supervisor's `thread_id` — the supervisor rehydrates from Lakebase and knows how to resume the correct sub-agent conversation

## Protocol

**Invoke** — no `custom_inputs` needed:
```python
response = client.responses.create(
    model="coa-supervisor",
    input=[{"type": "message", "role": "user", "content": "analyze Q4 revenue"}],
)
# response.custom_outputs => {"status": "interrupted", "thread_id": "sup-xxx", "interrupt": {...}, "sub_agent": "coa-sub-agent"}
```

**Resume** — pass `thread_id`, input text is the user's answer:
```python
response = client.responses.create(
    model="coa-supervisor",
    input=[{"type": "message", "role": "user", "content": "summary"}],
    extra_body={"custom_inputs": {"thread_id": "sup-xxx"}},
)
# response.custom_outputs => {"status": "complete", "thread_id": "sup-xxx"}
```

## Prerequisites

- Databricks workspace with **Model Serving** and **Lakebase** enabled
- Unity Catalog with a catalog and schema for model registration
- Databricks CLI authenticated to the workspace

## Deploy to a New Workspace

### 1. Configure

Edit `databricks.yml` — set your workspace URL and CLI profile:
```yaml
workspace:
  host: https://YOUR-WORKSPACE.cloud.databricks.com
targets:
  dev:
    default: true
    workspace:
      profile: YOUR-PROFILE
```
DABs will automatically deploy to `/Workspace/Users/<your-email>/.bundle/coa-interrupt-poc/dev`.

Edit the catalog/schema in both notebooks (`05_deploy_sub_agent.py` and `06_deploy_supervisor.py`):
```python
catalog = "YOUR_CATALOG"
schema = "YOUR_SCHEMA"
```

### 2. Create the Lakebase instance

Run `notebooks/01_setup_lakebase.py` to create the `coa-checkpoint` Lakebase instance, or create it manually:
```bash
databricks database create-instance coa-checkpoint \
  --capacity CU_1 --profile YOUR-PROFILE
```

### 3. Deploy and run

```bash
# Authenticate
databricks auth login --host https://YOUR-WORKSPACE.cloud.databricks.com --profile YOUR-PROFILE

# Deploy bundle
databricks bundle deploy --profile YOUR-PROFILE

# Run sub-agent first (supervisor depends on it)
databricks bundle run deploy_sub_agent --profile YOUR-PROFILE

# Run supervisor after sub-agent endpoint is READY
databricks bundle run deploy_supervisor --profile YOUR-PROFILE
```

Each notebook will: write agent code, validate locally, log to MLflow, register in Unity Catalog, deploy to Model Serving, wait for READY, and run end-to-end tests.

### 4. Test from any client

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
client = w.serving_endpoints.get_open_ai_client()

# Invoke
r1 = client.responses.create(
    model="coa-supervisor",
    input=[{"type": "message", "role": "user", "content": "analyze Q4 revenue"}],
)
print(r1.custom_outputs)  # status=interrupted, thread_id, interrupt payload, sub_agent

# Resume with user's answer
r2 = client.responses.create(
    model="coa-supervisor",
    input=[{"type": "message", "role": "user", "content": "summary"}],
    extra_body={"custom_inputs": {"thread_id": r1.custom_outputs["thread_id"]}},
)
print(r2.custom_outputs)  # status=complete
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_setup_lakebase.py` | Create Lakebase instance for checkpointing |
| `02_test_single_agent.py` | Test a single LangGraph agent with interrupt locally |
| `03_test_supervisor_local.py` | Test supervisor + sub-agent locally (no endpoints) |
| `04_test_protocol_local.py` | Test the full interrupt/resume protocol locally |
| `05_deploy_sub_agent.py` | Deploy sub-agent to Model Serving |
| `06_deploy_supervisor.py` | Deploy supervisor to Model Serving + end-to-end tests |

## Key Dependencies

| Package | Version | Why |
|---------|---------|-----|
| `mlflow` | `3.9.0` | Pinned — 3.10.x has a [deadlock issue](https://community.databricks.com/t5/generative-ai/model-serving-endpoints-scaling-from-zero-forever/td-p/150014) |
| `langgraph` | `1.0.10` | `interrupt()` + `Command(resume=...)` pattern |
| `databricks-langchain[memory]` | `0.17.0` | `CheckpointSaver` for Lakebase |
| `databricks-agents` | `1.9.3` | `agents.deploy()` for Model Serving |

## Known Issues

- **mlflow 3.10.x deadlock**: Endpoints get stuck "scaling from zero forever". Pin to `mlflow==3.9.0`.
- **`agents.deploy()` accumulates served entities**: Each call adds a new entity rather than replacing. If redeploying, delete the endpoint first via the UI or SDK to start clean.
- **Databricks SDK enum strings**: `str(ep.state.ready)` returns `"EndpointStateReady.READY"`, not `"READY"`. Use `"READY" in str(...)` for comparisons.
