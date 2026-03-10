# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2: Deploy Sub-Agent to Model Serving
# MAGIC
# MAGIC 1. Write the sub-agent as a ResponsesAgent
# MAGIC 2. Log with `DatabricksLakebase` resource for automatic auth passthrough
# MAGIC 3. Deploy via `agents.deploy()`
# MAGIC 4. Test invoke → interrupt → resume → complete

# COMMAND ----------

# MAGIC %pip install langgraph==1.0.10 langgraph-checkpoint-postgres==3.0.4 databricks-langchain[memory]==0.17.0 psycopg[binary]==3.3.3 psycopg-pool==3.3.0 mlflow==3.9.0 databricks-agents==1.9.3 databricks-sdk openai
# MAGIC %restart_python

# COMMAND ----------

import os
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

catalog = "brickcon_landparcels_classic_catalog"
schema = "coa_interrupt_poc"

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
print(f"Model registry: {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Write the Sub-Agent Model Code

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC import uuid
# MAGIC import json
# MAGIC from typing import TypedDict
# MAGIC
# MAGIC import mlflow
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
# MAGIC
# MAGIC from databricks_langchain import CheckpointSaver
# MAGIC from langgraph.graph import StateGraph, START, END
# MAGIC from langgraph.types import interrupt, Command
# MAGIC
# MAGIC
# MAGIC LAKEBASE_INSTANCE_NAME = "coa-checkpoint"
# MAGIC
# MAGIC
# MAGIC class SubAgentState(TypedDict):
# MAGIC     task: str
# MAGIC     clarification: str
# MAGIC     result: str
# MAGIC
# MAGIC
# MAGIC def sub_agent_work(state: SubAgentState) -> dict:
# MAGIC     clarification = interrupt({
# MAGIC         "question": f"To complete '{state['task']}', which format do you prefer?",
# MAGIC         "options": ["detailed", "summary", "bullet points"]
# MAGIC     })
# MAGIC     return {
# MAGIC         "clarification": clarification,
# MAGIC         "result": f"Completed '{state['task']}' in {clarification} format"
# MAGIC     }
# MAGIC
# MAGIC
# MAGIC def build_graph():
# MAGIC     builder = StateGraph(SubAgentState)
# MAGIC     builder.add_node("work", sub_agent_work)
# MAGIC     builder.add_edge(START, "work")
# MAGIC     builder.add_edge("work", END)
# MAGIC     return builder
# MAGIC
# MAGIC
# MAGIC class SubAgentModel(ResponsesAgent):
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         custom_inputs = dict(request.custom_inputs or {})
# MAGIC         thread_id = custom_inputs.get("thread_id", f"sub-{uuid.uuid4().hex[:8]}")
# MAGIC
# MAGIC         checkpointer = CheckpointSaver(instance_name=LAKEBASE_INSTANCE_NAME)
# MAGIC         graph = build_graph().compile(checkpointer=checkpointer)
# MAGIC         config = {"configurable": {"thread_id": thread_id}}
# MAGIC
# MAGIC         # Extract text from input messages
# MAGIC         input_text = "unknown"
# MAGIC         for item in (request.input or []):
# MAGIC             if hasattr(item, "content") and isinstance(item.content, str):
# MAGIC                 input_text = item.content
# MAGIC                 break
# MAGIC             elif isinstance(item, dict) and "content" in item:
# MAGIC                 content = item["content"]
# MAGIC                 if isinstance(content, str):
# MAGIC                     input_text = content
# MAGIC                     break
# MAGIC                 elif isinstance(content, list):
# MAGIC                     for part in content:
# MAGIC                         if isinstance(part, dict) and part.get("type") == "input_text":
# MAGIC                             input_text = part["text"]
# MAGIC                             break
# MAGIC
# MAGIC         if "thread_id" in custom_inputs:
# MAGIC             # Resume: input text is the resume value
# MAGIC             result = graph.invoke(Command(resume=input_text), config)
# MAGIC         else:
# MAGIC             # New invocation
# MAGIC             result = graph.invoke(
# MAGIC                 {"task": input_text, "clarification": "", "result": ""},
# MAGIC                 config
# MAGIC             )
# MAGIC
# MAGIC         snapshot = graph.get_state(config)
# MAGIC         if snapshot.next:
# MAGIC             interrupt_value = None
# MAGIC             if snapshot.tasks:
# MAGIC                 for t in snapshot.tasks:
# MAGIC                     if t.interrupts:
# MAGIC                         interrupt_value = t.interrupts[0].value
# MAGIC                         break
# MAGIC
# MAGIC             return ResponsesAgentResponse(
# MAGIC                 output=[{
# MAGIC                     "type": "message",
# MAGIC                     "id": f"msg-{uuid.uuid4().hex[:12]}",
# MAGIC                     "role": "assistant",
# MAGIC                     "content": [{"type": "output_text", "text": json.dumps(interrupt_value) if interrupt_value else "Interrupted"}],
# MAGIC                 }],
# MAGIC                 custom_outputs={
# MAGIC                     "status": "interrupted",
# MAGIC                     "thread_id": thread_id,
# MAGIC                     "interrupt": interrupt_value,
# MAGIC                     "awaiting_input": True,
# MAGIC                 },
# MAGIC             )
# MAGIC
# MAGIC         return ResponsesAgentResponse(
# MAGIC             output=[{
# MAGIC                 "type": "message",
# MAGIC                 "id": f"msg-{uuid.uuid4().hex[:12]}",
# MAGIC                 "role": "assistant",
# MAGIC                 "content": [{"type": "output_text", "text": result.get("result", "Done")}],
# MAGIC             }],
# MAGIC             custom_outputs={
# MAGIC                 "status": "complete",
# MAGIC                 "thread_id": thread_id,
# MAGIC                 "awaiting_input": False,
# MAGIC             },
# MAGIC         )
# MAGIC
# MAGIC
# MAGIC mlflow.models.set_model(SubAgentModel())

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Local Agent Validation

# COMMAND ----------

from agent import SubAgentModel
import json

AGENT = SubAgentModel()

print("--- Local validation: invoke (should interrupt) ---")
result1 = AGENT.predict({
    "input": [{"role": "user", "content": "analyze Q4 revenue"}],
})
print(json.dumps(result1.model_dump(exclude_none=True), indent=2))

custom1 = result1.custom_outputs or {}
assert custom1.get("status") == "interrupted", f"Expected interrupted, got: {custom1}"
thread_id = custom1["thread_id"]
print(f"\n✓ Sub-agent interrupted, thread_id: {thread_id}")

# COMMAND ----------

print("--- Local validation: resume (should complete) ---")
result2 = AGENT.predict({
    "input": [{"role": "user", "content": "bullet points"}],
    "custom_inputs": {"thread_id": thread_id},
})
print(json.dumps(result2.model_dump(exclude_none=True), indent=2))

custom2 = result2.custom_outputs or {}
assert custom2.get("status") == "complete", f"Expected complete, got: {custom2}"
print(f"\n✓ Sub-agent completed locally!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Log and Register Model

# COMMAND ----------

import mlflow
import time
from mlflow.models.resources import DatabricksLakebase
from databricks.sdk import WorkspaceClient
from agent import LAKEBASE_INSTANCE_NAME

w = WorkspaceClient()
catalog = "brickcon_landparcels_classic_catalog"
schema = "coa_interrupt_poc"

resources = [
    DatabricksLakebase(database_instance_name=LAKEBASE_INSTANCE_NAME),
]

input_example = {
    "input": [{"role": "user", "content": "analyze Q4 revenue"}],
}

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{schema}.coa_sub_agent"

with mlflow.start_run(run_name="coa-sub-agent"):
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        input_example=input_example,
        pip_requirements=[
            "langgraph==1.0.10",
            "langgraph-checkpoint-postgres==3.0.4",
            "databricks-langchain[memory]==0.17.0",
            "psycopg[binary]==3.3.3",
            "psycopg-pool==3.3.0",
            "databricks-sdk",
            "mlflow==3.9.0",
        ],
        resources=resources,
        registered_model_name=model_name,
    )
    print(f"Model logged: {logged_agent_info.model_uri}")
    print(f"Registered as: {model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Pre-Deployment Validation with mlflow.models.predict

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"input": [{"role": "user", "content": "analyze Q4 revenue"}]},
    env_manager="uv",
)
print("✓ Pre-deployment validation passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Deploy to Model Serving Endpoint

# COMMAND ----------

from databricks import agents

endpoint_name = "coa-sub-agent"

# Wait for any in-progress deployment to finish
try:
    ep = w.serving_endpoints.get(endpoint_name)
    if ep.state and ep.state.config_update and "IN_PROGRESS" in str(ep.state.config_update):
        print("Endpoint has in-progress deployment, waiting...")
        while True:
            time.sleep(30)
            ep = w.serving_endpoints.get(endpoint_name)
            update_state = str(ep.state.config_update) if ep.state and ep.state.config_update else "NONE"
            print(f"  Config update: {update_state}")
            if "IN_PROGRESS" not in update_state:
                break
except Exception as e:
    print(f"Endpoint may not exist yet: {e}")

versions = list(w.model_versions.list(full_name=model_name))
latest_version = max(v.version for v in versions)
print(f"Deploying {model_name} version {latest_version} via agents framework")

deployment = agents.deploy(
    model_name=model_name,
    model_version=latest_version,
    endpoint_name=endpoint_name,
    scale_to_zero=True,
)
print(f"✓ Endpoint '{endpoint_name}' deployment initiated")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Wait for Endpoint to be Ready

# COMMAND ----------

print("Waiting for endpoint to be ready...")
while True:
    time.sleep(30)
    ep = w.serving_endpoints.get(endpoint_name)
    ready = str(ep.state.ready) if ep.state and ep.state.ready else "UNKNOWN"
    update = str(ep.state.config_update) if ep.state and ep.state.config_update else "NONE"
    print(f"  Ready: {ready}, Config update: {update}")
    if "READY" in ready and "IN_PROGRESS" not in update:
        break
print(f"✓ Endpoint '{endpoint_name}' is ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test Sub-Agent Endpoint

# COMMAND ----------

client = w.serving_endpoints.get_open_ai_client()

print("=" * 60)
print("TEST: Sub-Agent Endpoint")
print("=" * 60)

print("\n--- Test 1: Invoke ---")
invoke_response = client.responses.create(
    model="coa-sub-agent",
    input=[{"type": "message", "role": "user", "content": "analyze Q4 revenue"}],
)
print(f"Response: {invoke_response}")

custom_out = invoke_response.custom_outputs or {}
assert custom_out.get("status") == "interrupted", f"Expected interrupted, got {custom_out}"
sub_thread_id = custom_out["thread_id"]
print(f"\n✓ Sub-agent interrupted, thread_id: {sub_thread_id}")

print("\n--- Test 2: Resume ---")
resume_response = client.responses.create(
    model="coa-sub-agent",
    input=[{"type": "message", "role": "user", "content": "bullet points"}],
    extra_body={"custom_inputs": {"thread_id": sub_thread_id}},
)
print(f"Response: {resume_response}")

custom_out2 = resume_response.custom_outputs or {}
assert custom_out2.get("status") == "complete", f"Expected complete, got {custom_out2}"
print(f"\n✓ Sub-agent completed!")

# COMMAND ----------

print(f"""
========================================
Phase 2 COMPLETE
========================================
✓ Sub-agent logged as MLflow ResponsesAgent model
✓ Registered in Unity Catalog
✓ Deployed to Model Serving endpoint
✓ Invoke → interrupt works
✓ Resume → complete works
========================================

Endpoint: coa-sub-agent
Model: {model_name}
========================================
""")
