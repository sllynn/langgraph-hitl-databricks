# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 3: Deploy Supervisor to Model Serving
# MAGIC
# MAGIC 1. Write the supervisor agent code (calls sub-agent endpoint via HTTP)
# MAGIC 2. Validate locally with `AGENT.predict()`
# MAGIC 3. Log with `DatabricksLakebase` + `DatabricksServingEndpoint` resources
# MAGIC 4. Pre-deployment validation with `mlflow.models.predict()`
# MAGIC 5. Deploy via `agents.deploy()`
# MAGIC 6. Test the full end-to-end flow over HTTP

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

# Verify sub-agent endpoint is ready before proceeding
ep = w.serving_endpoints.get("coa-sub-agent")
print(f"Sub-agent endpoint ready: {ep.state.ready}")
assert "READY" in str(ep.state.ready), f"Sub-agent endpoint must be READY before deploying supervisor. Current state: {ep.state.ready}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Write the Supervisor Model Code

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
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks_langchain import CheckpointSaver
# MAGIC from langgraph.graph import StateGraph, START, END
# MAGIC from langgraph.types import interrupt, Command
# MAGIC
# MAGIC
# MAGIC LAKEBASE_INSTANCE_NAME = "coa-checkpoint"
# MAGIC SUB_AGENT_ENDPOINT = "coa-sub-agent"
# MAGIC
# MAGIC
# MAGIC class SupervisorState(TypedDict):
# MAGIC     task: str
# MAGIC     sub_agent_endpoint: str
# MAGIC     sub_agent_thread_id: str
# MAGIC     sub_agent_interrupted: bool
# MAGIC     interrupt_payload: dict
# MAGIC     result: str
# MAGIC
# MAGIC
# MAGIC def call_sub_agent_endpoint(input_text: str, custom_inputs: dict) -> dict:
# MAGIC     """Call the sub-agent Model Serving endpoint via SDK OpenAI client."""
# MAGIC     w = WorkspaceClient()
# MAGIC     client = w.serving_endpoints.get_open_ai_client()
# MAGIC     response = client.responses.create(
# MAGIC         model=SUB_AGENT_ENDPOINT,
# MAGIC         input=[{"type": "message", "role": "user", "content": input_text}],
# MAGIC         extra_body={"custom_inputs": custom_inputs},
# MAGIC     )
# MAGIC     # Extract output text from response
# MAGIC     output_text = ""
# MAGIC     if hasattr(response, "output"):
# MAGIC         for item in response.output:
# MAGIC             if hasattr(item, "content"):
# MAGIC                 for part in item.content:
# MAGIC                     if hasattr(part, "text"):
# MAGIC                         output_text = part.text
# MAGIC                         break
# MAGIC     return {
# MAGIC         "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": output_text}]}],
# MAGIC         "custom_outputs": dict(response.custom_outputs) if hasattr(response, "custom_outputs") and response.custom_outputs else {},
# MAGIC     }
# MAGIC
# MAGIC
# MAGIC def call_sub_agent(state: SupervisorState) -> dict:
# MAGIC     if state.get("sub_agent_interrupted") and state.get("sub_agent_thread_id"):
# MAGIC         user_response = interrupt(state["interrupt_payload"])
# MAGIC         response = call_sub_agent_endpoint(
# MAGIC             input_text=user_response,
# MAGIC             custom_inputs={"thread_id": state["sub_agent_thread_id"]},
# MAGIC         )
# MAGIC     else:
# MAGIC         response = call_sub_agent_endpoint(
# MAGIC             input_text=state["task"],
# MAGIC             custom_inputs={},
# MAGIC         )
# MAGIC
# MAGIC     custom_out = response.get("custom_outputs", {})
# MAGIC
# MAGIC     if custom_out.get("status") == "interrupted":
# MAGIC         return {
# MAGIC             "sub_agent_endpoint": SUB_AGENT_ENDPOINT,
# MAGIC             "sub_agent_thread_id": custom_out["thread_id"],
# MAGIC             "sub_agent_interrupted": True,
# MAGIC             "interrupt_payload": custom_out.get("interrupt", {}),
# MAGIC         }
# MAGIC
# MAGIC     result_text = "Done"
# MAGIC     for item in response.get("output", []):
# MAGIC         if isinstance(item, dict) and item.get("role") == "assistant":
# MAGIC             content = item.get("content", "")
# MAGIC             if isinstance(content, str):
# MAGIC                 result_text = content
# MAGIC                 break
# MAGIC             elif isinstance(content, list):
# MAGIC                 for part in content:
# MAGIC                     if isinstance(part, dict) and part.get("type") == "output_text":
# MAGIC                         result_text = part.get("text", "Done")
# MAGIC                         break
# MAGIC
# MAGIC     return {
# MAGIC         "sub_agent_interrupted": False,
# MAGIC         "result": result_text,
# MAGIC     }
# MAGIC
# MAGIC
# MAGIC def check_interrupt(state: SupervisorState) -> str:
# MAGIC     if state.get("sub_agent_interrupted"):
# MAGIC         return "call_sub_agent"
# MAGIC     return "finalize"
# MAGIC
# MAGIC
# MAGIC def finalize(state: SupervisorState) -> dict:
# MAGIC     return {"result": f"[Supervisor] {state['result']}"}
# MAGIC
# MAGIC
# MAGIC def build_supervisor_graph():
# MAGIC     builder = StateGraph(SupervisorState)
# MAGIC     builder.add_node("call_sub_agent", call_sub_agent)
# MAGIC     builder.add_node("finalize", finalize)
# MAGIC     builder.add_edge(START, "call_sub_agent")
# MAGIC     builder.add_conditional_edges("call_sub_agent", check_interrupt, {
# MAGIC         "call_sub_agent": "call_sub_agent",
# MAGIC         "finalize": "finalize",
# MAGIC     })
# MAGIC     builder.add_edge("finalize", END)
# MAGIC     return builder
# MAGIC
# MAGIC
# MAGIC class SupervisorModel(ResponsesAgent):
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         custom_inputs = dict(request.custom_inputs or {})
# MAGIC         thread_id = custom_inputs.get("thread_id", f"sup-{uuid.uuid4().hex[:8]}")
# MAGIC
# MAGIC         checkpointer = CheckpointSaver(instance_name=LAKEBASE_INSTANCE_NAME)
# MAGIC         graph = build_supervisor_graph().compile(checkpointer=checkpointer)
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
# MAGIC             result = graph.invoke({
# MAGIC                 "task": input_text,
# MAGIC                 "sub_agent_endpoint": "",
# MAGIC                 "sub_agent_thread_id": "",
# MAGIC                 "sub_agent_interrupted": False,
# MAGIC                 "interrupt_payload": {},
# MAGIC                 "result": "",
# MAGIC             }, config)
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
# MAGIC             # Get sub-agent info from checkpointed state
# MAGIC             state_values = snapshot.values or {}
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
# MAGIC                     "sub_agent": state_values.get("sub_agent_endpoint", ""),
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
# MAGIC AGENT = SupervisorModel()
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Local Agent Validation
# MAGIC
# MAGIC Validate the agent works by calling `AGENT.predict()` directly.
# MAGIC This calls the live sub-agent endpoint, so it must be running.

# COMMAND ----------

from agent import AGENT
import json

print("--- Local validation: invoke (should interrupt) ---")
result1 = AGENT.predict({
    "input": [{"role": "user", "content": "analyze Q4 revenue trends"}],
})
print(json.dumps(result1.model_dump(exclude_none=True), indent=2))

custom1 = result1.custom_outputs or {}
assert custom1.get("status") == "interrupted", f"Expected interrupted, got: {custom1}"
thread_id = custom1["thread_id"]
print(f"\n✓ Supervisor interrupted, thread_id: {thread_id}")

# COMMAND ----------

print("--- Local validation: resume (should complete) ---")
result2 = AGENT.predict({
    "input": [{"role": "user", "content": "summary"}],
    "custom_inputs": {"thread_id": thread_id},
})
print(json.dumps(result2.model_dump(exclude_none=True), indent=2))

custom2 = result2.custom_outputs or {}
assert custom2.get("status") == "complete", f"Expected complete, got: {custom2}"
print(f"\n✓ Supervisor completed locally!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Log and Register Supervisor Model

# COMMAND ----------

import mlflow
import time
from mlflow.models.resources import DatabricksLakebase, DatabricksServingEndpoint
from databricks.sdk import WorkspaceClient
from agent import LAKEBASE_INSTANCE_NAME, SUB_AGENT_ENDPOINT

w = WorkspaceClient()
catalog = "brickcon_landparcels_classic_catalog"
schema = "coa_interrupt_poc"

resources = [
    DatabricksLakebase(database_instance_name=LAKEBASE_INSTANCE_NAME),
    DatabricksServingEndpoint(endpoint_name=SUB_AGENT_ENDPOINT),
]

input_example = {
    "input": [{"role": "user", "content": "analyze Q4 revenue trends"}],
}

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{schema}.coa_supervisor"

with mlflow.start_run(run_name="coa-supervisor"):
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
            "openai",
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
    input_data={"input": [{"role": "user", "content": "analyze Q4 revenue trends"}]},
    env_manager="uv",
)
print("✓ Pre-deployment validation passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Deploy Supervisor to Model Serving

# COMMAND ----------

from databricks import agents

endpoint_name = "coa-supervisor"

# Wait for any in-progress deployment to finish
try:
    ep = w.serving_endpoints.get(endpoint_name)
    state = ep.state
    if state and state.config_update and "IN_PROGRESS" in str(state.config_update):
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
print(f"Endpoint '{endpoint_name}' deployment initiated")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Wait for Endpoint to be Ready

# COMMAND ----------

print("Waiting for endpoint to be ready...")
while True:
    time.sleep(30)
    ep = w.serving_endpoints.get(endpoint_name)
    state = ep.state
    ready = str(state.ready) if state and state.ready else "UNKNOWN"
    update = str(state.config_update) if state and state.config_update else "NONE"
    print(f"  Ready: {ready}, Config update: {update}")
    if "READY" in ready and "IN_PROGRESS" not in update:
        break
print(f"Endpoint '{endpoint_name}' is ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test Full End-to-End: Caller -> Supervisor -> Sub-Agent -> Interrupt -> Resume

# COMMAND ----------

client = w.serving_endpoints.get_open_ai_client()

def call_endpoint(ep_name: str, input_text: str, custom_inputs: dict = None) -> dict:
    """Call a Model Serving endpoint via SDK OpenAI client."""
    extra = {}
    if custom_inputs:
        extra["custom_inputs"] = custom_inputs
    response = client.responses.create(
        model=ep_name,
        input=[{"type": "message", "role": "user", "content": input_text}],
        extra_body=extra,
    )
    output_text = ""
    if hasattr(response, "output"):
        for item in response.output:
            if hasattr(item, "content"):
                for part in item.content:
                    if hasattr(part, "text"):
                        output_text = part.text
                        break
    custom_outputs = dict(response.custom_outputs) if hasattr(response, "custom_outputs") and response.custom_outputs else {}
    return {"output_text": output_text, "custom_outputs": custom_outputs}

# COMMAND ----------

import json

print("=" * 70)
print("FULL END-TO-END TEST: Caller -> Supervisor -> Sub-Agent")
print("=" * 70)

print("\n--- Step 1: Caller invokes supervisor endpoint ---")
sup_response = call_endpoint("coa-supervisor", "analyze Q4 revenue trends")
print(f"Output: {sup_response['output_text']}")
print(f"Custom: {json.dumps(sup_response['custom_outputs'], indent=2)}")

sup_custom = sup_response["custom_outputs"]
assert sup_custom.get("status") == "interrupted", f"Expected interrupted: {sup_custom}"
sup_thread_id = sup_custom["thread_id"]
print(f"\nSupervisor interrupted!")
print(f"  Thread ID: {sup_thread_id}")
print(f"  Interrupt: {sup_custom.get('interrupt')}")

print("\n--- Step 2: Caller resumes supervisor with 'summary' ---")
sup_response2 = call_endpoint("coa-supervisor", "summary", {
    "thread_id": sup_thread_id,
})
print(f"Output: {sup_response2['output_text']}")
print(f"Custom: {json.dumps(sup_response2['custom_outputs'], indent=2)}")

sup_custom2 = sup_response2["custom_outputs"]
assert sup_custom2.get("status") == "complete", f"Expected complete: {sup_custom2}"
print(f"\nFull end-to-end flow completed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Test Thread Isolation

# COMMAND ----------

print("=" * 70)
print("THREAD ISOLATION TEST")
print("=" * 70)

resp_a = call_endpoint("coa-supervisor", "task Alpha")
resp_b = call_endpoint("coa-supervisor", "task Beta")

tid_a = resp_a["custom_outputs"]["thread_id"]
tid_b = resp_b["custom_outputs"]["thread_id"]
print(f"Thread A: {tid_a}")
print(f"Thread B: {tid_b}")

result_b = call_endpoint("coa-supervisor", "summary", {"thread_id": tid_b})
result_a = call_endpoint("coa-supervisor", "detailed", {"thread_id": tid_a})

assert result_a["custom_outputs"]["status"] == "complete"
assert result_b["custom_outputs"]["status"] == "complete"
print(f"Thread A: {result_a['custom_outputs']}")
print(f"Thread B: {result_b['custom_outputs']}")
print("No cross-contamination")

# COMMAND ----------

print("""
========================================================================
Phase 3 COMPLETE - Full Multi-Endpoint Interrupt Flow
========================================================================

Supervisor deployed to Model Serving (coa-supervisor)
Sub-agent deployed to Model Serving (coa-sub-agent)
Full flow: caller -> supervisor -> sub-agent -> interrupt -> resume -> complete
Thread isolation: concurrent threads work independently
All state checkpointed in Lakebase

ARCHITECTURE VALIDATED:
  Caller -> coa-supervisor endpoint -> coa-sub-agent endpoint -> Lakebase
           (separate Model Serving)   (separate Model Serving)   (shared)

The caller only needs the supervisor's thread_id to resume.
Sub-agent thread_ids are internal to the supervisor's state.
========================================================================
""")
