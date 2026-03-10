# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 1c: Simulate Cross-Endpoint Interrupt Protocol
# MAGIC
# MAGIC This is the critical test. We simulate what Model Serving will do:
# MAGIC - Sub-agent and supervisor are **separate graphs with separate checkpointers**
# MAGIC - They communicate via a structured protocol (simulating HTTP calls)
# MAGIC - The sub-agent returns `{"status": "interrupted", ...}` when it hits interrupt()
# MAGIC - The supervisor detects this, stores the sub-agent's thread_id, and interrupts itself
# MAGIC - On resume, the supervisor calls the sub-agent again with the resume value
# MAGIC
# MAGIC No actual HTTP or Model Serving — just validating the protocol logic.

# COMMAND ----------

# MAGIC %pip install langgraph langgraph-checkpoint-postgres "databricks-langchain[memory]>=0.16.0" psycopg[binary] psycopg-pool
# MAGIC %restart_python

# COMMAND ----------

import uuid
from urllib.parse import quote_plus
from databricks.sdk import WorkspaceClient
from typing import TypedDict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import interrupt, Command

w = WorkspaceClient()
LAKEBASE_INSTANCE_NAME = "coa-checkpoint"
DB_NAME = "coa_checkpoints"

instance = w.database.get_database_instance(LAKEBASE_INSTANCE_NAME)
cred = w.database.generate_database_credential(
    request_id=str(uuid.uuid4()),
    instance_names=[LAKEBASE_INSTANCE_NAME]
)
user_email = w.current_user.me().user_name
conn_string = f"postgresql://{quote_plus(user_email)}:{quote_plus(cred.token)}@{instance.read_write_dns}:5432/{DB_NAME}?sslmode=require"
print(f"Connected to Lakebase: {instance.read_write_dns}/{DB_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Define the Sub-Agent Graph
# MAGIC
# MAGIC Same as before — a simple graph that interrupts to ask a question.

# COMMAND ----------

class SubAgentState(TypedDict):
    task: str
    clarification: str
    result: str


def sub_agent_work(state: SubAgentState) -> dict:
    """Sub-agent does work and needs human clarification."""
    clarification = interrupt({
        "question": f"To complete '{state['task']}', which format do you prefer?",
        "options": ["detailed", "summary", "bullet points"]
    })
    return {
        "clarification": clarification,
        "result": f"Completed '{state['task']}' in {clarification} format"
    }


sub_builder = StateGraph(SubAgentState)
sub_builder.add_node("work", sub_agent_work)
sub_builder.add_edge(START, "work")
sub_builder.add_edge("work", END)

print("Sub-agent graph defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Sub-Agent Endpoint Simulator
# MAGIC
# MAGIC This simulates what the Model Serving `ResponsesAgent.predict()` would do.
# MAGIC It's a function that takes a request dict and returns a response dict.

# COMMAND ----------

def sub_agent_endpoint(request: dict, conn_str: str) -> dict:
    """
    Simulates the sub-agent Model Serving endpoint.

    Request format:
      {"action": "invoke", "input": {"task": "..."}}
      {"action": "resume", "thread_id": "...", "resume_value": "..."}

    Response format:
      {"status": "complete", "thread_id": "...", "result": {...}}
      {"status": "interrupted", "thread_id": "...", "interrupt": {...}}
    """
    with PostgresSaver.from_conn_string(conn_str) as checkpointer:
        checkpointer.setup()
        graph = sub_builder.compile(checkpointer=checkpointer)

        action = request.get("action", "invoke")
        thread_id = request.get("thread_id", f"sub-{uuid.uuid4().hex[:8]}")
        config = {"configurable": {"thread_id": thread_id}}

        if action == "resume":
            resume_value = request["resume_value"]
            result = graph.invoke(Command(resume=resume_value), config)
        else:
            input_state = request.get("input", {})
            # Ensure all required state keys
            input_state.setdefault("clarification", "")
            input_state.setdefault("result", "")
            result = graph.invoke(input_state, config)

        # Check if graph is paused at an interrupt
        snapshot = graph.get_state(config)
        if snapshot.next:
            # Graph is interrupted — extract the interrupt value
            interrupt_value = None
            if snapshot.tasks:
                for task in snapshot.tasks:
                    if task.interrupts:
                        interrupt_value = task.interrupts[0].value
                        break

            return {
                "status": "interrupted",
                "thread_id": thread_id,
                "interrupt": interrupt_value,
            }

        return {
            "status": "complete",
            "thread_id": thread_id,
            "result": dict(result),
        }

print("Sub-agent endpoint simulator defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test Sub-Agent Endpoint Directly

# COMMAND ----------

# Test invoke → should interrupt
response1 = sub_agent_endpoint(
    {"action": "invoke", "input": {"task": "analyze Q4 revenue"}},
    conn_string
)
print(f"Response 1: {response1}")
assert response1["status"] == "interrupted", f"Expected interrupted, got {response1['status']}"
assert response1["interrupt"] is not None
print(f"\n✓ Sub-agent interrupted with: {response1['interrupt']}")

# Test resume → should complete
response2 = sub_agent_endpoint(
    {"action": "resume", "thread_id": response1["thread_id"], "resume_value": "bullet points"},
    conn_string
)
print(f"\nResponse 2: {response2}")
assert response2["status"] == "complete", f"Expected complete, got {response2['status']}"
assert "bullet points" in response2["result"]["result"]
print(f"\n✓ Sub-agent completed with result: {response2['result']['result']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Define the Supervisor Graph
# MAGIC
# MAGIC The supervisor has a node that calls the sub-agent endpoint.
# MAGIC When the sub-agent returns "interrupted", the supervisor:
# MAGIC 1. Stores the sub-agent's thread_id in its own state
# MAGIC 2. Calls `interrupt()` itself to surface the question to the caller
# MAGIC 3. On resume, calls the sub-agent endpoint with the resume value

# COMMAND ----------

class SupervisorState(TypedDict):
    task: str
    sub_agent_thread_id: str
    sub_agent_interrupted: bool
    interrupt_payload: dict
    result: str


def call_sub_agent(state: SupervisorState) -> dict:
    """
    Supervisor node that calls the sub-agent endpoint.
    Handles both initial invocation and resume.
    """
    if state.get("sub_agent_interrupted") and state.get("sub_agent_thread_id"):
        # We're resuming — the interrupt() below will return the user's response
        user_response = interrupt(state["interrupt_payload"])

        # Now call the sub-agent endpoint to resume it
        response = sub_agent_endpoint({
            "action": "resume",
            "thread_id": state["sub_agent_thread_id"],
            "resume_value": user_response,
        }, conn_string)
    else:
        # First call — invoke the sub-agent
        response = sub_agent_endpoint({
            "action": "invoke",
            "input": {"task": state["task"]},
        }, conn_string)

    if response["status"] == "interrupted":
        # Sub-agent is interrupted — store its thread_id and interrupt ourselves
        return {
            "sub_agent_thread_id": response["thread_id"],
            "sub_agent_interrupted": True,
            "interrupt_payload": response["interrupt"],
        }

    # Sub-agent completed
    return {
        "sub_agent_interrupted": False,
        "result": response["result"]["result"],
    }


def check_interrupt(state: SupervisorState) -> str:
    """Route back to call_sub_agent if interrupted, else to finalize."""
    if state.get("sub_agent_interrupted"):
        return "call_sub_agent"
    return "finalize"


def finalize(state: SupervisorState) -> dict:
    """Wrap up the supervisor's response."""
    return {"result": f"[Supervisor] {state['result']}"}


sup_builder = StateGraph(SupervisorState)
sup_builder.add_node("call_sub_agent", call_sub_agent)
sup_builder.add_node("finalize", finalize)
sup_builder.add_edge(START, "call_sub_agent")
sup_builder.add_conditional_edges("call_sub_agent", check_interrupt, {
    "call_sub_agent": "call_sub_agent",
    "finalize": "finalize",
})
sup_builder.add_edge("finalize", END)

print("Supervisor graph defined")
print("Flow: START → call_sub_agent → (if interrupted: loop back after interrupt()) → finalize → END")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Supervisor Endpoint Simulator

# COMMAND ----------

def supervisor_endpoint(request: dict, conn_str: str) -> dict:
    """
    Simulates the supervisor Model Serving endpoint.
    Same protocol as sub-agent endpoint.
    """
    with PostgresSaver.from_conn_string(conn_str) as checkpointer:
        checkpointer.setup()
        graph = sup_builder.compile(checkpointer=checkpointer)

        action = request.get("action", "invoke")
        thread_id = request.get("thread_id", f"sup-{uuid.uuid4().hex[:8]}")
        config = {"configurable": {"thread_id": thread_id}}

        if action == "resume":
            resume_value = request["resume_value"]
            result = graph.invoke(Command(resume=resume_value), config)
        else:
            input_state = {
                "task": request.get("input", {}).get("task", ""),
                "sub_agent_thread_id": "",
                "sub_agent_interrupted": False,
                "interrupt_payload": {},
                "result": "",
            }
            result = graph.invoke(input_state, config)

        # Check if graph is paused at an interrupt
        snapshot = graph.get_state(config)
        if snapshot.next:
            interrupt_value = None
            if snapshot.tasks:
                for task in snapshot.tasks:
                    if task.interrupts:
                        interrupt_value = task.interrupts[0].value
                        break

            return {
                "status": "interrupted",
                "thread_id": thread_id,
                "interrupt": interrupt_value,
            }

        return {
            "status": "complete",
            "thread_id": thread_id,
            "result": result.get("result", ""),
        }

print("Supervisor endpoint simulator defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Test Full End-to-End Flow
# MAGIC
# MAGIC Caller → Supervisor endpoint → Sub-agent endpoint → interrupt → resume → complete

# COMMAND ----------

print("=" * 60)
print("FULL END-TO-END TEST: Cross-Endpoint Interrupt Protocol")
print("=" * 60)

# Step 1: Caller invokes supervisor
print("\n--- Step 1: Caller invokes supervisor ---")
sup_response1 = supervisor_endpoint(
    {"action": "invoke", "input": {"task": "analyze Q4 revenue"}},
    conn_string
)
print(f"Supervisor response: {sup_response1}")

assert sup_response1["status"] == "interrupted", f"Expected interrupted, got {sup_response1['status']}"
print(f"\n✓ Supervisor returned interrupt to caller")
print(f"  Thread ID: {sup_response1['thread_id']}")
print(f"  Interrupt payload: {sup_response1['interrupt']}")

# Step 2: Caller resumes supervisor with their answer
print("\n--- Step 2: Caller resumes with 'detailed' ---")
sup_response2 = supervisor_endpoint(
    {
        "action": "resume",
        "thread_id": sup_response1["thread_id"],
        "resume_value": "detailed",
    },
    conn_string
)
print(f"Supervisor response: {sup_response2}")

assert sup_response2["status"] == "complete", f"Expected complete, got {sup_response2['status']}"
assert "detailed" in sup_response2["result"]
print(f"\n✓ Supervisor returned final result: {sup_response2['result']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test Cold Resume (Fresh Credentials)

# COMMAND ----------

print("=" * 60)
print("COLD RESUME TEST")
print("=" * 60)

# Invoke and get interrupted
print("\n--- Invoke supervisor ---")
cold_response1 = supervisor_endpoint(
    {"action": "invoke", "input": {"task": "compare cloud providers"}},
    conn_string
)
assert cold_response1["status"] == "interrupted"
cold_thread_id = cold_response1["thread_id"]
print(f"✓ Interrupted, thread_id: {cold_thread_id}")

# Get fresh credentials (simulates endpoint scale-to-zero and back)
print("\n--- Simulating endpoint restart (fresh credentials) ---")
cred2 = w.database.generate_database_credential(
    request_id=str(uuid.uuid4()),
    instance_names=[LAKEBASE_INSTANCE_NAME]
)
conn_string_2 = f"postgresql://{quote_plus(user_email)}:{quote_plus(cred2.token)}@{instance.read_write_dns}:5432/{DB_NAME}?sslmode=require"

# Resume with fresh connection
print("--- Resume with fresh connection ---")
cold_response2 = supervisor_endpoint(
    {"action": "resume", "thread_id": cold_thread_id, "resume_value": "summary"},
    conn_string_2
)
assert cold_response2["status"] == "complete"
assert "summary" in cold_response2["result"]
print(f"✓ Cold resume succeeded: {cold_response2['result']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Test Thread Isolation

# COMMAND ----------

print("=" * 60)
print("THREAD ISOLATION TEST")
print("=" * 60)

# Start two concurrent threads
print("\n--- Starting two concurrent supervisor threads ---")
thread_a = supervisor_endpoint(
    {"action": "invoke", "input": {"task": "task Alpha"}},
    conn_string
)
thread_b = supervisor_endpoint(
    {"action": "invoke", "input": {"task": "task Beta"}},
    conn_string
)
assert thread_a["status"] == "interrupted"
assert thread_b["status"] == "interrupted"
print(f"✓ Thread A interrupted: {thread_a['thread_id']}")
print(f"✓ Thread B interrupted: {thread_b['thread_id']}")

# Resume B first, then A (out of order)
print("\n--- Resuming B first, then A ---")
result_b = supervisor_endpoint(
    {"action": "resume", "thread_id": thread_b["thread_id"], "resume_value": "summary"},
    conn_string
)
result_a = supervisor_endpoint(
    {"action": "resume", "thread_id": thread_a["thread_id"], "resume_value": "detailed"},
    conn_string
)

assert result_a["status"] == "complete"
assert result_b["status"] == "complete"
assert "detailed" in result_a["result"]
assert "summary" in result_b["result"]
# Verify no cross-contamination
assert "Alpha" in result_a["result"] or "detailed" in result_a["result"]
assert "Beta" in result_b["result"] or "summary" in result_b["result"]
print(f"✓ Thread A result: {result_a['result']}")
print(f"✓ Thread B result: {result_b['result']}")
print("✓ No cross-contamination between threads")

# COMMAND ----------

print("""
========================================================================
Phase 1c COMPLETE — Cross-Endpoint Interrupt Protocol Validated
========================================================================

✓ Sub-agent endpoint: invoke → interrupt → resume → complete
✓ Supervisor endpoint: invoke → calls sub-agent → interrupt propagates up → resume flows down → complete
✓ Cold resume: fresh credentials, graph reconstructs from Lakebase checkpoint
✓ Thread isolation: concurrent threads resume independently, no cross-contamination

KEY DESIGN PATTERN (validated):
  1. Sub-agent hits interrupt() → checkpoints to Lakebase → returns {"status": "interrupted", "thread_id": "sub-xxx", "interrupt": {...}}
  2. Supervisor's call_sub_agent node receives interrupted response
  3. Supervisor stores sub-agent thread_id in its own state
  4. Supervisor calls interrupt() ITSELF → checkpoints → returns {"status": "interrupted", "thread_id": "sup-yyy", "interrupt": {...}}
  5. Caller only knows supervisor's thread_id
  6. On resume: supervisor resumes → calls sub-agent endpoint with resume → sub-agent resumes → result flows back

NEXT: Phase 2 — Deploy sub-agent to actual Model Serving endpoint
========================================================================
""")
