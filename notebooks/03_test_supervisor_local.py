# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 1b: Supervisor + Sub-Agent with Interrupt (In-Process)
# MAGIC
# MAGIC Test that a supervisor graph can call a sub-agent as a subgraph,
# MAGIC and the sub-agent's `interrupt()` propagates up to the supervisor.
# MAGIC Both graphs are checkpointed in Lakebase.

# COMMAND ----------

# MAGIC %pip install langgraph langgraph-checkpoint-postgres "databricks-langchain[memory]>=0.16.0" psycopg[binary] psycopg-pool
# MAGIC %restart_python

# COMMAND ----------

import uuid
from urllib.parse import quote_plus
from databricks.sdk import WorkspaceClient
from typing import TypedDict, Annotated, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import interrupt, Command

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

w = WorkspaceClient()
LAKEBASE_INSTANCE_NAME = "coa-checkpoint"

instance = w.database.get_database_instance(LAKEBASE_INSTANCE_NAME)
cred = w.database.generate_database_credential(
    request_id=str(uuid.uuid4()),
    instance_names=[LAKEBASE_INSTANCE_NAME]
)
user_email = w.current_user.me().user_name
DB_NAME = "coa_checkpoints"
conn_string = f"postgresql://{quote_plus(user_email)}:{quote_plus(cred.token)}@{instance.read_write_dns}:5432/{DB_NAME}?sslmode=require"
print(f"Connected to Lakebase: {instance.read_write_dns}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Sub-Agent Graph
# MAGIC
# MAGIC A simple sub-agent that:
# MAGIC 1. Receives a task
# MAGIC 2. Interrupts to ask for clarification
# MAGIC 3. Completes the task with the clarification

# COMMAND ----------

class SubAgentState(TypedDict):
    task: str
    clarification: str
    result: str


def sub_agent_work(state: SubAgentState) -> dict:
    """Sub-agent node that needs human clarification."""
    clarification = interrupt({
        "question": f"I need more details to complete: '{state['task']}'",
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

# Compile WITHOUT checkpointer — the parent's checkpointer handles both
sub_graph = sub_builder.compile()

print("Sub-agent graph: START → work (interrupt) → END")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Supervisor Graph
# MAGIC
# MAGIC Supervisor that:
# MAGIC 1. Receives a task from the user
# MAGIC 2. Delegates to the sub-agent (as a subgraph node)
# MAGIC 3. Returns the sub-agent's result

# COMMAND ----------

class SupervisorState(TypedDict):
    task: str
    clarification: str
    result: str
    final_answer: str


def route_to_sub_agent(state: SupervisorState) -> dict:
    """Supervisor delegates task to sub-agent subgraph."""
    # When used as a subgraph, the sub-agent's interrupt propagates up automatically
    sub_result = sub_graph.invoke({
        "task": state["task"],
        "clarification": "",
        "result": ""
    })
    return {
        "clarification": sub_result["clarification"],
        "result": sub_result["result"]
    }


def summarize(state: SupervisorState) -> dict:
    """Supervisor summarizes the sub-agent's result."""
    return {
        "final_answer": f"Supervisor says: {state['result']}"
    }


sup_builder = StateGraph(SupervisorState)
sup_builder.add_node("delegate", route_to_sub_agent)
sup_builder.add_node("summarize", summarize)
sup_builder.add_edge(START, "delegate")
sup_builder.add_edge("delegate", "summarize")
sup_builder.add_edge("summarize", END)

print("Supervisor graph: START → delegate (calls sub-agent) → summarize → END")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 1: Invoke supervisor — sub-agent's interrupt should propagate up

# COMMAND ----------

thread_id = f"sup-{uuid.uuid4().hex[:8]}"
config = {"configurable": {"thread_id": thread_id}}

with PostgresSaver.from_conn_string(conn_string) as checkpointer:
    checkpointer.setup()
    supervisor = sup_builder.compile(checkpointer=checkpointer)

    # Invoke — should pause when sub-agent hits interrupt
    result = supervisor.invoke({
        "task": "analyze Q4 revenue",
        "clarification": "",
        "result": "",
        "final_answer": ""
    }, config)

    print(f"Result after invoke: {result}")

    # Check state
    snapshot = supervisor.get_state(config)
    print(f"\nSupervisor paused: {bool(snapshot.next)}")
    print(f"Pending nodes: {snapshot.next}")

    if snapshot.tasks:
        for task in snapshot.tasks:
            if task.interrupts:
                for intr in task.interrupts:
                    print(f"Interrupt value: {intr.value}")

    assert snapshot.next, "Supervisor should be paused due to sub-agent interrupt!"
    print("\n✓ Sub-agent interrupt propagated up to supervisor")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 2: Resume supervisor — should flow through to sub-agent

# COMMAND ----------

with PostgresSaver.from_conn_string(conn_string) as checkpointer:
    supervisor = sup_builder.compile(checkpointer=checkpointer)

    # Resume with the human's response
    result = supervisor.invoke(Command(resume="bullet points"), config)
    print(f"Result after resume: {result}")

    # Verify completed
    snapshot = supervisor.get_state(config)
    assert not snapshot.next, "Supervisor should be complete!"

    assert result["clarification"] == "bullet points"
    assert "bullet points" in result["result"]
    assert "Supervisor says:" in result["final_answer"]

    print(f"\n✓ Sub-agent completed with clarification: {result['clarification']}")
    print(f"✓ Sub-agent result: {result['result']}")
    print(f"✓ Supervisor final answer: {result['final_answer']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 3: Cold resume (fresh connection)

# COMMAND ----------

thread_id_2 = f"sup-cold-{uuid.uuid4().hex[:8]}"
config_2 = {"configurable": {"thread_id": thread_id_2}}

# Invoke and hit interrupt
with PostgresSaver.from_conn_string(conn_string) as checkpointer:
    supervisor = sup_builder.compile(checkpointer=checkpointer)
    supervisor.invoke({
        "task": "compare AWS vs Azure pricing",
        "clarification": "", "result": "", "final_answer": ""
    }, config_2)
    assert supervisor.get_state(config_2).next
    print("✓ First call: supervisor paused at sub-agent interrupt")

# Fresh credentials
cred2 = w.database.generate_database_credential(
    request_id=str(uuid.uuid4()),
    instance_names=[LAKEBASE_INSTANCE_NAME]
)
conn_string_2 = f"postgresql://{quote_plus(user_email)}:{quote_plus(cred2.token)}@{instance.read_write_dns}:5432/{DB_NAME}?sslmode=require"

# Resume from fresh connection
with PostgresSaver.from_conn_string(conn_string_2) as checkpointer:
    supervisor = sup_builder.compile(checkpointer=checkpointer)
    result = supervisor.invoke(Command(resume="detailed"), config_2)
    assert not supervisor.get_state(config_2).next
    print(f"✓ Cold resume: {result['final_answer']}")

# COMMAND ----------

print("""
========================================
Phase 1b COMPLETE
========================================
✓ Sub-agent interrupt propagates up to supervisor automatically
✓ Resume flows from supervisor down to sub-agent
✓ Both graphs share checkpointer in Lakebase
✓ Cold resume works with fresh connection
========================================

NOTE: This tested IN-PROCESS subgraph propagation.
Phase 1c will test the CROSS-ENDPOINT protocol where
the sub-agent and supervisor are separate processes.
========================================
""")
