# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 1a: Single Agent with Interrupt + Lakebase Checkpoint
# MAGIC
# MAGIC Simplest possible test: a LangGraph graph with one node that calls `interrupt()`.
# MAGIC Verify it pauses, checkpoints to Lakebase, and resumes correctly.

# COMMAND ----------

# MAGIC %pip install langgraph langgraph-checkpoint-postgres "databricks-langchain[memory]>=0.16.0" psycopg[binary] psycopg-pool
# MAGIC %restart_python

# COMMAND ----------

import uuid
import psycopg
from urllib.parse import quote_plus
from databricks.sdk import WorkspaceClient
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import interrupt, Command

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: Lakebase connection

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
# MAGIC ## Define the simplest possible graph with an interrupt

# COMMAND ----------

class State(TypedDict):
    name: str
    greeting: str


def ask_name(state: State) -> dict:
    """Node that interrupts to ask for a name."""
    name = interrupt({"question": "What is your name?"})
    return {"name": name}


def greet(state: State) -> dict:
    """Node that uses the name to produce a greeting."""
    return {"greeting": f"Hello, {state['name']}! Welcome to the COA Interrupt POC."}


# Build the graph
builder = StateGraph(State)
builder.add_node("ask_name", ask_name)
builder.add_node("greet", greet)
builder.add_edge(START, "ask_name")
builder.add_edge("ask_name", "greet")
builder.add_edge("greet", END)

print("Graph defined: START → ask_name (interrupt) → greet → END")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 1: Invoke graph — should pause at interrupt

# COMMAND ----------

thread_id = f"test-{uuid.uuid4().hex[:8]}"
config = {"configurable": {"thread_id": thread_id}}

with PostgresSaver.from_conn_string(conn_string) as checkpointer:
    checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)

    # First invocation — should hit interrupt and pause
    result = graph.invoke({"name": "", "greeting": ""}, config)
    print(f"Result after first invoke: {result}")

    # Check the state — should show the graph is paused
    snapshot = graph.get_state(config)
    print(f"\nGraph paused: {bool(snapshot.next)}")
    print(f"Pending nodes: {snapshot.next}")

    if snapshot.tasks:
        for task in snapshot.tasks:
            if task.interrupts:
                for intr in task.interrupts:
                    print(f"Interrupt value: {intr.value}")

    assert snapshot.next, "Graph should be paused at interrupt!"
    print("\n✓ Graph correctly paused at interrupt")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 2: Verify checkpoint exists in Lakebase

# COMMAND ----------

with psycopg.connect(conn_string) as conn:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT thread_id, checkpoint_ns, checkpoint_id FROM checkpoints WHERE thread_id = %s",
            (thread_id,)
        )
        rows = cur.fetchall()
        print(f"Checkpoints for thread {thread_id}:")
        for row in rows:
            print(f"  thread_id={row[0]}, ns={row[1]}, checkpoint_id={row[2]}")

        assert len(rows) > 0, "No checkpoints found in Lakebase!"
        print(f"\n✓ Found {len(rows)} checkpoint(s) in Lakebase")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 3: Resume the graph with a name

# COMMAND ----------

with PostgresSaver.from_conn_string(conn_string) as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)

    # Resume with the human's response
    result = graph.invoke(Command(resume="Alice"), config)
    print(f"Result after resume: {result}")

    # Verify the graph completed
    snapshot = graph.get_state(config)
    print(f"\nGraph paused: {bool(snapshot.next)}")
    assert not snapshot.next, "Graph should be complete!"

    assert result["name"] == "Alice", f"Expected name='Alice', got '{result['name']}'"
    assert "Alice" in result["greeting"], f"Greeting should contain 'Alice': {result['greeting']}"

    print("\n✓ Graph resumed and completed correctly")
    print(f"✓ Greeting: {result['greeting']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 4: Verify this works with a fresh connection (simulates cold resume)
# MAGIC
# MAGIC This simulates what happens on Model Serving — the graph is reconstructed
# MAGIC from the checkpoint on each request.

# COMMAND ----------

# Start a new thread, invoke to get interrupt, then resume with a FRESH connection
thread_id_2 = f"cold-{uuid.uuid4().hex[:8]}"
config_2 = {"configurable": {"thread_id": thread_id_2}}

# First call: invoke and hit interrupt
with PostgresSaver.from_conn_string(conn_string) as checkpointer:
    checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
    graph.invoke({"name": "", "greeting": ""}, config_2)
    snapshot = graph.get_state(config_2)
    assert snapshot.next, "Should be paused"
    print("✓ First call: graph paused at interrupt")

# Simulate endpoint restart: get fresh credentials and reconnect
cred2 = w.database.generate_database_credential(
    request_id=str(uuid.uuid4()),
    instance_names=[LAKEBASE_INSTANCE_NAME]
)
conn_string_2 = f"postgresql://{quote_plus(user_email)}:{quote_plus(cred2.token)}@{instance.read_write_dns}:5432/{DB_NAME}?sslmode=require"

# Second call: resume from cold with fresh connection
with PostgresSaver.from_conn_string(conn_string_2) as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    result = graph.invoke(Command(resume="Bob"), config_2)
    print(f"✓ Cold resume result: {result}")
    assert result["name"] == "Bob"
    assert "Bob" in result["greeting"]
    print("✓ Cold resume works — checkpoint survives reconnection")

# COMMAND ----------

print("""
========================================
Phase 1a COMPLETE
========================================
✓ interrupt() pauses the graph
✓ Checkpoint persisted in Lakebase
✓ Command(resume=...) resumes correctly
✓ Cold resume from fresh connection works
========================================
""")
