# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 0: Setup & Verify Lakebase
# MAGIC
# MAGIC Verify connectivity to the `coa-checkpoint` Lakebase Provisioned instance
# MAGIC and test that LangGraph's PostgresSaver can create checkpoint tables.

# COMMAND ----------

# MAGIC %pip install langgraph langgraph-checkpoint-postgres "databricks-langchain[memory]>=0.16.0" psycopg[binary] psycopg-pool nest_asyncio
# MAGIC %restart_python

# COMMAND ----------

import os
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

LAKEBASE_INSTANCE_NAME = "coa-checkpoint"
DB_NAME = "coa_checkpoints"

# Get connection details
instance = w.database.get_database_instance(LAKEBASE_INSTANCE_NAME)
print(f"Instance: {instance.name}")
print(f"State: {instance.state}")
print(f"Host: {instance.read_write_dns}")
print(f"PG Version: {instance.pg_version}")

assert str(instance.state) == "DatabaseInstanceState.AVAILABLE", f"Instance not ready: {instance.state}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create dedicated database and test connection

# COMMAND ----------

import uuid
import psycopg
from urllib.parse import quote_plus

# Generate OAuth credential
cred = w.database.generate_database_credential(
    request_id=str(uuid.uuid4()),
    instance_names=[LAKEBASE_INSTANCE_NAME]
)

user_email = w.current_user.me().user_name
host = instance.read_write_dns

# Connect to default postgres db first to create our database
admin_conn_string = f"postgresql://{quote_plus(user_email)}:{quote_plus(cred.token)}@{host}:5432/postgres?sslmode=require"

with psycopg.connect(admin_conn_string, autocommit=True) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        version = cur.fetchone()[0]
        print(f"Connected! PostgreSQL version: {version}")

        # Create database if not exists
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
        if not cur.fetchone():
            cur.execute(f"CREATE DATABASE {DB_NAME}")
            print(f"Created database: {DB_NAME}")
        else:
            print(f"Database already exists: {DB_NAME}")

# Now connect to our dedicated database
conn_string = f"postgresql://{quote_plus(user_email)}:{quote_plus(cred.token)}@{host}:5432/{DB_NAME}?sslmode=require"

with psycopg.connect(conn_string) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT current_database()")
        print(f"Connected to database: {cur.fetchone()[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test PostgresSaver checkpoint table creation

# COMMAND ----------

from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string(conn_string) as checkpointer:
    checkpointer.setup()
    print("Checkpoint tables created successfully!")

# Verify tables exist
with psycopg.connect(conn_string) as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = [row[0] for row in cur.fetchall()]
        print(f"Tables in public schema: {tables}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test AsyncCheckpointSaver (databricks_langchain wrapper)

# COMMAND ----------

import asyncio
import nest_asyncio
nest_asyncio.apply()

from databricks_langchain import AsyncCheckpointSaver

async def test_async_checkpointer():
    async with AsyncCheckpointSaver(instance_name=LAKEBASE_INSTANCE_NAME) as checkpointer:
        await checkpointer.setup()
        print("AsyncCheckpointSaver connected and setup successfully!")

asyncio.get_event_loop().run_until_complete(test_async_checkpointer())

# COMMAND ----------

print(f"""
Phase 0 complete! Lakebase is ready for checkpointing.

Connection details:
  Instance: {LAKEBASE_INSTANCE_NAME}
  Host: {host}
  Database: {DB_NAME}

Use this conn_string pattern in subsequent notebooks:
  postgresql://<user>:<token>@{host}:5432/{DB_NAME}?sslmode=require
""")
