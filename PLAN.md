# COA Interrupt POC — Test Plan

## Objective

Validate that a **multi-agent LangGraph architecture** with **human-in-the-loop interrupts** works on Databricks **Model Serving**, where:

1. A **supervisor agent** runs in one Model Serving endpoint
2. One or more **sub-agents** run in separate Model Serving endpoints
3. A sub-agent can **interrupt** execution to request human input
4. The interrupt is **surfaced back** through the supervisor to the caller
5. The caller provides input and the sub-agent **resumes** from its checkpoint
6. All graph state is **checkpointed in Lakebase** (Postgres)

## Prior Art

- **Databricks `agent-langgraph-short-term-memory` template**: Uses `ResponsesAgent` pattern with `@invoke()`/`@stream()` decorators, `AsyncCheckpointSaver`, and `databricks-langchain[memory]`. This is the modern pattern we'll follow for the checkpointer setup.
- **Puneet Jain's Unilever PoC** (`puneet-jain159/agent_with_state_human_in_loop`): Uses older `ChatAgent.predict()` pattern. More complex than we need, but validates the concept of interrupt + checkpointed state on Model Serving.
- **Stuart Lynn's whiteboard with Arm** (March 2026): `Interrupt` + `thread_id` + `awaiting_input` flag in `custom_outputs`, stored in supervisor graph state.

## Feasibility Summary

| Capability | Status | Notes |
|-----------|--------|-------|
| LangGraph `interrupt()` + `Command(resume=...)` | Supported (v0.2.57+) | Replaces older `NodeInterrupt` pattern |
| `AsyncCheckpointSaver` / `PostgresSaver` with Lakebase | Confirmed working | Used in official app template; SSL keepalive config needed |
| `databricks_langchain[memory]` | Available (>=0.16.0) | Wraps PostgresSaver with Databricks credential mgmt |
| Subgraph interrupt propagation (in-process) | Automatic | Interrupts bubble up via `checkpoint_ns` |
| `ResponsesAgent` on Model Serving | Current pattern | `mlflow.pyfunc.ResponsesAgent` with OpenAI Responses API compatibility |

### The Model Serving Challenge

Model Serving endpoints are **stateless request/response**. LangGraph's `interrupt()` pauses execution and expects to be resumed — but each HTTP request is independent.

**Solution:** Lakebase checkpointing + `ResponsesAgent` wrapper with `custom_outputs`.

When a sub-agent hits `interrupt()`:
1. Graph state is checkpointed to Lakebase (via `AsyncCheckpointSaver`)
2. The `ResponsesAgent` detects the interrupt via `graph.get_state(config).next`
3. Returns response with `custom_outputs={"awaiting_input": True, "interrupt_value": ..., "thread_id": ...}`
4. Supervisor recognizes this and propagates it upward via its own `custom_outputs`
5. On resume, caller sends `custom_inputs={"action": "resume", "resume_value": ..., "thread_id": ...}`
6. The handler invokes `graph.invoke(Command(resume=value), config)` to continue

The supervisor's own graph also checkpoints to Lakebase, tracking the sub-agent's `thread_id` and interrupt state.

### Known Gotchas

1. **Lakebase Provisioned only** — `CheckpointSaver` does not yet support Lakebase Autoscaling (support imminent as of March 2026)
2. **SSL idle disconnects** — `SSL SYSCALL error: EOF detected` after 8-10 hours idle. Mitigate with connection pooling + keepalives.
3. **Review App / AI Playground** do not natively understand interrupt/resume — testing via API calls or custom UI.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      Caller / UI                          │
│                                                           │
│  1. POST /serving-endpoints/supervisor/invocations         │
│     input: [{"role":"user","content":"analyze data"}]      │
│                                                           │
│  4. Response includes custom_outputs:                      │
│     {"awaiting_input": true, "thread_id": "sup-123",      │
│      "interrupt": {"question": "Which metric?"}}           │
│                                                           │
│  5. POST /serving-endpoints/supervisor/invocations         │
│     custom_inputs: {"action": "resume",                    │
│       "thread_id": "sup-123", "resume_value": "revenue"}   │
│                                                           │
│  8. Response: final result                                 │
└──────────────────────────────────────────────────────────┘
         │                                    ▲
         ▼                                    │
┌─────────────────────────────────────────────┐
│  Supervisor Endpoint (Model Serving)         │
│  ResponsesAgent wrapping LangGraph           │
│                                              │
│  - New task → invoke supervisor graph        │
│  - Resume → Command(resume=...) with         │
│    supervisor thread_id                      │
│  - Supervisor graph node calls sub-agent     │
│    endpoint via HTTP                         │
│  - Detects interrupt from sub-agent          │
│  - Stores sub-agent thread_id in own state   │
│  - Calls interrupt() itself to pause         │
│  - Returns custom_outputs with interrupt     │
│    or final result to caller                 │
└──────────────┬──────────────────────────────┘
               │
          2. HTTP POST to sub-agent endpoint
          6. HTTP POST with resume
               │
               ▼
┌─────────────────────────────────────────────┐
│  Sub-Agent Endpoint (Model Serving)          │
│  ResponsesAgent wrapping LangGraph           │
│                                              │
│  - New task → invoke sub-agent graph         │
│  - Graph hits interrupt() → state saved      │
│  - get_state().next detects pause            │
│  - Returns custom_outputs with interrupt     │
│  - Resume → Command(resume=...) with         │
│    sub-agent thread_id                       │
│  - Returns final result                      │
└─────────────────────────────────────────────┘

         Both endpoints share:
┌─────────────────────────────────────────────┐
│  Lakebase Provisioned (Postgres)             │
│  ├── checkpoint tables (auto-created)        │
│  └── checkpoint_writes tables                │
└─────────────────────────────────────────────┘
```

---

## Test Phases

### Phase 0: Prerequisites & Environment Setup
**Workspace:** brickcon-landparcels-classic (FE-VM Classic)
**URL:** https://fevm-brickcon-landparcels-classic.cloud.databricks.com

- [ ] Authenticate to workspace via Databricks CLI
- [ ] Create a Lakebase **Provisioned** Postgres instance
- [ ] Verify connectivity to Lakebase from a notebook
- [ ] Test `AsyncCheckpointSaver` / `PostgresSaver` creates checkpoint tables
- [ ] Verify Foundation Model API access (for the LLM backing the agents)

### Phase 1: Notebook Validation (In-Process)
Build and test the full flow in a notebook before deploying to endpoints. Keep it minimal — simplest possible graphs that prove the pattern.

#### 1a. Single agent with interrupt + Lakebase checkpoint
- [ ] Create a trivial LangGraph graph: one node that calls `interrupt("What is your name?")`
- [ ] Configure checkpointer pointed at Lakebase
- [ ] Invoke the graph → verify it pauses at interrupt
- [ ] Check `graph.get_state(config).next` confirms paused state
- [ ] Resume with `Command(resume="Alice")` → verify it completes with "Alice" in result
- [ ] Query Lakebase tables directly to verify checkpoint data exists

#### 1b. Supervisor + sub-agent with interrupt (in-process subgraph)
- [ ] Sub-agent graph: tool-calling agent where one tool triggers `interrupt()`
- [ ] Supervisor graph: routes to sub-agent as a subgraph node
- [ ] Invoke supervisor → verify interrupt propagates up automatically
- [ ] Resume → verify sub-agent completes and supervisor returns result

#### 1c. Simulate cross-endpoint protocol (in-process)
- [ ] Build the `ResponsesAgent` wrapper with interrupt detection
- [ ] Sub-agent wrapper: invoke graph, detect interrupt via `get_state().next`, return `custom_outputs`
- [ ] Supervisor wrapper: call sub-agent (simulated), detect interrupt, store thread_id, call its own `interrupt()`
- [ ] Test full invoke → interrupt → resume → complete cycle
- [ ] This validates the protocol before deploying to actual endpoints

### Phase 2: Deploy Sub-Agent to Model Serving
- [ ] Log sub-agent `ResponsesAgent` as MLflow model
- [ ] Register in Unity Catalog
- [ ] Create Model Serving endpoint
- [ ] Test direct invocation: new task → interrupt in `custom_outputs`
- [ ] Test direct resume: same thread_id + resume value → completion

### Phase 3: Deploy Supervisor to Model Serving
- [ ] Log supervisor `ResponsesAgent` as MLflow model
- [ ] Supervisor's graph node calls sub-agent endpoint via Databricks SDK / HTTP
- [ ] Create Model Serving endpoint for supervisor
- [ ] Test full end-to-end flow via supervisor endpoint
- [ ] Verify interrupt surfaces through supervisor to caller
- [ ] Verify resume flows through supervisor to sub-agent

### Phase 4: Edge Cases & Robustness
- [ ] Multiple concurrent threads with different interrupt states
- [ ] Checkpoint persistence — endpoint scales to zero and back
- [ ] Multiple sub-agents, only one interrupts
- [ ] Error handling — sub-agent endpoint unavailable during resume

---

## Key Design: "Interrupts All The Way Up"

When the sub-agent and supervisor are on **separate endpoints**, LangGraph's automatic subgraph interrupt propagation doesn't apply (that only works in-process). Instead:

1. **Sub-agent endpoint** hits `interrupt()` → checkpoints to Lakebase → returns `custom_outputs: {awaiting_input: true, thread_id: "sub-xxx", interrupt: {...}}`

2. **Supervisor's graph node** (the one that calls the sub-agent via HTTP) sees the `awaiting_input` response → stores `sub_agent_thread_id` in supervisor state → calls `interrupt()` **itself**, passing the sub-agent's interrupt payload up

3. **Supervisor endpoint** detects its own graph is now paused → returns `custom_outputs: {awaiting_input: true, thread_id: "sup-yyy", interrupt: {...}}`

4. **Caller** sees the interrupt → later sends resume with `thread_id: "sup-yyy"` and `resume_value: "the answer"`

5. **Supervisor endpoint** resumes its graph with `Command(resume="the answer")` → the supervisor node that called `interrupt()` receives "the answer" → makes HTTP call to sub-agent endpoint with `action: "resume", thread_id: "sub-xxx", resume_value: "the answer"`

6. **Sub-agent endpoint** resumes its graph → completes → returns result → supervisor returns final result to caller

The caller only ever knows about the **supervisor's thread_id**. The sub-agent thread_id is internal to the supervisor's state.

---

## Test Scenarios

### Scenario 1: Simple Interrupt and Resume
- Caller invokes supervisor: "analyze this data"
- Supervisor routes to sub-agent endpoint
- Sub-agent needs clarification → returns `awaiting_input: true`
- Supervisor stores sub-agent thread_id, interrupts itself
- Caller sees interrupt, responds "revenue"
- Supervisor resumes, forwards to sub-agent
- Sub-agent completes → result flows back
- **Success:** Correct result, both checkpoints in Lakebase

### Scenario 2: Checkpoint Survives Scale-to-Zero
- Sub-agent hits interrupt, endpoint scales to zero
- Caller resumes after delay
- Endpoint scales back up, loads checkpoint from Lakebase
- Sub-agent resumes and completes
- **Success:** Graph resumes correctly from cold checkpoint

### Scenario 3: Multiple Concurrent Threads
- Multiple callers invoke the supervisor simultaneously
- Different threads have different interrupt states
- Each resumes independently
- **Success:** No cross-contamination between threads

---

## Code Structure

```
coa-interrupt-poc/
├── PLAN.md                              # This file
├── common/
│   ├── lakebase_config.py               # Lakebase connection config
│   └── checkpointer.py                  # CheckpointSaver factory
├── sub_agent/
│   ├── graph.py                         # Sub-agent LangGraph graph (minimal)
│   └── agent.py                         # ResponsesAgent wrapper
├── supervisor/
│   ├── graph.py                         # Supervisor LangGraph graph
│   └── agent.py                         # ResponsesAgent wrapper
├── notebooks/
│   ├── 01_setup_lakebase.py             # Create Lakebase Provisioned instance
│   ├── 02_test_single_agent.py          # Phase 1a: single agent interrupt
│   ├── 03_test_supervisor_local.py      # Phase 1b: in-process supervisor
│   ├── 04_test_protocol_local.py        # Phase 1c: simulate endpoint protocol
│   ├── 05_deploy_sub_agent.py           # Phase 2: deploy sub-agent endpoint
│   ├── 06_deploy_supervisor.py          # Phase 3: deploy supervisor endpoint
│   └── 07_test_e2e.py                   # Phase 3-4: end-to-end tests
└── requirements.txt
```

---

## Dependencies

Based on the `agent-langgraph-short-term-memory` template:

```
databricks-langchain[memory]>=0.16.0
databricks-agents>=1.9.3
mlflow>=3.10.0
langgraph>=1.0.9
```

---

## Success Criteria

1. **Interrupt surfaces correctly**: Sub-agent `interrupt()` → supervisor `interrupt()` → caller sees `awaiting_input: true` in `custom_outputs`, across two Model Serving endpoints
2. **Resume works end-to-end**: Caller's `custom_inputs` → supervisor resumes → sub-agent resumes → result returned
3. **Lakebase checkpointing**: All graph state persisted; survives endpoint scale-to-zero
4. **Thread isolation**: Multiple concurrent threads don't interfere
5. **Simplicity**: The pattern is clean enough to hand to the customer as a reference

## Failure Criteria (Setup vs Feature)

**Setup issues (we fix these):**
- Lakebase connectivity from Model Serving (networking/auth)
- Package version incompatibilities
- MLflow model logging/loading issues

**Actual feature limitations (we document these):**
- `PostgresSaver` / `AsyncCheckpointSaver` incompatible with Lakebase
- LangGraph `interrupt()` can't be detected via `get_state().next`
- `Command(resume=...)` doesn't work from cold checkpoint
- Model Serving request timeout too short for checkpoint reconstruction

## References

- App template (checkpointer pattern): `github.com/databricks/app-templates/tree/main/agent-langgraph-short-term-memory`
- Puneet's PoC (interrupt concept): `github.com/puneet-jain159/agent_with_state_human_in_loop`
- Slack channels: #agents, #apa-lakebase, #lakebase-integration-agent-memory
- Key contacts: Bryan Qiu (Agent Platform), Jenny Sun (Lakebase SDK), Puneet Jain (HITL PoC)
