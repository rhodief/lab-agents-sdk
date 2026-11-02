import json
import queue
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.types import Command
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel

import deep_session as ds

# Namespace components must be safe for the store (same rule deepagents enforces)
NAME = re.compile(r"^[A-Za-z0-9\-_.@+:~]+$")

pool = ConnectionPool(
    ds.DATABASE_URL,
    max_size=10,
    # settings the LangGraph saver and store require on every connection
    kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    open=False,
)
checkpointer = PostgresSaver(pool)
store = PostgresStore(pool)
agents = None
running = set()  # thread_ids with a turn in progress; one turn per session at a time
running_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_app):
    global agents
    tracer_provider = ds.setup_tracing()
    pool.open()
    checkpointer.setup()
    store.setup()
    agents = ds.build_agents(checkpointer, store)
    yield
    pool.close()
    tracer_provider.force_flush()


app = FastAPI(title="Deep research agent", lifespan=lifespan)


def scope(tenant: str, user: str, team: str | None) -> ds.Tenant:
    """Demo only: the scope comes from the request. In production, take it from the authenticated user."""
    for value in (tenant, user, *([team] if team else [])):
        if not NAME.match(value):
            raise HTTPException(400, f"invalid name {value!r}: use letters, digits and - _ . @ + : ~")
    return ds.Tenant(tenant_id=tenant, user_id=user, team_id=team or None)


def checked(fn, *args):
    try:
        return fn(*args)
    except PermissionError as error:
        raise HTTPException(403, str(error)) from error


class JoinTeam(BaseModel):
    tenant: str
    user: str
    team: str


class Turn(BaseModel):
    tenant: str
    user: str
    team: str | None = None
    thread_id: str | None = None  # None starts a new session
    message: str | None = None  # a new prompt...
    answers: dict | None = None  # ...or {interrupt_id: answer} resuming pending interrupts


@app.get("/api/teams")
def teams(tenant: str, user: str):
    return {"teams": ds.user_teams(store, scope(tenant, user, None))}


@app.post("/api/teams")
def join(body: JoinTeam):
    ds.join_team(store, scope(body.tenant, body.user, body.team), body.team)
    return {"teams": ds.user_teams(store, scope(body.tenant, body.user, None))}


@app.get("/api/sessions")
def sessions(tenant: str, user: str, team: str | None = Query(None)):
    return {"sessions": [
        {"id": s.key, "title": s.value.get("title", s.key), "status": s.value.get("status", "new"),
         "updated_at": s.updated_at.isoformat()}
        for s in ds.list_sessions(store, scope(tenant, user, team))
    ]}


@app.get("/api/sessions/{thread_id}")
def session(thread_id: str, tenant: str, user: str, team: str | None = Query(None)):
    return checked(ds.load_chat, agents, store, scope(tenant, user, team), thread_id)


@app.get("/api/memories")
def memories(tenant: str, user: str, team: str | None = Query(None)):
    return ds.list_memories(store, scope(tenant, user, team))


@app.post("/api/turn")
def turn(body: Turn):
    """Run one turn and stream its events as Server-Sent Events."""
    tenant = scope(body.tenant, body.user, body.team)
    if body.answers:
        if not body.thread_id:
            raise HTTPException(400, "answers need a thread_id")
        payload = Command(resume=body.answers)
    elif body.message and body.message.strip():
        payload = {"messages": [{"role": "user", "content": body.message}]}
    else:
        raise HTTPException(400, "send a message or answers")
    thread_id = body.thread_id or ds.create_session(store, tenant, body.message)
    checked(ds.thread_config, store, tenant, thread_id)  # 403 before streaming starts
    with running_lock:
        if thread_id in running:
            raise HTTPException(409, "this session already has a turn running")
        running.add(thread_id)

    # The agent runs synchronously in its own thread (its own OTel context too); events cross over a queue
    events = queue.Queue()

    def work():
        try:
            pending = ds.run_turn(agents, store, tenant, thread_id, payload, on_event=events.put)
            events.put({"type": "pending", "items": [ds.interrupt_view(i) for i in pending]})
        except Exception as error:  # surface failures in the UI instead of a silently closed stream
            events.put({"type": "error", "message": f"{type(error).__name__}: {error}"})
        finally:
            with running_lock:
                running.discard(thread_id)
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def sse():
        yield f"data: {json.dumps({'type': 'session', 'id': thread_id})}\n\n"
        while (event := events.get()) is not None:
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


app.mount("/", StaticFiles(directory=Path(__file__).parent / "web", html=True), name="web")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
