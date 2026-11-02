"""server.py with team-shared sessions (see team_session.py). Same API, served with the web_team UI; run instead of server.py.

Differences from server.py:
- a session started in a team is visible to, and can be continued by, every member of that team;
- session lists and chats include `created_by` and `last_user`;
- the one-turn-per-session lock is a Postgres advisory lock, so it holds across several server processes.
"""
import json
import queue
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

import team_session as ts
from server import JoinTeam, Turn, checked, scope

pool = ConnectionPool(
    ts.DATABASE_URL,
    max_size=10,
    kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    open=False,
)
checkpointer = PostgresSaver(pool)
store = PostgresStore(pool)
agents = None


@asynccontextmanager
async def lifespan(_app):
    global agents
    tracer_provider = ts.setup_tracing()
    pool.open()
    checkpointer.setup()
    store.setup()
    agents = ts.build_agents(checkpointer, store)
    yield
    pool.close()
    tracer_provider.force_flush()


app = FastAPI(title="Deep research agent (team sessions)", lifespan=lifespan)


@app.get("/api/teams")
def teams(tenant: str, user: str):
    return {"teams": ts.user_teams(store, scope(tenant, user, None))}


@app.post("/api/teams")
def join(body: JoinTeam):
    ts.join_team(store, scope(body.tenant, body.user, body.team), body.team)
    return {"teams": ts.user_teams(store, scope(body.tenant, body.user, None))}


@app.get("/api/sessions")
def sessions(tenant: str, user: str, team: str | None = Query(None)):
    return {"sessions": [
        {"id": s.key, "title": s.value.get("title", s.key), "status": s.value.get("status", "new"),
         "created_by": s.value.get("created_by"), "last_user": s.value.get("last_user"),
         "updated_at": s.updated_at.isoformat()}
        for s in checked(ts.list_sessions, store, scope(tenant, user, team))
    ]}


@app.get("/api/sessions/{thread_id}")
def session(thread_id: str, tenant: str, user: str, team: str | None = Query(None)):
    return checked(ts.load_chat, agents, store, scope(tenant, user, team), thread_id)


@app.get("/api/memories")
def memories(tenant: str, user: str, team: str | None = Query(None)):
    return ts.list_memories(store, scope(tenant, user, team))


@app.post("/api/turn")
def turn(body: Turn):
    """Run one turn and stream its events as Server-Sent Events."""
    tenant = scope(body.tenant, body.user, body.team)
    if body.answers:
        if not body.thread_id:
            raise HTTPException(400, "answers need a thread_id")
        payload = Command(resume=body.answers)
    elif body.message and body.message.strip():
        # name = the author; stored on the message so a shared session shows who said what
        payload = {"messages": [{"role": "user", "content": body.message, "name": tenant.user_id}]}
    else:
        raise HTTPException(400, "send a message or answers")
    thread_id = body.thread_id or checked(ts.create_session, store, tenant, body.message)
    checked(ts.thread_config, store, tenant, thread_id)  # 403 before streaming starts
    lock = ts.TurnLock(thread_id)
    if not lock.acquire():
        raise HTTPException(409, "this session already has a turn running")

    events = queue.Queue()

    def work():
        try:
            pending = ts.run_turn(agents, store, tenant, thread_id, payload, on_event=events.put)
            events.put({"type": "pending", "items": [ts.interrupt_view(i) for i in pending]})
        except Exception as error:
            events.put({"type": "error", "message": f"{type(error).__name__}: {error}"})
        finally:
            lock.release()
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def sse():
        yield f"data: {json.dumps({'type': 'session', 'id': thread_id})}\n\n"
        while (event := events.get()) is not None:
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


app.mount("/", StaticFiles(directory=Path(__file__).parent / "web_team", html=True), name="web")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
