"""Team chat rooms where agents are members (see room_session.py), with the web_room UI.

Runs on port 8001 by default so it can sit next to server.py / team_server.py. Uses its own Postgres schema
(room_session.SCHEMA), so it never reads or writes their data: team memberships and memories start empty here.
"""
import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from psycopg_pool import ConnectionPool
from pydantic import BaseModel

import room_session as rs
from server import JoinTeam, scope

pool = ConnectionPool(rs.DATABASE_URL, max_size=20, kwargs=rs.CONNECTION_KWARGS, open=False)
checkpointer = PostgresSaver(pool)
store = PostgresStore(pool)
rooms: rs.Rooms | None = None


@asynccontextmanager
async def lifespan(_app):
    global rooms
    tracer_provider = rs.setup_tracing()
    rs.ensure_schema()
    pool.open()
    checkpointer.setup()
    store.setup()
    rs.setup_tables(pool)
    rooms = rs.Rooms(pool, store, rs.build_room_agents(checkpointer, store))
    yield
    pool.close()
    tracer_provider.force_flush()


app = FastAPI(title="Team rooms", lifespan=lifespan)


def call(fn, *args):
    try:
        return fn(*args)
    except rs.RoomError as error:
        raise HTTPException(error.status, str(error)) from error


def room_view(r):
    return {"id": r["id"], "title": r["title"], "created_by": r["created_by"], "updated_at": r["updated_at"].isoformat()}


def message_view(m):
    return {"id": m["id"], "author": m["author"], "kind": m["kind"], "text": m["text"], "data": m["data"],
            "created_at": m["created_at"].isoformat()}


class Scoped(BaseModel):
    tenant: str
    user: str
    team: str


class NewRoom(Scoped):
    title: str


class Say(Scoped):
    text: str


class Answers(Scoped):
    agent: str
    answers: dict


@app.get("/api/agents")
def agents():
    return {"agents": [{"name": n, "about": about} for n, about in rs.AGENTS.items()]}


@app.get("/api/teams")
def teams(tenant: str, user: str):
    return {"teams": rs.user_teams(store, scope(tenant, user, None))}


@app.post("/api/teams")
def join(body: JoinTeam):
    rs.join_team(store, scope(body.tenant, body.user, body.team), body.team)
    return {"teams": rs.user_teams(store, scope(body.tenant, body.user, None))}


@app.get("/api/rooms")
def list_rooms(tenant: str, user: str, team: str | None = Query(None)):
    return {"rooms": [room_view(r) for r in call(rooms.list, scope(tenant, user, team))]}


@app.post("/api/rooms")
def create_room(body: NewRoom):
    return room_view(call(rooms.create, scope(body.tenant, body.user, body.team), body.title))


@app.post("/api/rooms/{room_id}/messages")
def say(room_id: str, body: Say):
    if not body.text.strip():
        raise HTTPException(400, "empty message")
    return message_view(call(rooms.say, scope(body.tenant, body.user, body.team), room_id, body.text.strip()))


@app.post("/api/rooms/{room_id}/answers")
def answer(room_id: str, body: Answers):
    call(rooms.answer, scope(body.tenant, body.user, body.team), room_id, body.agent, body.answers)
    return {"ok": True}


@app.get("/api/rooms/{room_id}/events")
async def events(request: Request, room_id: str, tenant: str, user: str, team: str, after: int = 0):
    """Server-Sent Events for one room: new messages, and each agent's live status/draft/pending questions.

    Polls Postgres twice a second, so it works across several server processes without extra infrastructure.
    """
    who = scope(tenant, user, team)
    await run_in_threadpool(call, rooms.room, who, room_id)

    async def stream():
        last, seen, idle = after, None, 0
        while not await request.is_disconnected():
            messages = await run_in_threadpool(rooms.messages_after, room_id, last)
            for m in messages:
                last = m["id"]
                yield f"data: {json.dumps({'type': 'message', **message_view(m)}, default=str)}\n\n"
            agents_now = await run_in_threadpool(rooms.agents_view, room_id)
            if agents_now != seen:
                seen = agents_now
                yield f"data: {json.dumps({'type': 'agents', 'agents': agents_now}, default=str)}\n\n"
            elif not messages:
                idle += 1
                if idle % 30 == 0:
                    yield ": ping\n\n"  # keeps proxies from closing a quiet stream
            if messages:
                continue  # more history may be waiting; fetch the next page right away
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


app.mount("/", StaticFiles(directory=Path(__file__).parent / "web_room", html=True), name="web")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("ROOM_PORT", "8001")))
