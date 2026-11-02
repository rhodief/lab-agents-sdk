"""Team chat rooms where agents are members.

People in a team talk in a room; an agent only acts when someone mentions it (`@agent`). It then reads every room
message posted since its last turn (each tagged with its author), does what was asked, and posts its reply to the
room. Its questions and write approvals are posted to the room too, and any member can answer them.

Everything lives in its own Postgres schema (SCHEMA), separate from the data of server.py and team_server.py:
the agents' checkpoints, the long-term store (memberships, memories) and the room tables.

Each agent keeps one LangGraph thread per room (`room:<room_id>:<agent>`), so several agents can share a room
without sharing their internal state (todos, files, pending interrupts). The room log is the source of truth.
"""
import json
import re
import threading
import time
import uuid

import psycopg
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.middleware.filesystem import FilesystemPermission
from langgraph.types import Command
from openinference.instrumentation import using_attributes
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from deep import APPLICATION, ENVIRONMENT, internet_search, iter_events, model
from deep_session import (  # noqa: F401  (re-exported for room_server)
    DATABASE_URL, ActiveContext, Tenant, ask_user, grill_instructions, interrupt_view, join_team,
    setup_tracing, team_ns, tenant_ns, tracer, user_ns, user_teams,
)
from team_session import SpeakerNames

SCHEMA = "rooms"
# every connection of the room server only sees the rooms schema
CONNECTION_KWARGS = {"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row,
                     "options": f"-c search_path={SCHEMA}"}

# The agents that can be members of a room: mention name -> what it does (shown to people and to the other agents).
# To add one, add an entry here (and, if it needs its own tools or prompt, a branch in build_room_agent).
AGENTS = {
    "agent": "the team's research assistant: web research, reports and the team's shared memory",
    "critic": "the teams's idea critic: analyse what peaple claim or propose and be honest about it, but keep kindness"
}
# how many times in a row agents may trigger each other by mentions before a person has to step in
MAX_AGENT_HOPS = 3
# how many of the latest unread messages an agent reads per turn (plus any older one that mentions it)
CONTEXT_LIMIT = 200
# `@name` not preceded by a word character, so e-mail addresses don't count
MENTION = re.compile(r"(?<![\w@])@([A-Za-z0-9_\-]+)")


def mentions(text):
    return {m.lower() for m in MENTION.findall(text or "") if m.lower() in AGENTS}


# ---------- agents ----------

def room_instructions(name):
    others = [f"`@{n}` ({d})" for n, d in AGENTS.items() if n != name]
    return grill_instructions + f"""
## You are `@{name}` in a team chat room

You are one member of a chat room shared by a team: {AGENTS[name]}.{" Other agents in the room: " + ", ".join(others) + "." if others else ""}
People talk among themselves in the room; you are only called when someone writes `@{name}`.
Each turn you receive the room messages posted since your last turn. Each starts with its author in brackets, like `[alice]` (the prefix is added for you, it's not part of what they typed).
Messages that don't mention you are context: read them, but act on what was asked of you, usually in the latest messages that mention you.
Your reply is posted to the room as a chat message. Match its length to the request: a short answer for a quick question, a full report only when one is asked for. Address people by name when several are involved.
`ask_user` posts your question to the room and any member may answer it, so say whom you're asking.
"Active context" names the person who called you; `/memories/` is their private memory, so never save details about other people there.
{"To hand part of the work to another agent, mention it in your reply." if others else ""}
"""


def build_room_agent(name, checkpointer, store):
    """Like team_session.build_agent, with the room prompt. Rooms always belong to a team, so /team/ is always on."""
    routes = {
        "/sector/": StoreBackend(namespace=lambda rt: tenant_ns(rt.context, "memories")),
        "/team/": StoreBackend(namespace=lambda rt: team_ns(rt.context, "memories")),
        "/memories/": StoreBackend(namespace=lambda rt: user_ns(rt.context, "memories")),
    }
    return create_deep_agent(
        model=model,
        tools=[internet_search, ask_user],
        system_prompt=room_instructions(name),
        middleware=[ActiveContext(), SpeakerNames()],
        checkpointer=checkpointer,
        store=store,
        context_schema=Tenant,
        permissions=[FilesystemPermission(operations=["write"], paths=["/sector/**", "/team/**"], mode="interrupt")],
        backend=CompositeBackend(default=StateBackend(), routes=routes),
    )


def build_room_agents(checkpointer, store):
    return {name: build_room_agent(name, checkpointer, store) for name in AGENTS}


# ---------- database ----------

def ensure_schema():
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")


TABLES = [
    """CREATE TABLE IF NOT EXISTS chat_rooms (
        id text PRIMARY KEY,
        tenant_id text NOT NULL,
        team_id text NOT NULL,
        title text NOT NULL,
        created_by text NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now())""",
    "CREATE INDEX IF NOT EXISTS chat_rooms_team ON chat_rooms (tenant_id, team_id, updated_at DESC)",
    # kind: user (a person), agent (an agent's reply), question (an agent's pending questions/approvals),
    # answer (someone answering them), system (notices and errors)
    """CREATE TABLE IF NOT EXISTS chat_messages (
        id bigserial PRIMARY KEY,
        room_id text NOT NULL REFERENCES chat_rooms (id) ON DELETE CASCADE,
        author text NOT NULL,
        kind text NOT NULL,
        text text NOT NULL DEFAULT '',
        data jsonb NOT NULL DEFAULT '{}',
        created_at timestamptz NOT NULL DEFAULT now())""",
    "CREATE INDEX IF NOT EXISTS chat_messages_room ON chat_messages (room_id, id)",
    # per agent and room: how far it has read, what it is doing right now, and its unanswered interrupts
    """CREATE TABLE IF NOT EXISTS chat_room_agents (
        room_id text NOT NULL REFERENCES chat_rooms (id) ON DELETE CASCADE,
        agent text NOT NULL,
        cursor bigint NOT NULL DEFAULT 0,
        status text,
        draft text,
        pending jsonb,
        PRIMARY KEY (room_id, agent))""",
]


def setup_tables(pool):
    with pool.connection() as conn:
        for statement in TABLES:
            conn.execute(statement)


class RoomLock:
    """One turn per agent and room at a time, across server processes: a Postgres advisory lock held on its own
    connection for the length of the turn. Closing the connection releases it, even if the process dies."""

    def __init__(self, room_id, agent):
        self.key = f"room:{room_id}:{agent}"
        self.conn = None

    def acquire(self):
        self.conn = psycopg.connect(DATABASE_URL, autocommit=True)
        if not self.conn.execute("SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (self.key,)).fetchone()[0]:
            self.release()
            return False
        return True

    def release(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None


class RoomError(Exception):
    """An error the API turns into an HTTP status."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def describe_answers(pending, answers):
    """Room text for someone answering an agent's pending questions or approvals."""
    lines = []
    for item in pending:
        reply = answers.get(item["id"])
        if item["kind"] == "question":
            lines.append(f"> {item['question']}\n\n{reply}")
        else:
            for action, decision in zip(item["actions"], (reply or {}).get("decisions", [])):
                verdict = "approved" if decision.get("type") == "approve" else "rejected"
                reason = f": {decision['message']}" if decision.get("message") else ""
                lines.append(f"{verdict} `{action['name']}` › `{action['file_path']}`{reason}")
    return "\n\n".join(lines)


# ---------- rooms ----------

class Rooms:
    """The rooms of every sector and team, and the agents working in them."""

    def __init__(self, pool, store, agents):
        self.pool, self.store, self.agents = pool, store, agents

    def sql(self, query, params=(), one=False):
        with self.pool.connection() as conn:
            cur = conn.execute(query, params)
            if cur.description is None:
                return None
            return cur.fetchone() if one else cur.fetchall()

    # access

    def room(self, tenant, room_id):
        """The room, after checking this user is a member of its team."""
        if not tenant.team_id:
            raise RoomError(400, "rooms belong to a team: pick one first")
        if tenant.team_id not in user_teams(self.store, tenant):
            raise RoomError(403, f"{tenant.user_id} is not a member of team {tenant.team_id}")
        room = self.sql("SELECT * FROM chat_rooms WHERE id = %s AND tenant_id = %s AND team_id = %s",
                        (room_id, tenant.tenant_id, tenant.team_id), one=True)
        if room is None:
            raise RoomError(404, f"no room {room_id} in team {tenant.team_id}")
        return room

    def list(self, tenant):
        if not tenant.team_id:
            return []
        if tenant.team_id not in user_teams(self.store, tenant):
            raise RoomError(403, f"{tenant.user_id} is not a member of team {tenant.team_id}")
        return self.sql("SELECT * FROM chat_rooms WHERE tenant_id = %s AND team_id = %s ORDER BY updated_at DESC LIMIT 200",
                        (tenant.tenant_id, tenant.team_id))

    def create(self, tenant, title):
        if not tenant.team_id:
            raise RoomError(400, "rooms belong to a team: pick one first")
        if tenant.team_id not in user_teams(self.store, tenant):
            raise RoomError(403, f"{tenant.user_id} is not a member of team {tenant.team_id}")
        room_id = str(uuid.uuid4())
        self.sql("INSERT INTO chat_rooms (id, tenant_id, team_id, title, created_by) VALUES (%s, %s, %s, %s, %s)",
                 (room_id, tenant.tenant_id, tenant.team_id, title.strip()[:80] or "untitled", tenant.user_id))
        self.post(room_id, "system", "system", f"{tenant.user_id} created the room. Mention "
                  + ", ".join(f"@{n}" for n in AGENTS) + " to ask an agent.")
        return self.room(tenant, room_id)

    # the log

    def post(self, room_id, author, kind, text, data=None):
        self.sql("UPDATE chat_rooms SET updated_at = now() WHERE id = %s", (room_id,))
        return self.sql("INSERT INTO chat_messages (room_id, author, kind, text, data) VALUES (%s, %s, %s, %s, %s) RETURNING *",
                        (room_id, author, kind, text, Jsonb(data or {})), one=True)

    def messages_after(self, room_id, after, limit=200):
        return self.sql("SELECT * FROM chat_messages WHERE room_id = %s AND id > %s ORDER BY id LIMIT %s",
                        (room_id, after, limit))

    def agent_state(self, room_id, name):
        self.sql("INSERT INTO chat_room_agents (room_id, agent) VALUES (%s, %s) ON CONFLICT DO NOTHING", (room_id, name))
        return self.sql("SELECT * FROM chat_room_agents WHERE room_id = %s AND agent = %s", (room_id, name), one=True)

    def set_agent(self, room_id, name, **fields):
        self.agent_state(room_id, name)
        values = {k: Jsonb(v) if k == "pending" and v is not None else v for k, v in fields.items()}
        assignments = ", ".join(f"{k} = %s" for k in values)  # keys come from this module, never from requests
        self.sql(f"UPDATE chat_room_agents SET {assignments} WHERE room_id = %s AND agent = %s",
                 (*values.values(), room_id, name))

    def agents_view(self, room_id):
        rows = {r["agent"]: r for r in self.sql("SELECT * FROM chat_room_agents WHERE room_id = %s", (room_id,))}
        return [{"name": n, "about": about, "status": rows.get(n, {}).get("status"), "draft": rows.get(n, {}).get("draft"),
                 "pending": rows.get(n, {}).get("pending")} for n, about in AGENTS.items()]

    # people talking

    def say(self, tenant, room_id, text):
        """Post someone's message; wake every agent it mentions."""
        self.room(tenant, room_id)
        message = self.post(room_id, tenant.user_id, "user", text)
        for name in sorted(mentions(text)):
            if self.agent_state(room_id, name)["pending"]:
                self.post(room_id, "system", "system",
                          f"@{name} is waiting for an answer to its question above; it will read this message right after.")
            self.wake(room_id, name, tenant.tenant_id, tenant.team_id)
        return message

    def answer(self, tenant, room_id, name, answers):
        """Answer an agent's pending questions/approvals and resume it."""
        self.room(tenant, room_id)
        if name not in AGENTS:
            raise RoomError(404, f"no agent @{name}")
        pending = self.agent_state(room_id, name)["pending"]
        if not pending:
            raise RoomError(409, f"@{name} has no pending question (someone may have answered it already)")
        if set(answers) != {item["id"] for item in pending}:
            raise RoomError(400, "answer every pending item")
        lock = RoomLock(room_id, name)
        if not lock.acquire():
            raise RoomError(409, f"@{name} is busy, try again in a moment")
        self.set_agent(room_id, name, pending=None)
        self.post(room_id, tenant.user_id, "answer", describe_answers(pending, answers), {"agent": name})
        self.wake(room_id, name, tenant.tenant_id, tenant.team_id, resume=(answers, tenant.user_id), lock=lock)

    # agents working

    def wake(self, room_id, name, tenant_id, team_id, resume=None, lock=None):
        threading.Thread(target=self.work, args=(room_id, name, tenant_id, team_id, resume, lock), daemon=True).start()

    def unread(self, room_id, name):
        """What this agent hasn't read yet: (messages to give it, the ones that call it, how many were skipped).

        Calls are looked for across *all* unread messages, so a mention is never missed in a busy room. The agent
        gets the latest CONTEXT_LIMIT unread messages plus every older one that calls it; the rest are skipped.
        """
        cursor = self.agent_state(room_id, name)["cursor"]
        unread = "FROM chat_messages WHERE room_id = %s AND id > %s AND author <> %s AND kind IN ('user', 'agent')"
        args = (room_id, cursor, name)
        total = self.sql(f"SELECT count(*) AS n {unread}", args, one=True)["n"]
        latest = self.sql(f"SELECT * {unread} ORDER BY id DESC LIMIT %s", (*args, CONTEXT_LIMIT))
        # ILIKE narrows the search in SQL (it may over-match: _ is a wildcard); mentions() decides
        candidates = self.sql(f"SELECT * {unread} AND text ILIKE %s ORDER BY id", (*args, f"%@{name}%"))
        calls = [m for m in candidates if name in mentions(m["text"])
                 and (m["kind"] == "user" or m["data"].get("hops", 0) < MAX_AGENT_HOPS)]
        messages = sorted({m["id"]: m for m in [*latest, *calls]}.values(), key=lambda m: m["id"])
        return messages, calls, total - len(messages)

    def work(self, room_id, name, tenant_id, team_id, resume=None, lock=None):
        """Run the agent until no unread message calls it. One worker per agent and room at a time (RoomLock);
        a message arriving while it works is picked up by the same worker, or by the re-check after it unlocks."""
        while True:
            if lock is None:
                lock = RoomLock(room_id, name)
                if not lock.acquire():
                    return  # another worker is on it and will see the new messages
            try:
                if resume is not None:
                    answers, by = resume
                    resume = None
                    self.turn(room_id, name, Tenant(tenant_id, by, team_id), Command(resume=answers), by, hops=1)
                while not self.agent_state(room_id, name)["pending"]:
                    messages, calls, skipped = self.unread(room_id, name)
                    if not calls:
                        break
                    last = calls[-1]
                    # an agent's message calls on behalf of the person who started the chain
                    by = last["author"] if last["kind"] == "user" else last["data"].get("for", last["author"])
                    hops = 0 if last["kind"] == "user" else last["data"].get("hops", 0)
                    payload = {"messages": [{"role": "user", "content": m["text"], "name": m["author"]} for m in messages]}
                    if skipped:  # say so, rather than let the agent think it has the whole conversation
                        payload["messages"].insert(0, {"role": "user", "name": "room", "content":
                            f"({skipped} older room messages since your last turn were skipped; below are the latest ones "
                            "and every message that mentions you.)"})
                    self.turn(room_id, name, Tenant(tenant_id, by, team_id), payload, by, hops=hops + 1,
                              read_upto=messages[-1]["id"])
            finally:
                lock.release()
                lock = None
            if self.agent_state(room_id, name)["pending"] or not self.unread(room_id, name)[1]:  # [1]: the calls
                return

    def turn(self, room_id, name, tenant, payload, by, hops, read_upto=None):
        """One agent turn: stream its progress into the room, then post its reply or its questions."""
        thread_id = f"room:{room_id}:{name}"
        agent = self.agents[name]
        metadata = {"tenant_id": tenant.tenant_id, "team_id": tenant.team_id, "user_id": tenant.user_id,
                    "room_id": room_id, "agent": name, "application": APPLICATION, "environment": ENVIRONMENT}
        config = {"configurable": {"thread_id": thread_id}, "metadata": metadata}
        tags = [APPLICATION, ENVIRONMENT, f"tenant_id:{tenant.tenant_id}", f"team_id:{tenant.team_id}", f"agent:{name}"]
        draft, status, flushed = "", "thinking", 0.0
        self.set_agent(room_id, name, status=status, draft="")
        if read_upto:  # advance first, so a failing turn isn't retried forever on the same messages
            self.set_agent(room_id, name, cursor=read_upto)
        try:
            with using_attributes(session_id=thread_id, user_id=tenant.user_id, metadata=metadata, tags=tags), \
                    tracer.start_as_current_span("room.turn", attributes={
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                        SpanAttributes.SESSION_ID: thread_id,
                        SpanAttributes.USER_ID: tenant.user_id,
                        SpanAttributes.METADATA: json.dumps(metadata),
                        SpanAttributes.TAG_TAGS: tags,
                    }):
                for event in iter_events(payload, config, agent=agent, context=tenant):
                    if event["type"] == "text" and event["agent"] is None:
                        draft += event["text"]
                    elif event["type"] == "tool_call":
                        status = f"using {event['name']}" if event["agent"] is None else f"{event['agent']}: using {event['name']}"
                    if time.monotonic() - flushed > 0.5:  # what everyone sees while it works
                        self.set_agent(room_id, name, status=status, draft=draft)
                        flushed = time.monotonic()
                state = agent.get_state(config)
        except Exception as error:
            self.set_agent(room_id, name, status=None, draft=None)
            self.post(room_id, "system", "system", f"@{name} failed: {type(error).__name__}: {error}")
            return
        if state.interrupts:
            items = [interrupt_view(i) for i in state.interrupts]
            text = "\n\n".join(i["question"] if i["kind"] == "question" else
                               "Wants to write " + ", ".join(f"`{a['file_path']}`" for a in i["actions"]) for i in items)
            self.set_agent(room_id, name, pending=items, status=None, draft=None)
            self.post(room_id, name, "question", text, {"items": items, "for": by})
            return
        reply = state.values["messages"][-1].text
        self.set_agent(room_id, name, status=None, draft=None)
        self.post(room_id, name, "agent", reply, {"hops": hops, "for": by})
        if hops < MAX_AGENT_HOPS:
            for other in sorted(mentions(reply) - {name}):
                self.wake(room_id, other, tenant.tenant_id, tenant.team_id)
