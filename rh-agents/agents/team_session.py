"""Team-shared sessions, built on deep_session.

deep_session keeps every session in its author's private index (tenants/<t>/users/<u>/sessions), so only the
author can open it. Here a session started inside a team is indexed under the team instead
(tenants/<t>/teams/<team>/sessions), and any current member of that team can list, open and continue it.
Sessions started without a team stay private to their author, exactly as before.

Only the session index and its access check change; the agents, memory layers and tracing are reused as-is.
"""
import json
from datetime import datetime, timezone
import uuid

import psycopg
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.middleware.filesystem import FilesystemPermission
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from openinference.instrumentation import using_attributes
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

from deep import APPLICATION, ENVIRONMENT, internet_search, iter_events, model
from deep_session import (  # noqa: F401  (re-exported for team_server)
    DATABASE_URL, ActiveContext, Tenant, agent_for, ask_user, describe_payload, grill_instructions,
    interrupt_view, join_team, list_memories, setup_tracing, team_ns, tenant_ns, tracer, user_ns, user_teams,
)

team_instructions = grill_instructions + """
## Shared sessions

A session can be shared by a team, so several people may write in the same conversation.
Each user message starts with its author's name in brackets, like `[alice]`. That prefix is added for you and is not part of what they typed.
Keep track of who asked what and address people by name when it helps. "Active context" names who sent the latest message.
Save personal details only about that person to `/memories/`, never about other people in the conversation.
"""


def with_speaker(message):
    """The message as the model sees it: prefixed with its author, when it has one."""
    if not isinstance(message, HumanMessage) or not message.name:
        return message
    tag = f"[{message.name}] "
    if isinstance(message.content, str):
        content = tag + message.content
    else:
        content = [{"type": "text", "text": tag.strip()}, *message.content]
    return message.model_copy(update={"content": content})


class SpeakerNames(AgentMiddleware):
    """Shows the model who wrote each user message, without changing the stored messages.

    The author is kept in HumanMessage.name (which the UI reads), but not every provider forwards `name`,
    so it is also put in the text, only for the model call.
    """

    def wrap_model_call(self, request, handler):
        return handler(request.override(messages=[with_speaker(m) for m in request.messages]))


def build_agent(checkpointer, store, with_team):
    """deep_session.build_agent plus SpeakerNames and the shared-session instructions."""
    routes = {
        "/sector/": StoreBackend(namespace=lambda rt: tenant_ns(rt.context, "memories")),
        "/memories/": StoreBackend(namespace=lambda rt: user_ns(rt.context, "memories")),
    }
    if with_team:
        routes["/team/"] = StoreBackend(namespace=lambda rt: team_ns(rt.context, "memories"))
    return create_deep_agent(
        model=model,
        tools=[internet_search, ask_user],
        system_prompt=team_instructions,
        middleware=[ActiveContext(), SpeakerNames()],
        checkpointer=checkpointer,
        store=store,
        context_schema=Tenant,
        permissions=[
            *([] if with_team else [FilesystemPermission(operations=["read", "write"], paths=["/team", "/team/**"], mode="deny")]),
            FilesystemPermission(operations=["write"], paths=["/sector/**", "/team/**"], mode="interrupt"),
        ],
        backend=CompositeBackend(default=StateBackend(), routes=routes),
    )


def build_agents(checkpointer, store):
    return {True: build_agent(checkpointer, store, True), False: build_agent(checkpointer, store, False)}


def sessions_ns(tenant):
    """Where the active context's sessions are indexed: the team's index, or the user's own without a team."""
    return team_ns(tenant, "sessions") if tenant.team_id else user_ns(tenant, "sessions")


def require_member(store, tenant):
    if tenant.team_id and tenant.team_id not in user_teams(store, tenant):
        raise PermissionError(f"{tenant.user_id} is not a member of team {tenant.team_id}")


def create_session(store, tenant, title):
    require_member(store, tenant)
    thread_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    store.put(sessions_ns(tenant), thread_id, {
        "title": title[:60],
        "team_id": tenant.team_id,
        "created_by": tenant.user_id,
        "last_user": tenant.user_id,
        "created_at": now,
    })
    return thread_id


def get_session(store, tenant, thread_id):
    """The session's index entry, after checking this user may open it.

    thread_ids are global in the checkpointer, so this check is what keeps one team out of another's threads.
    """
    require_member(store, tenant)  # checked first: leaving a team revokes access to all of its sessions
    session = store.get(sessions_ns(tenant), thread_id)
    if session is None:
        where = f"team {tenant.team_id}" if tenant.team_id else f"{tenant.tenant_id}/{tenant.user_id}"
        raise PermissionError(f"session {thread_id} does not belong to {where}")
    return session


def thread_config(store, tenant, thread_id):
    session = get_session(store, tenant, thread_id)
    return {
        "configurable": {"thread_id": thread_id},
        # user_id is whoever ran this turn; created_by is the session's author
        "metadata": {"tenant_id": tenant.tenant_id, "team_id": tenant.team_id, "user_id": tenant.user_id,
                     "created_by": session.value.get("created_by", tenant.user_id)},
    }


def list_sessions(store, tenant):
    """The active context's sessions, newest first: the whole team's when a team is active."""
    require_member(store, tenant)
    sessions = (s for s in store.search(sessions_ns(tenant), limit=100) if s.value.get("team_id") == tenant.team_id)
    return sorted(sessions, key=lambda s: s.updated_at, reverse=True)


class TurnLock:
    """One turn per session at a time, across every server process.

    A session-level Postgres advisory lock on its own connection (not the pool's, so a long turn doesn't
    hold a pool slot). Closing the connection releases the lock even if the process dies mid-turn.
    """

    def __init__(self, thread_id):
        self.thread_id = thread_id
        self.conn = None

    def acquire(self):
        self.conn = psycopg.connect(DATABASE_URL, autocommit=True)
        row = self.conn.execute("SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (f"turn:{self.thread_id}",)).fetchone()
        if not row[0]:
            self.release()
            return False
        return True

    def release(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None


def run_turn(agents, store, tenant, thread_id, payload, on_event):
    """deep_session.run_turn against the shared index: same tracing, plus who ran the turn."""
    config = thread_config(store, tenant, thread_id)
    agent = agent_for(agents, tenant)
    metadata = {**config["metadata"], "application": APPLICATION, "environment": ENVIRONMENT}
    tags = [APPLICATION, ENVIRONMENT, f"tenant_id:{tenant.tenant_id}", *([f"team_id:{tenant.team_id}"] if tenant.team_id else [])]
    with using_attributes(session_id=thread_id, user_id=tenant.user_id, metadata=metadata, tags=tags), \
            tracer.start_as_current_span("research.turn", attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                SpanAttributes.SESSION_ID: thread_id,
                SpanAttributes.USER_ID: tenant.user_id,
                SpanAttributes.METADATA: json.dumps(metadata),
                SpanAttributes.TAG_TAGS: tags,
                SpanAttributes.INPUT_VALUE: describe_payload(payload),
            }) as span:
        for event in iter_events(payload, config, agent=agent, context=tenant):
            on_event(event)
        state = agent.get_state(config)
        pending = state.interrupts
        if pending:
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps([i.value for i in pending], default=str))
            span.add_event("turn.interrupted", {"interrupts": len(pending)})
        else:
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, state.values["messages"][-1].text)
            span.add_event("turn.completed")
    index = store.get(sessions_ns(tenant), thread_id)
    store.put(sessions_ns(tenant), thread_id, {
        **index.value,
        # any team member can answer a pending question or approval, so "you" is whoever opens it
        "status": "waiting for you" if pending else "done",
        "last_user": tenant.user_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    return pending


def load_chat(agents, store, tenant, thread_id):
    """deep_session.load_chat against the shared index, plus the session's author and last user."""
    session = get_session(store, tenant, thread_id)
    state = agent_for(agents, tenant).get_state(thread_config(store, tenant, thread_id))
    messages = state.values.get("messages", [])
    results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    chat = []
    for i, m in enumerate(messages):
        detailed = i > last_human
        if isinstance(m, HumanMessage):
            # messages sent before authors were recorded fall back to the session's author
            chat.append({"role": "user", "text": m.text, "author": m.name or session.value.get("created_by")})
        elif isinstance(m, AIMessage):
            if detailed and (thinking := m.additional_kwargs.get("reasoning_content")):
                chat.append({"role": "thinking", "text": thinking})
            for call in m.tool_calls:
                result = results.get(call["id"])
                chat.append({"role": "tool", "id": call["id"], "name": call["name"],
                             **({"args": call["args"], "result": str(result.content)[:4000] if result else None} if detailed else {})})
            if m.text:
                chat.append({"role": "assistant", "text": m.text})
    return {"messages": chat, "pending": [interrupt_view(i) for i in state.interrupts],
            "created_by": session.value.get("created_by"), "last_user": session.value.get("last_user")}
