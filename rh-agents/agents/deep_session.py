import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.types import Command, interrupt
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.middleware.filesystem import FilesystemPermission
from openinference.instrumentation import using_attributes
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from phoenix.otel import register

from deep import (
    APPLICATION, BOLD, CYAN, DIM, ENVIRONMENT, RESET, YELLOW,
    internet_search, iter_events, model, print_events, research_instructions,
)

load_dotenv(override=True)

DATABASE_URL = os.environ["DATABASE_URL"]
PHOENIX_PROJECT = "deep-agents"
# no-op until setup_tracing() installs the Phoenix provider, so tests and imports don't export spans
tracer = trace.get_tracer(APPLICATION)


@dataclass
class Tenant:
    """Who a run is for and which team context is active. Passed as the runtime context of every run."""
    tenant_id: str  # the sector
    user_id: str
    team_id: str | None = None  # a person can belong to many teams; this is the one active right now


def tenant_ns(tenant: Tenant, *rest):
    return ("tenants", tenant.tenant_id, *rest)


def team_ns(tenant: Tenant, *rest):
    return tenant_ns(tenant, "teams", tenant.team_id, *rest)


def user_ns(tenant: Tenant, *rest):
    return tenant_ns(tenant, "users", tenant.user_id, *rest)


def user_teams(store, tenant):
    # Demo membership list. In production this comes from your identity provider / HR system.
    return sorted(item.key for item in store.search(user_ns(tenant, "teams"), limit=100))


def join_team(store, tenant, team_id):
    store.put(user_ns(tenant, "teams"), team_id, {"joined_at": datetime.now(timezone.utc).isoformat()})


@tool
def ask_user(question: str) -> str:
    """Ask the user one question and wait for their answer."""
    # Pauses the run; the checkpoint is saved to Postgres and the run ends here.
    # Any worker can later resume the thread with Command(resume=...).
    return interrupt({"question": question})


grill_instructions = research_instructions + """
## `ask_user`

Most requests need no questions: if the request is clear, factual or simple, or the memory layers already tell you what you need, just answer it.
Only ask when the topic is subtle and a wrong assumption would make the report miss the point — for example it is ambiguous, depends on the user's situation (legal, medical, financial, strategic decisions), or could reasonably go in very different directions.
In that case ask only what you can't reasonably infer, starting with the single most important question: one question per `ask_user` call, never several calls in parallel, at most 3 in total. Then research and write the report, stating any assumptions you made.

## Memory layers

Long-term memory has three layers, from broadest to most specific:
- `/sector/` — shared by everyone in the user's sector (organization-wide conventions, priorities, glossary).
- `/team/` — shared by the members of the currently active team (team goals, projects, decisions). Only available when a team is active.
- `/memories/` — private to this user (role, interests, preferences).

At the start of a session read `/sector/sector.md`, `/team/team.md` (if a team is active) and `/memories/user.md` when they exist. When they conflict, the more specific layer wins.
Save what you learn in the narrowest layer it belongs to, and never put personal details in `/team/` or `/sector/`.
"""


class ActiveContext(AgentMiddleware):
    """Appends the run's sector/team/user to the system prompt on every model call."""

    def wrap_model_call(self, request, handler):
        tenant = request.runtime.context
        team = f"team `{tenant.team_id}` (/team/ is that team's memory)" if tenant.team_id else "no team (/team/ is unavailable)"
        note = f"\n\n## Active context\n\nSector `{tenant.tenant_id}`, user `{tenant.user_id}`, {team}."
        system = [*request.system_message.content_blocks, {"type": "text", "text": note}] if request.system_message else note
        return handler(request.override(system_message=SystemMessage(content=system)))


def build_agent(checkpointer, store, with_team):
    """The agent itself holds no state: conversations live in the checkpointer, long-term files in the store."""
    routes = {
        "/sector/": StoreBackend(namespace=lambda rt: tenant_ns(rt.context, "memories")),
        "/memories/": StoreBackend(namespace=lambda rt: user_ns(rt.context, "memories")),
    }
    if with_team:
        routes["/team/"] = StoreBackend(namespace=lambda rt: team_ns(rt.context, "memories"))
    return create_deep_agent(
        model=model,
        tools=[internet_search, ask_user],
        system_prompt=grill_instructions,
        middleware=[ActiveContext()],
        checkpointer=checkpointer,
        store=store,
        # runtime.context is the Tenant passed to each run, so one agent serves every sector, team and user
        context_schema=Tenant,
        # first matching rule wins: without a team, /team/ is refused with a tool error instead of silently
        # landing in thread state; writes to the shared layers pause for the user's approval
        permissions=[
            *([] if with_team else [FilesystemPermission(operations=["read", "write"], paths=["/team", "/team/**"], mode="deny")]),
            FilesystemPermission(operations=["write"], paths=["/sector/**", "/team/**"], mode="interrupt"),
        ],
        # scratch files and todos are part of the thread state, so they're checkpointed too
        backend=CompositeBackend(default=StateBackend(), routes=routes),
    )


def build_agents(checkpointer, store):
    """Both variants share the same checkpointer and store; `agent_for` picks one per run."""
    return {True: build_agent(checkpointer, store, True), False: build_agent(checkpointer, store, False)}


def agent_for(agents, tenant):
    return agents[bool(tenant.team_id)]


def create_session(store, tenant, title):
    thread_id = str(uuid.uuid4())
    store.put(user_ns(tenant, "sessions"), thread_id, {
        "title": title[:60],
        # a session is bound to the team it started in: its history may hold that team's knowledge
        "team_id": tenant.team_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    return thread_id


def thread_config(store, tenant, thread_id):
    """Config for a thread, after checking it belongs to this user — thread_ids are global in the checkpointer."""
    session = store.get(user_ns(tenant, "sessions"), thread_id)
    if session is None:
        raise PermissionError(f"session {thread_id} does not belong to {tenant.tenant_id}/{tenant.user_id}")
    if session.value.get("team_id") != tenant.team_id:
        raise PermissionError(f"session {thread_id} belongs to team {session.value.get('team_id') or '-'}, not {tenant.team_id or '-'}")
    if tenant.team_id and tenant.team_id not in user_teams(store, tenant):
        raise PermissionError(f"{tenant.user_id} is not a member of team {tenant.team_id}")
    return {
        "configurable": {"thread_id": thread_id},
        # saved on every checkpoint, so checkpoints can be filtered or audited per sector, team and user
        "metadata": {"tenant_id": tenant.tenant_id, "team_id": tenant.team_id, "user_id": tenant.user_id},
    }


def describe_payload(payload):
    """The turn's input as Phoenix shows it: the user's message, or the answers that resumed the run."""
    if isinstance(payload, Command):
        return json.dumps(payload.resume, default=str)
    return payload["messages"][-1]["content"]


def setup_tracing():
    """Send traces to Phoenix (PHOENIX_COLLECTOR_ENDPOINT), one span tree per turn, grouped by session."""
    tracer_provider = register(project_name=PHOENIX_PROJECT, batch=True, verbose=False)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    return tracer_provider


def run_turn(agents, store, tenant, thread_id, payload, on_event=None):
    """One stateless unit of work: load the thread from Postgres, run until it finishes or interrupts, save.

    `payload` is a new user message (`{"messages": [...]}`) or `Command(resume=...)` answering pending questions.
    `on_event` receives each streamed event (see `deep.iter_events`); without it the turn is printed to the terminal.
    This is what a worker in a pool would run for an incoming (tenant, thread_id, payload) request.
    """
    config = thread_config(store, tenant, thread_id)
    agent = agent_for(agents, tenant)
    metadata = {**config["metadata"], "application": APPLICATION, "environment": ENVIRONMENT}
    tags = [APPLICATION, ENVIRONMENT, f"tenant_id:{tenant.tenant_id}", *([f"team_id:{tenant.team_id}"] if tenant.team_id else [])]
    # using_attributes stamps session/user/metadata/tags on every span the LangChain instrumentor creates;
    # the turn span carries them too, so Phoenix groups the turns of a thread into one session
    with using_attributes(session_id=thread_id, user_id=tenant.user_id, metadata=metadata, tags=tags), \
            tracer.start_as_current_span("research.turn", attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                SpanAttributes.SESSION_ID: thread_id,
                SpanAttributes.USER_ID: tenant.user_id,
                SpanAttributes.METADATA: json.dumps(metadata),
                SpanAttributes.TAG_TAGS: tags,
                SpanAttributes.INPUT_VALUE: describe_payload(payload),
            }) as span:
        events = iter_events(payload, config, agent=agent, context=tenant)
        if on_event:
            for event in events:
                on_event(event)
        else:
            print_events(events)
        state = agent.get_state(config)
        pending = state.interrupts
        if pending:
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps([i.value for i in pending], default=str))
            span.add_event("turn.interrupted", {"interrupts": len(pending)})
        else:
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, state.values["messages"][-1].text)
            span.add_event("turn.completed")
    index = store.get(user_ns(tenant, "sessions"), thread_id)
    store.put(user_ns(tenant, "sessions"), thread_id, {
        **index.value,
        "status": "waiting for you" if pending else "done",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    return pending


def interrupt_view(item):
    """A pending interrupt as plain data: an `ask_user` question or a HITL write approval."""
    if "action_requests" in item.value:
        return {"id": item.id, "kind": "approval", "actions": [
            {"name": a["name"], "file_path": a["args"].get("file_path"),
             "content": a["args"].get("content") or a["args"].get("new_string") or ""}
            for a in item.value["action_requests"]
        ]}
    return {"id": item.id, "kind": "question", "question": item.value["question"]}


def list_sessions(store, tenant):
    """The active team's sessions for this user, newest first."""
    sessions = (s for s in store.search(user_ns(tenant, "sessions"), limit=100) if s.value.get("team_id") == tenant.team_id)
    return sorted(sessions, key=lambda s: s.updated_at, reverse=True)


def load_chat(agents, store, tenant, thread_id):
    """A thread's messages for display: full detail (thinking, tool args/results) only for the last turn."""
    state = agent_for(agents, tenant).get_state(thread_config(store, tenant, thread_id))
    messages = state.values.get("messages", [])
    results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    chat = []
    for i, m in enumerate(messages):
        detailed = i > last_human
        if isinstance(m, HumanMessage):
            chat.append({"role": "user", "text": m.text})
        elif isinstance(m, AIMessage):
            if detailed and (thinking := m.additional_kwargs.get("reasoning_content")):
                chat.append({"role": "thinking", "text": thinking})
            for call in m.tool_calls:
                result = results.get(call["id"])
                chat.append({"role": "tool", "id": call["id"], "name": call["name"],
                             **({"args": call["args"], "result": str(result.content)[:4000] if result else None} if detailed else {})})
            if m.text:
                chat.append({"role": "assistant", "text": m.text})
    return {"messages": chat, "pending": [interrupt_view(i) for i in state.interrupts]}


def list_memories(store, tenant):
    """Every memory file visible from this context, grouped by layer (the folders the agent sees)."""
    layers = {"sector": ("/sector", tenant_ns(tenant, "memories")),
              "team": ("/team", team_ns(tenant, "memories") if tenant.team_id else None),
              "personal": ("/memories", user_ns(tenant, "memories"))}
    return {
        layer: [] if ns is None else [
            {"path": folder + item.key, "content": item.value.get("content", ""), "updated_at": item.updated_at.isoformat()}
            for item in store.search(ns, limit=100) if item.namespace == ns
        ]
        for layer, (folder, ns) in layers.items()
    }


def review(actions):
    """Approve or reject each pending write to a shared layer. Returns None if the user leaves."""
    decisions = []
    for action in actions:
        args = action["args"]
        body = args.get("content") or args.get("new_string") or ""
        print(f"\n{BOLD}{YELLOW}approve {action['name']} › {args.get('file_path')}{RESET}\n{DIM}{body[:600]}{RESET}")
        reply = input(f"{BOLD}y / n [reason] ›{RESET} ").strip()
        if not reply:
            return None
        if reply.lower() in ("y", "yes"):
            decisions.append({"type": "approve"})
        else:
            reason = reply[1:].strip() if reply.lower().startswith("n") else reply
            decisions.append({"type": "reject", **({"message": reason} if reason else {})})
    return {"decisions": decisions}


def answer(pending):
    """Collect an answer for each pending interrupt: an `ask_user` question or a write approval.

    Returns None if the user leaves (the interrupts stay pending in Postgres).
    """
    answers = {}
    for item in pending:
        if "action_requests" in item.value:  # HITLRequest from a FilesystemPermission(mode="interrupt") rule
            reply = review(item.value["action_requests"])
        else:
            print(f"\n{BOLD}{CYAN}agent asks ›{RESET} {item.value['question']}")
            reply = input(f"{BOLD}you ›{RESET} ").strip() or None
        if reply is None:
            return None
        answers[item.id] = reply
    return answers


def open_session(agents, store, tenant, thread_id=None):
    """Open an existing session, or start a new one when thread_id is None."""
    prompted = False
    while True:
        state = agent_for(agents, tenant).get_state(thread_config(store, tenant, thread_id)) if thread_id else None
        if state and state.interrupts:
            answers = answer(state.interrupts)
            if answers is None:
                return
            payload = Command(resume=answers)
        elif not prompted:
            if state and (messages := state.values.get("messages")):
                last = messages[-1].text.replace("\n", " ")
                print(f"\n{DIM}last answer: {last[:400]}{'…' if len(last) > 400 else ''}{RESET}")
            prompt = input(f"\n{BOLD}prompt ›{RESET} ").strip()
            if not prompt:
                return
            thread_id = thread_id or create_session(store, tenant, prompt)
            payload = {"messages": [{"role": "user", "content": prompt}]}
            prompted = True
        else:
            return
        run_turn(agents, store, tenant, thread_id, payload)


def pick_team(store, tenant):
    teams = user_teams(store, tenant)
    print(f"\n{BOLD}team{RESET}  {DIM}(number to pick, a new name to join it, - for no team, Enter keeps {tenant.team_id or '-'}){RESET}")
    for i, team in enumerate(teams, 1):
        print(f"  {i}) {team}{'  ← active' if team == tenant.team_id else ''}")
    choice = input(f"{BOLD}› {RESET}").strip()
    if choice == "-":
        tenant.team_id = None
    elif choice.isdigit() and 1 <= int(choice) <= len(teams):
        tenant.team_id = teams[int(choice) - 1]
    elif choice:
        join_team(store, tenant, choice)
        tenant.team_id = choice


def pick_session(store, tenant):
    """Lists the sessions of the active team. Returns a thread_id, "new", "team", or None to quit."""
    sessions = list_sessions(store, tenant)
    print(f"\n{BOLD}sessions ·{tenant.tenant_id}/{tenant.team_id or '-'}/{tenant.user_id}{RESET}")
    for i, s in enumerate(sessions, 1):
        print(f"  {i}) {s.value.get('title', s.key)}  {DIM}[{s.value.get('status', 'new')}] {s.updated_at:%Y-%m-%d %H:%M}{RESET}")
    print("  n) new session\n  t) switch team\n  q) quit")
    while True:
        choice = input(f"{BOLD}› {RESET}").strip().lower()
        if choice == "q":
            return None
        if choice == "n":
            return "new"
        if choice == "t":
            return "team"
        if choice.isdigit() and 1 <= int(choice) <= len(sessions):
            return sessions[int(choice) - 1].key


if __name__ == '__main__':
    tenant = Tenant(
        tenant_id=input(f"{BOLD}sector ›{RESET} ").strip() or "acme",
        user_id=input(f"{BOLD}user ›{RESET} ").strip() or "alice",
    )
    tracer_provider = setup_tracing()
    try:
        with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer, \
                PostgresStore.from_conn_string(DATABASE_URL) as store:
            checkpointer.setup()
            store.setup()
            agents = build_agents(checkpointer, store)
            tenant.team_id = next(iter(user_teams(store, tenant)), None)
            pick_team(store, tenant)
            while (choice := pick_session(store, tenant)) is not None:
                if choice == "team":
                    pick_team(store, tenant)
                else:
                    open_session(agents, store, tenant, None if choice == "new" else choice)
    finally:
        # spans are exported in batches; flush so the last turn shows up in Phoenix
        tracer_provider.force_flush()
