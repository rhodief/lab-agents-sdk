import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.types import Command, interrupt
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.middleware.filesystem import FilesystemPermission

from deep import BOLD, CYAN, DIM, RESET, internet_search, model, research_instructions, stream

load_dotenv(override=True)

DATABASE_URL = os.environ["DATABASE_URL"]


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
        # without a team, /team/ is refused with a tool error instead of silently landing in thread state
        permissions=None if with_team else [
            FilesystemPermission(operations=["read", "write"], paths=["/team", "/team/**"], mode="deny"),
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


def run_turn(agents, store, tenant, thread_id, payload):
    """One stateless unit of work: load the thread from Postgres, run until it finishes or interrupts, save.

    `payload` is a new user message (`{"messages": [...]}`) or `Command(resume=...)` answering pending questions.
    This is what a worker in a pool would run for an incoming (tenant, thread_id, payload) request.
    """
    config = thread_config(store, tenant, thread_id)
    agent = agent_for(agents, tenant)
    stream(payload, config, agent=agent, context=tenant)
    pending = agent.get_state(config).interrupts
    index = store.get(user_ns(tenant, "sessions"), thread_id)
    store.put(user_ns(tenant, "sessions"), thread_id, {
        **index.value,
        "status": "waiting for you" if pending else "done",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    return pending


def answer(pending):
    """Collect an answer for each pending question. Returns None if the user leaves (questions stay pending)."""
    answers = {}
    for item in pending:
        print(f"\n{BOLD}{CYAN}agent asks ›{RESET} {item.value['question']}")
        reply = input(f"{BOLD}you ›{RESET} ").strip()
        if not reply:
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
    sessions = sorted(
        (s for s in store.search(user_ns(tenant, "sessions"), limit=100) if s.value.get("team_id") == tenant.team_id),
        key=lambda s: s.updated_at, reverse=True,
    )
    print(f"\n{BOLD}sessions · {tenant.tenant_id}/{tenant.team_id or '-'}/{tenant.user_id}{RESET}")
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
