import os
import time
from collections import Counter
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.config import get_config
from tavily import TavilyClient
from tavily.errors import BadRequestError, TimeoutError as TavilyTimeoutError
from deepagents import create_deep_agent

load_dotenv(override=True)

# Gemini served through the LiteLLM proxy (same setup as index.py)
MODEL = "gemini-3.8-flash"
APPLICATION = "research-agent"
ENVIRONMENT = "development"
PINK = "\033[95m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


class ChatLiteLLM(ChatOpenAI):
    """ChatOpenAI that keeps the proxy's non-standard `reasoning_content` deltas and tags each call for LiteLLM."""

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        try:
            config = get_config()  # the running graph's config: thread_id plus the run's metadata
        except RuntimeError:
            return payload  # called outside an agent run
        metadata = config.get("metadata", {})
        session_id = config.get("configurable", {}).get("thread_id")
        if user_id := metadata.get("user_id"):
            payload["user"] = user_id
        # Same enrichment as index.py: LiteLLM uses it for spend tracking, logs and per-session correlation
        payload["extra_body"] = {
            **payload.get("extra_body", {}),
            "metadata": {
                "session_id": session_id,
                "tenant_id": metadata.get("tenant_id"),
                "team_id": metadata.get("team_id"),
                "application": APPLICATION,
                "langgraph_node": metadata.get("langgraph_node"),
            },
            "tags": [APPLICATION, ENVIRONMENT, *(f"{k}:{metadata[k]}" for k in ("tenant_id", "team_id") if metadata.get(k))],
        }
        return payload

    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        choices = chunk.get("choices") or []
        if generation_chunk and choices and choices[0].get("delta"):
            if reasoning := choices[0]["delta"].get("reasoning_content"):
                generation_chunk.message.additional_kwargs["reasoning_content"] = reasoning
        return generation_chunk


model = ChatLiteLLM(
    model=MODEL,
    base_url="http://litellm:4000/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    # gemini only streams reasoning_content when reasoning_effort is set
    reasoning_effort="medium",
    stream_usage=True,
)

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    try:
        return tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
    except (BadRequestError, TavilyTimeoutError) as error:
        # Hand fixable errors back to the model so it can rephrase or retry instead of crashing the run
        return {"error": f"{type(error).__name__}: {error}. Adjust the query and try again."}


# System prompt to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
"""

agent = create_deep_agent(
    model=model,
    tools=[internet_search],
    system_prompt=research_instructions,
)


def iter_events(input, config=None, agent=agent, context=None):
    """Run the agent and yield plain event dicts: thinking/text deltas, tool calls, tool results, then stats.

    `input` is a query string, an input dict, or a `Command(resume=...)`. Consumers (terminal, SSE) only render.
    """
    if isinstance(input, str):
        input = {"messages": [{"role": "user", "content": input}]}
    stats = Counter()
    tool_calls = Counter()
    started = time.perf_counter()

    # subgraphs=True also streams the subagents spawned through the `task` tool
    for namespace, (msg, _metadata) in agent.stream(
        input,
        config,
        context=context,
        stream_mode="messages",
        subgraphs=True,
    ):
        where = namespace[-1].split(":")[0] if namespace else None

        if isinstance(msg, AIMessageChunk):
            if reasoning := msg.additional_kwargs.get("reasoning_content"):
                yield {"type": "thinking", "agent": where, "text": reasoning}
            if msg.text:
                yield {"type": "text", "agent": where, "text": msg.text}
            for tc in msg.tool_call_chunks:
                if tc.get("name"):
                    tool_calls[tc["name"]] += 1
                    yield {"type": "tool_call", "agent": where, "index": tc.get("index"), "id": tc.get("id"), "name": tc["name"]}
                if tc.get("args"):
                    yield {"type": "tool_args", "agent": where, "index": tc.get("index"), "args": tc["args"]}
            if msg.usage_metadata:
                usage = msg.usage_metadata
                stats["llm_calls"] += 1
                stats["input_tokens"] += usage.get("input_tokens", 0)
                stats["output_tokens"] += usage.get("output_tokens", 0)
                stats["reasoning_tokens"] += (usage.get("output_token_details") or {}).get("reasoning", 0)
        elif isinstance(msg, ToolMessage):
            yield {"type": "tool_result", "agent": where, "tool_call_id": msg.tool_call_id, "name": msg.name,
                   "content": str(msg.content)[:4000]}

    yield {"type": "stats", "seconds": round(time.perf_counter() - started, 1), **stats,
           "tool_calls": dict(tool_calls.most_common())}


def print_events(events):
    """Terminal renderer for `iter_events`: one colored section per kind of output."""
    section = None  # reprint a header when the kind of output changes

    def enter(name, color, label):
        nonlocal section
        if section != name:
            print(f"{RESET}\n\n{BOLD}{color}[{label}]{RESET}{color}", end="\n", flush=True)
            section = name
        else:
            print(color, end="")

    for event in events:
        where = f" · {event['agent']}" if event.get("agent") else ""
        kind = event["type"]
        if kind == "thinking":
            enter(f"thinking{where}", PINK, f"thinking{where}")
            print(event["text"], end="", flush=True)
        elif kind == "text":
            enter(f"text{where}", RESET, f"text{where}")
            print(event["text"], end="", flush=True)
        elif kind == "tool_call":
            section = None
            enter(f"tool{where}", YELLOW, f"tool call{where}")
            print(f"{event['name']}(", end="")
        elif kind == "tool_args":
            print(f"{YELLOW}{event['args']}", end="", flush=True)
        elif kind == "tool_result":
            enter(f"result{where}", CYAN, f"tool result · {event['name']}{where}")
            content = event["content"].replace("\n", " ")
            print(f"{DIM}{content[:300]}{'…' if len(content) > 300 else ''}", end="", flush=True)
        elif kind == "stats":
            tool_calls = event["tool_calls"]
            print(f"{RESET}\n\n{BOLD}[stats]{RESET}")
            print(f"  time        {event['seconds']:.1f}s")
            print(f"  llm calls   {event.get('llm_calls', 0)}")
            print(f"  tokens      {event.get('input_tokens', 0)} in / {event.get('output_tokens', 0)} out"
                  f" ({event.get('reasoning_tokens', 0)} reasoning)")
            print(f"  tool calls  {sum(tool_calls.values())}"
                  + (" — " + ", ".join(f"{n}×{c}" for n, c in tool_calls.items()) if tool_calls else ""))


def stream(input, config=None, agent=agent, context=None):
    """Stream one run to the terminal."""
    print_events(iter_events(input, config, agent=agent, context=context))


if __name__ == '__main__':
    stream("What is langgraph?")
