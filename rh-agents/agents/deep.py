import os
import time
from collections import Counter
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from tavily.errors import BadRequestError, TimeoutError as TavilyTimeoutError
from deepagents import create_deep_agent

load_dotenv(override=True)

# Gemini served through the LiteLLM proxy (same setup as index.py)
MODEL = "gemini-3.8-flash"
PINK = "\033[95m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


class ChatLiteLLM(ChatOpenAI):
    """ChatOpenAI that keeps the proxy's non-standard `reasoning_content` deltas."""

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


def stream(input, config=None, agent=agent, context=None):
    """Stream one run. `input` is a query string, an input dict, or a `Command(resume=...)`."""
    if isinstance(input, str):
        input = {"messages": [{"role": "user", "content": input}]}
    stats = Counter()
    tool_calls = Counter()
    section = None  # "thinking" | "text" | "tool" — reprint a header when it changes
    started = time.perf_counter()

    def enter(name, color, label):
        nonlocal section
        if section != name:
            print(f"{RESET}\n\n{BOLD}{color}[{label}]{RESET}{color}", end="\n", flush=True)
            section = name
        else:
            print(color, end="")

    # subgraphs=True also streams the subagents spawned through the `task` tool
    for namespace, (msg, _metadata) in agent.stream(
        input,
        config,
        context=context,
        stream_mode="messages",
        subgraphs=True,
    ):
        where = f" · {namespace[-1].split(':')[0]}" if namespace else ""

        if isinstance(msg, AIMessageChunk):
            if reasoning := msg.additional_kwargs.get("reasoning_content"):
                enter(f"thinking{where}", PINK, f"thinking{where}")
                print(reasoning, end="", flush=True)
            if msg.text:
                enter(f"text{where}", RESET, f"text{where}")
                print(msg.text, end="", flush=True)
            for tc in msg.tool_call_chunks:
                if tc.get("name"):
                    tool_calls[tc["name"]] += 1
                    section = None
                    enter(f"tool{where}", YELLOW, f"tool call{where}")
                    print(f"{tc['name']}(", end="")
                if tc.get("args"):
                    print(f"{YELLOW}{tc['args']}", end="", flush=True)
            if msg.usage_metadata:
                usage = msg.usage_metadata
                stats["llm_calls"] += 1
                stats["input_tokens"] += usage.get("input_tokens", 0)
                stats["output_tokens"] += usage.get("output_tokens", 0)
                stats["reasoning_tokens"] += (usage.get("output_token_details") or {}).get("reasoning", 0)
        elif isinstance(msg, ToolMessage):
            enter(f"result{where}", CYAN, f"tool result · {msg.name}{where}")
            content = str(msg.content).replace("\n", " ")
            print(f"{DIM}{content[:300]}{'…' if len(content) > 300 else ''}", end="", flush=True)

    elapsed = time.perf_counter() - started
    print(f"{RESET}\n\n{BOLD}[stats]{RESET}")
    print(f"  time        {elapsed:.1f}s")
    print(f"  llm calls   {stats['llm_calls']}")
    print(f"  tokens      {stats['input_tokens']} in / {stats['output_tokens']} out"
          f" ({stats['reasoning_tokens']} reasoning)")
    print(f"  tool calls  {sum(tool_calls.values())}"
          + (" — " + ", ".join(f"{n}×{c}" for n, c in tool_calls.most_common()) if tool_calls else ""))


if __name__ == '__main__':
    stream("What is langgraph?")
