import asyncio
from openai import AsyncOpenAI
from agents import ModelSettings, set_default_openai_client
import os
from dotenv import load_dotenv
from phoenix.otel import register
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from openai.types.responses import ResponseTextDeltaEvent
from httpx import HTTPError
from opentelemetry import trace
from phoenix.client import Client as PhoenixClient

load_dotenv(override=True)

 # configure the Phoenix tracer
tracer_provider = register(
  project_name="agents", # Default is 'default'
    auto_instrument=False,
  set_global_tracer_provider=False,
  batch=True,
  verbose=False
)
OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = tracer_provider.get_tracer("judiciary-agent")

client = AsyncOpenAI(
    base_url="http://litellm:4000/v1",
    api_key=os.getenv('OPENAI_API_KEY'),
)

set_default_openai_client(client)

from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner

#MODEL = "gemini-3.8-flash"
MODEL = "gpt-5.6-luna"
USER_ID = "user-id-teste"
SESSION_ID = "session-001"
TENANT_ID = "tenant-demo"
ENVIRONMENT = "development"
PHOENIX_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:6006")

async def main():
    agent = Agent(
        name="Judiciary Agent",
        model=OpenAIChatCompletionsModel(model=MODEL, openai_client=client),
        model_settings=ModelSettings(
            extra_args={"user": USER_ID},
            extra_body={
                "metadata": {
                    "session_id": SESSION_ID,
                    "tenant_id": TENANT_ID,
                    "application": "judiciary-agent",
                },
                "tags": ["judiciary", ENVIRONMENT, "pt-br"],
            },
        ),
        instructions="Your are a Judiciary helper. Answare the question in a very-polite-lawer-fashion"
    )

    run_config = RunConfig(
        workflow_name="judiciary-assistant",
        group_id=SESSION_ID,
        trace_metadata={
            "user_id": USER_ID,
            "session_id": SESSION_ID,
            "tenant_id": TENANT_ID,
            "environment": ENVIRONMENT,
            "channel": "cli",
            "locale": "pt-BR",
            "use_case": "legal-question",
        },
    )

    with tracer.start_as_current_span("judiciary.request") as request_span:
        request_span.set_attributes(
            {
                "app.user_id": USER_ID,
                "app.session_id": SESSION_ID,
                "app.tenant_id": TENANT_ID,
                "app.environment": ENVIRONMENT,
                "app.use_case": "legal-question",
            }
        )
        request_span.add_event("request.classified", {"category": "animal-law"})

        result = Runner.run_streamed(
            agent,
            input="Eu mordi um cachorro, quais são minhas chances de ser preso?",
            run_config=run_config,
        )

        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)

        request_span.set_attribute("app.response_status", "completed")
        request_span.add_event("response.completed")
        span_context = request_span.get_span_context()
        otel_span_id = format(span_context.span_id, "016x")
        trace_id = format(span_context.trace_id, "032x")

    tracer_provider.force_flush()
    phoenix = PhoenixClient(base_url=PHOENIX_ENDPOINT)
    try:
        span = None
        for _ in range(10):
            phoenix_spans = phoenix.spans.get_spans(
                project_identifier="agents",
                trace_ids=[trace_id],
                limit=100,
            )
            span = next(
                (
                    span
                    for span in phoenix_spans
                    if span["context"]["span_id"] == otel_span_id
                ),
                None,
            )
            if span is not None:
                break
            await asyncio.sleep(0.2)
        if span is None:
            raise LookupError(otel_span_id)
        # Annotation writes use the OpenTelemetry span ID, not Phoenix's internal record ID.
        span_id = otel_span_id
        phoenix.spans.add_span_note(
            span_id=span_id,
            note="Demo review: inspect latency, model usage, and legal-answer quality.",
        )
        phoenix.spans.add_span_annotation(
            span_id=span_id,
            annotation_name="demo-quality",
            annotator_kind="CODE",
            label="review-needed",
            score=0.8,
            explanation="Demo annotation attached after the streamed response completed.",
            metadata={"application": "judiciary-agent", "environment": ENVIRONMENT},
            sync=True,
        )
        print(f"\nPhoenix annotation added to span {span_id}")
    except LookupError:
        print(f"\nPhoenix span not available for annotation: {otel_span_id}")
    except HTTPError as error:
        print(f"\nPhoenix annotation API unavailable: {error.response.status_code}")

    
    
if __name__ == '__main__':
    asyncio.run(main())