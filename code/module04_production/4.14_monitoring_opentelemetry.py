# Monitoring — OpenTelemetry Distributed Tracing
# pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
import time

from langchain_ollama import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler


# --- Setup: TracerProvider with console exporter (swap to OTLP for Jaeger/Grafana) ---

resource = Resource.create({"service.name": "agentic-ai-app"})
provider = TracerProvider(resource=resource)
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("module04.monitoring")


# --- OTel-instrumented LangChain callback ---

class OTelLLMLogger(BaseCallbackHandler):
    def on_llm_start(self, *args, **kwargs):
        self._span = tracer.start_span("llm.call")
        prompts = kwargs.get("prompts")
        if prompts:
            self._span.set_attribute("llm.prompt_preview", prompts[0][:200])
            self._span.set_attribute("llm.est_input_tokens", max(1, len(prompts[0]) // 4))

    def on_llm_end(self, *args, **kwargs):
        self._span.set_status(trace.StatusCode.OK)
        self._span.end()

    def on_llm_error(self, error, *args, **kwargs):
        self._span.set_status(trace.StatusCode.ERROR, str(error))
        self._span.record_exception(error)
        self._span.end()


# --- Trace any function with a span ---

def traced_tool_call(tool_name: str, payload: dict) -> str:
    with tracer.start_as_current_span("tool.call") as span:
        span.set_attribute("tool.name", tool_name)
        span.set_attribute("tool.payload", str(payload))
        # Simulate tool execution
        result = f"mock result for {tool_name}"
        span.set_attribute("tool.result_preview", result[:200])
        return result


if __name__ == "__main__":
    # Every LLM call now emits an OTel span (visible in console, or Jaeger/Grafana via OTLP)
    llm = ChatOllama(model="llama3.2", temperature=0.2, callbacks=[OTelLLMLogger()])

    with tracer.start_as_current_span("user.request") as parent:
        parent.set_attribute("user.query", "What is an agent?")
        resp = llm.invoke("In one line, what is an agent?")
        print(f"Response: {resp.content}")

        traced_tool_call("search", {"q": "LLM agents"})

    # Flush spans
    provider.force_flush()
