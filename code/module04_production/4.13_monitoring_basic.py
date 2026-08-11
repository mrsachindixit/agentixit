# Monitoring — Hand-Rolled JSON Logging

import json
import logging
import time
import uuid
from langchain_ollama import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # ~4 chars per token for English


class SimpleLogger(BaseCallbackHandler):
    def on_llm_start(self, *args, **kwargs):
        self.start = time.time()
        self.req_id = str(uuid.uuid4())
        prompts = kwargs.get("prompts")
        prompt_preview = prompts[0][:200] if prompts else ""
        log_entry = {
            "event": "llm_start",
            "req_id": self.req_id,
            "prompt_preview": prompt_preview,
            "est_input_tokens": estimate_tokens(prompts[0]) if prompts else 0,
        }
        logging.info(json.dumps(log_entry))

    def on_llm_end(self, *args, **kwargs):
        elapsed = time.time() - self.start
        log_entry = {"event": "llm_end", "req_id": self.req_id, "elapsed_s": round(elapsed, 3)}
        logging.info(json.dumps(log_entry))

    def on_llm_error(self, *args, **kwargs):
        log_entry = {"event": "llm_error", "req_id": getattr(self, "req_id", "unknown")}
        logging.exception(json.dumps(log_entry))


def log_tool_call(tool_name: str, payload: dict, result: str):
    logging.info("tool=%s payload=%s result_preview=%s", tool_name, payload, result[:200])


if __name__ == "__main__":
    llm = ChatOllama(model="llama3.2", temperature=0.2, callbacks=[SimpleLogger()])
    resp = llm.invoke("In one line, what is an agent?")
    logging.info("Response: %s", resp.content)

    log_tool_call("mock_tool", {"q": "agents"}, "ok")
