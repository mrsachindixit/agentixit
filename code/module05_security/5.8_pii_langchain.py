# PII Detection — LangChain (Reversible Anonymiser + PIIMiddleware)
# pip install langchain-experimental langchain-ollama presidio-analyzer presidio-anonymizer

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


# --- Reversible Anonymiser: anonymise → LLM → de-anonymise ---
# LLM never sees real PII; real values restored in final output

try:
    from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer
    ANONYMIZER_AVAILABLE = True
except ImportError:
    ANONYMIZER_AVAILABLE = False


def demo_reversible_anonymizer():
    if not ANONYMIZER_AVAILABLE:
        print("[SKIP] pip install langchain-experimental")
        return

    anonymizer = PresidioReversibleAnonymizer(
        languages_config={"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}]}
    )

    original = "Sachin Dixit's email is sachin@example.com and his phone is +91 98765-43210."
    anonymized_text = anonymizer.anonymize(original)
    print(f"Anonymised: {anonymized_text}")

    print("Mapping:")
    for entity_type, mapping in anonymizer.deanonymizer_mapping.items():
        for fake, real in mapping.items():
            print(f"  {entity_type:20s}  {fake} → {real}")

    llm = ChatOllama(model="llama3.1:latest", base_url="http://localhost:11434")
    llm_response = llm.invoke([HumanMessage(content=f"Summarise this record:\n\n{anonymized_text}")])
    print(f"LLM response (fake names): {llm_response.content}")

    # De-anonymise — swap fake names back to real
    restored = anonymizer.deanonymize(llm_response.content)
    print(f"De-anonymised            : {restored}")


# --- PIIMiddleware: declarative PII handling on agents ---

try:
    from langchain.agents import create_agent
    from langchain.agents.middleware import PIIMiddleware
    MIDDLEWARE_AVAILABLE = True
except ImportError:
    MIDDLEWARE_AVAILABLE = False


def demo_pii_middleware():
    if not MIDDLEWARE_AVAILABLE:
        print("[SKIP] PIIMiddleware not available in this langchain version")
        return

    llm = ChatOllama(model="llama3.1:latest", base_url="http://localhost:11434")
    agent = create_agent(
        model=llm,
        tools=[],
        middleware=[
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
            PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32,}", strategy="block", apply_to_input=True),
            PIIMiddleware("email", strategy="redact", apply_to_output=True),
        ],
    )

    test_input = "My email is sachin@example.com and my card is 4111-1111-1111-1111."
    print(f"User: {test_input}")
    result = agent.invoke({"messages": [HumanMessage(content=test_input)]})
    print(f"Agent: {result['messages'][-1].content}")


if __name__ == "__main__":
    print("=" * 60)
    print("Reversible Anonymiser (anonymise → LLM → de-anonymise)")
    print("=" * 60)
    demo_reversible_anonymizer()

    print()
    print("=" * 60)
    print("PIIMiddleware on a LangChain Agent")
    print("=" * 60)
    demo_pii_middleware()
