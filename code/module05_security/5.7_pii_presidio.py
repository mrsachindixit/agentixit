# PII Detection — Microsoft Presidio (spaCy + Transformer NER)
# pip install presidio-analyzer presidio-anonymizer
# python -m spacy download en_core_web_lg

from utils.ollama_client import chat

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
except ImportError as exc:
    raise SystemExit(
        "pip install presidio-analyzer presidio-anonymizer && python -m spacy download en_core_web_lg"
    ) from exc

_spacy_analyzer = AnalyzerEngine()
_anonymizer = AnonymizerEngine()

OPERATORS = {
    "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
    "PERSON": OperatorConfig("replace", {"new_value": "[NAME_REDACTED]"}),
    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL_REDACTED]"}),
    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE_REDACTED]"}),
    "CREDIT_CARD": OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": 12, "from_end": False}),
}


def redact_spacy(text: str, language: str = "en") -> str:
    results = _spacy_analyzer.analyze(text=text, language=language)
    return _anonymizer.anonymize(text=text, analyzer_results=results, operators=OPERATORS).text


# --- Optional: swap spaCy NER for a HuggingFace transformer model ---
# pip install "presidio-analyzer[transformers]" && python -m spacy download en_core_web_sm

TRANSFORMERS_AVAILABLE = False
_transformer_analyzer = None

try:
    from presidio_analyzer.nlp_engine import TransformersNlpEngine

    _transformer_model_config = [{
        "lang_code": "en",
        "model_name": {
            "spacy": "en_core_web_sm",              # tokeniser only
            "transformers": "dslim/bert-base-NER",   # NER model
        },
    }]
    _transformer_nlp_engine = TransformersNlpEngine(models=_transformer_model_config)
    _transformer_analyzer = AnalyzerEngine(nlp_engine=_transformer_nlp_engine)
    TRANSFORMERS_AVAILABLE = True
except Exception:
    pass


def redact_transformer(text: str, language: str = "en") -> str:
    if not TRANSFORMERS_AVAILABLE or _transformer_analyzer is None:
        return redact_spacy(text, language)
    results = _transformer_analyzer.analyze(text=text, language=language)
    return _anonymizer.anonymize(text=text, analyzer_results=results, operators=OPERATORS).text


def redact(text: str) -> str:
    return redact_transformer(text) if TRANSFORMERS_AVAILABLE else redact_spacy(text)


if __name__ == "__main__":
    sample = (
        "Email me at sachin@example.com and call +91 98765-43210. "
        "My SSN is 123-45-6789 and card 4111-1111-1111-1111. "
        "My name is Sachin Dixit and I live at 42 Elm Street."
    )

    print("Presidio + spaCy NER:")
    print(redact_spacy(sample))
    print("\nDetected entities (spaCy):")
    for r in _spacy_analyzer.analyze(text=sample, language="en"):
        print(f"  {r.entity_type:20s}  score={r.score:.2f}  '{sample[r.start:r.end]}'")

    if TRANSFORMERS_AVAILABLE:
        print("\nPresidio + Transformer NER:")
        print(redact_transformer(sample))
        print("\nDetected entities (transformer):")
        for r in _transformer_analyzer.analyze(text=sample, language="en"):
            print(f"  {r.entity_type:20s}  score={r.score:.2f}  '{sample[r.start:r.end]}'")

    print("\nLLM call with Presidio guard:")
    user_msg = "My name is Sachin Dixit, email sachin@example.com — summarise my account."
    safe_input = redact(user_msg)
    print(f"  User (redacted): {safe_input}")
    messages = [
        {"role": "system", "content": "Never reveal personal data."},
        {"role": "user", "content": safe_input},
    ]
    response = chat(messages, temperature=0.2)
    print(f"  LLM  (scanned) : {redact(response)}")
