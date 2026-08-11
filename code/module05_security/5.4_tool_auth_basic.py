from typing import Callable


# token -> {user_id, scopes}. Add a token or scope here and watch access change.
TOKENS = {
    "token_viewer": {"user_id": "u1", "scopes": {"weather.read"}},
    "token_finance": {"user_id": "u2", "scopes": {"weather.read", "invoice.read"}},
}


def resolve_auth(token: str) -> dict:
    if token not in TOKENS:
        raise PermissionError("ACCESS_DENIED: invalid token")
    return TOKENS[token]


def authorize(auth: dict, required_scope: str) -> None:
    if required_scope not in auth["scopes"]:
        raise PermissionError(f"ACCESS_DENIED: missing scope '{required_scope}'")


def get_weather(city: str) -> str:
    return f"{city}: 24C, clear sky"


def get_invoice(invoice_id: str) -> str:
    return f"{invoice_id}: paid"


def call_tool(tool: Callable[..., str], required_scope: str, token: str, **kwargs) -> str:
    auth = resolve_auth(token)
    authorize(auth, required_scope)
    return tool(**kwargs)


if __name__ == "__main__":
    print("Tool auth demo")

    try:
        result = call_tool(get_weather, "weather.read", "token_viewer", city="Pune")
        print("Weather allowed:", result)
    except PermissionError as e:
        print("Weather denied:", e)

    try:
        result = call_tool(get_invoice, "invoice.read", "token_viewer", invoice_id="INV-1001")
        print("Invoice allowed:", result)
    except PermissionError as e:
        print("Invoice denied:", e)

    try:
        result = call_tool(get_invoice, "invoice.read", "token_finance", invoice_id="INV-1001")
        print("Invoice allowed:", result)
    except PermissionError as e:
        print("Invoice denied:", e)
