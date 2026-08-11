import os
import importlib


# token -> {user_id, scopes}. Same auth map as 5.4, now behind MCP tools.
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


def get_weather(auth_token: str, city: str) -> str:
    auth = resolve_auth(auth_token)
    authorize(auth, "weather.read")
    return f"{city}: 24C, clear sky"


def get_invoice(auth_token: str, invoice_id: str) -> str:
    auth = resolve_auth(auth_token)
    authorize(auth, "invoice.read")
    return f"{invoice_id}: paid"


def build_mcp_server():
    FastMCP = importlib.import_module("mcp.server.fastmcp").FastMCP
    mcp = FastMCP("mcp-auth-demo", stateless_http=True, json_response=True)

    @mcp.tool()
    def get_weather_tool(auth_token: str, city: str) -> str:
        return get_weather(auth_token, city)

    @mcp.tool()
    def get_invoice_tool(auth_token: str, invoice_id: str) -> str:
        return get_invoice(auth_token, invoice_id)

    return mcp


if __name__ == "__main__":
    print("MCP tool auth demo")

    for call in [
        lambda: get_weather("token_viewer", "Pune"),
        lambda: get_invoice("token_viewer", "INV-1001"),
        lambda: get_invoice("token_finance", "INV-1001"),
    ]:
        try:
            print("Result:", call())
        except PermissionError as e:
            print("Denied:", e)

    if os.getenv("RUN_MCP_SERVER", "0") == "1":
        mcp = build_mcp_server()
        mcp.run(transport="streamable-http")
