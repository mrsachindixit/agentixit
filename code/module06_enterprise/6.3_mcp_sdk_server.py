from mcp.server.fastmcp import FastMCP

mcp = FastMCP("annotation-demo", stateless_http=True, json_response=True)


@mcp.tool()
def get_weather(city: str) -> str:
    print(f"Received get_weather call with city: {city}")
    temps = {"london": "8°C", "tokyo": "22°C", "new york": "18°C", "bogotá": "12°C"}
    return f"{city}: {temps.get(city.lower(), '15°C')}, partly cloudy"


@mcp.tool()
def get_pincode(city: str) -> str:
    print(f"Received get_pincode call with city: {city}")
    pincodes = {"london": "SW1A1AA", "tokyo": "100-0001", "new york": "10001", "bogotá": "110111"}
    return f"{city}: {pincodes.get(city.lower(), '000000')}"


@mcp.resource("data://cities")
def list_cities() -> str:
    return "london, tokyo, new york, bogotá"


@mcp.resource("data://city/{name}")
def city_detail(name: str) -> str:
    details = {"london": "Capital of UK", "tokyo": "Capital of Japan"}
    return details.get(name.lower(), f"{name}: no details available")


@mcp.prompt()
def weather_prompt(city: str) -> str:
    return f"Get the weather for {city} and summarize it in one sentence."


@mcp.prompt()
def pincode_prompt(city: str) -> str:
    return f"Find the postal code for {city} and return only one line."


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
