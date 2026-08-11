

import requests

def capital_teller(country):
    """
    Single-turn chat example: Query LLM with system instruction and user input.
    
    Args:
        country: Country name to get capital for
    """
    
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama3.1:latest",
        "messages": [
            {
                "role": "system",
                "content": "You are a country's capital teller. You need to respond with the capital of the user's country. Do not use capital letters, just provide the name of the capital. If you don't know the answer, say 'unknown'."
            },
            {
                "role": "user",
                "content": f"My country is {country}."
            }
        ],
        "stream": False
    }
    print(f"Assistant looking up capital for {country}...")
    response = requests.post(url, json=payload)
    result = response.json()
    capital = result.get("message", {}).get("content", "")
    print(f"Capital: {capital}\n")
    return capital


if __name__ == "__main__":
    capital_teller("France")
    capital_teller("India")
    capital_teller("Japan")