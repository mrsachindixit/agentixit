

import requests

url = "http://localhost:11434/api/chat"
payload = {
    "model": "llama3.1:latest",
    "stream": False
}

def summarize(history_text: str) -> str:
    """Compress conversation history into key facts."""
    messages = [
        {"role": "system", "content": "Summarize the following chat into key facts."},
        {"role": "user", "content": history_text}
    ]
    payload["messages"] = messages
    response = requests.post(url, json=payload)
    return response.json().get("message", {}).get("content", "")

if __name__ == "__main__":
    history = []
    
    def ask(user_msg):
        """
        Send message with sliding window context.
        Keeps last N messages for LLM reasoning.
        """
        window_size = 6  # Keep last 6 messages
        window = history[-window_size:]  # Sliding window
        
        # Build message list: system instruction + recent context + new query
        messages = [{"role": "system", "content": "You are helpful and concise."}]
        messages += window  # Add recent context
        messages.append({"role": "user", "content": user_msg})
        
        payload["messages"] = messages
        response = requests.post(url, json=payload)
        model_response = response.json().get("message", {}).get("content", "")
        
        # Append to history for next turn
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": model_response})
        
        return model_response

    # Conversation turns
    print("Turn 1:", ask('My name is Sachin.'))
    print("Turn 2:", ask('I work as an architect.'))
    print("Turn 3:", ask('What city do I live in?'))  # No info yet, will say unknown
    print("Turn 4:", ask('I live in Pune.'))
    print("Turn 5:", ask('What city do I live in?'))  # Now it remembers
    print("Turn 6:", ask('Remember this preference: I like minimal code samples.'))

    # Memory compression: Summarize everything then reset with summary + recent
    print("\n=== MEMORY COMPRESSION ===")
    long_summary = summarize("".join([m["role"] + ":" + m["content"] for m in history]))
    print(f"Summary: {long_summary}\n")
    
    # Keep only summary + last 6 messages
    history = [{"role": "system", "content": "Long-term memory: " + long_summary}] + history[-6:]
    
    print("Turn 7:", ask('Given my preferences, how would you structure a tutorial?'))


