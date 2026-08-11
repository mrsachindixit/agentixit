from langchain.tools import tool
from langchain.agents import create_agent
import requests

from langchain_ollama import ChatOllama
model = ChatOllama(
    model="lfm2.5-thinking:latest",  # Note: or "lfm2.5-thinking:latest" for better reasoning
    base_url='http://localhost:11434'
)

@tool
def get_user_details(user_id):
    """Get user details for a given user ID.
    Args:
    user_id (str): The ID of the user to fetch details for.
    Returns:
    dict: A dictionary containing user details such as name, age, and astrological sign.

    """
    print(f"— Tool call triggered to get user details for user_id: {user_id}")
    # Simulate fetching user details from a database or external API
    if user_id == "basic123":
        user_details = {
            "user_id": user_id,
            "name": "John Doe",
            "age": 30,
            "astrological_sign": "Aquarius",
            "country": "USA"
        }
    elif user_id == "premium123":
        user_details = {
            "user_id": user_id,
            "name": "Jane Smith",
            "age": 28,
            "astrological_sign": "Leo",
            "country": "UK"
        }
    print(f"— Tool call result for get user details {user_details}")
    return user_details

@tool
def get_horoscope(sign):
    """
    Get today's horoscope for an astrological sign.

    Args:
        sign (str): The astrological sign to get the horoscope for.
        
    Returns:
        str: The horoscope for the given astrological sign.
    """
    print(f"— Tool call triggered to get horoscope for {sign}")
    url = f"https://www.freehoroscopeapi.com/api/v1/get-horoscope/daily?sign={sign.lower()}"
    try:
        # WARNING: Disabling SSL verification is insecure and should only be used for testing.
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()
        horoscope = data.get("horoscope", "No horoscope found.")
        print(f"— Tool call result for get horoscope {sign}: {horoscope}")
        return horoscope
    except Exception as e:
        print(f"Error fetching horoscope for {sign}: {e}")
        return f"{sign.capitalize()}: Error fetching horoscope. ({e})"

@tool
def get_membership(user_id):
    """
    Get membership details for a given user ID.

    Args:
        user_id (str): The ID of the user to fetch membership details for.
        
    Returns:
        str: The membership details for the given user ID.
    """
    print(f"— Tool call triggered to get membership details for user_id: {user_id}")
    # Simulate fetching membership details from a database or external API
    if user_id == "basic123":
        membership_details = {
            "user_id": user_id,
            #"membership_type": "Premium",
            "membership_type": "Basic",
            "expiry_date": "2024-12-31"
        }
    elif user_id == "premium123":
        membership_details = {
            "user_id": user_id,
            "membership_type": "Premium",
            "expiry_date": "2024-12-31"
        }
    print(f"— Tool call result for get membership details {membership_details}")
    return membership_details

@tool
def update_horoscope_for_user(user_id, horoscope):
    """ Update the user's horoscope in the database.
    Args:
    user_id (str): The ID of the user to update the horoscope for.
    horoscope (str): The horoscope to update for the user.
    Returns:
    str: A confirmation message indicating the horoscope has been updated for the user.
    """
    print(f"— Tool call triggered to update horoscope for user_id: {user_id} with horoscope: {horoscope}")
    # Simulate updating the user's horoscope in a database
    print(f"— Tool call result for update horoscope for user_id {user_id}: {horoscope}")
    return f"Horoscope for user_id {user_id} updated to: {horoscope}"

@tool
def get_weather(country: str, units: str):
    """
    Fetch current weather information for a given country and units.

    Args:
        country (str): The country to fetch weather for.
        units (str): The units for temperature (e.g., 'celsius' or 'fahrenheit').

    Returns:
        str: The current weather information for the given country and units.
    """
    print(f"— Tool call triggered to get weather information for {country} in {units}")
    weather_info = f"The current weather in {country} is 25 degrees {units}."
    print(f"— Tool call result for get weather information {weather_info}")
    return weather_info


agent = create_agent(
    model=model,
    tools=[get_horoscope, get_user_details, update_horoscope_for_user, get_weather, get_membership],
    # debug=True
)

def run_agent(user_message):
    messages = [
        {"role":"system", "content": "You are a personal assistant.\n" 
        "When the user provides their user ID, Do following\n"
        "1. Fetch user's profile using get_user_details and get_membership tool\n"
        "If membership returned by get_membership tool is PREMIUM\n"
        "- Get their astrological sign returned by get_user_details tool, fetch their horoscope for the day using get_horoscope tool\n"
        "- Update the horoscope fetched by get_horoscope tool in the database using update_horoscope_for_user tool.\n" 
        "If membership returned by get_membership tool is not PREMIUM\n"
        "- Get the user's country returned by get_user_details tool, fetch the current weather for the user's country using the get_weather tool.\n"
        "Use the provided tools to perform these actions. Use only known arguments for the giving inputs to tools.\n "
        "Try to sequencialize the tools one after other according to dependencies. Summarize all the data fetched. Please analyze and add step by step resoning <reasoning> tag\n"},
        {"role": "user", "content": user_message}
    ]

    ai_msg = agent.invoke({"messages": messages})
    return ai_msg.get('messages')[-1].content

if __name__ == "__main__":
    user_message = "My user ID is basic123."
    print(f"User : {user_message}")
    response = run_agent(user_message)
    print(f"Agent : {response}")

    # print("="*60)
    # user_message = "My user ID is premium123."
    # print(f"User : {user_message}")
    # response = run_agent(user_message)
    # print(f"Agent : {response}")