
#TODO : Need proper demostration of sub agent

from langchain.tools import tool
from langchain.agents import create_agent

import requests
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import InMemorySaver

from langchain_ollama import ChatOllama
model = ChatOllama(
    model="lfm2.5-thinking:latest",  # Note: or "lfm2.5-thinking:latest" for better reasoning
    base_url='http://localhost:11434',
    reasoning=True
)
@tool
def get_user_details(user_id):
    """Get user details for a given user ID.
    Args:
    user_id (str): The ID of the user
    Returns:
    dict: A dictionary containing user details such as country and astrological sign.

    """
    print(f"— Tool call triggered to get user details for user_id: {user_id}")
    # Simulate fetching user details from a database or external API
    if user_id == "user123":
        user_details = {
            "user_id": user_id,
            "astrological_sign": "Aquarius",
            "country": "USA"
        }
    elif user_id == "user456":
        user_details = {
            "user_id": user_id,
            "astrological_sign": "Leo",
            "country": "UK"
        }
    print(f"— Tool call result for get user details {user_details}")
    return user_details

@tool
def get_horoscope(sign):
    """
    Fetch today's horoscope for given astrological sign.

    Args:
        sign (str): The astrological sign.
        
    Returns:
        str: The horoscope for the given astrological sign.
    """
    print(f"— Tool call triggered to get horoscope for {sign}")
    horoscope = f"{sign} : Today is a good day to learn something new."
    print(f"— Tool call result for get horoscope {sign}: {horoscope}")
    return horoscope

@tool
def get_membership(user_id):
    """
    Get membership details for a given user ID.
    Args:
        user_id (str): The ID of the user
    Returns:
        dict: The membership details for the given user ID.
    """
    print(f"— Tool call triggered to get membership details for user_id: {user_id}")
    # Simulate fetching membership details from a database or external API
    if user_id == "user123":
        membership_details = {
            "user_id": user_id,
            #"membership_type": "Premium",
            "membership_type": "Basic"
        }
    elif user_id == "user456":
        membership_details = {
            "user_id": user_id,
            "membership_type": "Premium"
        }
    print(f"— Tool call result for get membership details {membership_details}")
    return membership_details

@tool
def update_horoscope_for_user(user_id, horoscope):
    """ Update the horoscope
    Args:
    user_id (str): The ID of the user
    horoscope (str): The horoscope to update for the user.
    Returns:
    bool: True/False.
    """
    print(f"— Tool call triggered to update horoscope for user_id: {user_id} with horoscope: {horoscope}")
    # Simulate updating the user's horoscope in a database
    print(f"— Tool call result for update horoscope for user_id {user_id}: {horoscope}")
    return True

@tool
def get_weather(country: str):
    """
    Fetch current weather
    Args:
        country (str): The country
    Returns:
        str: The current weather
    """
    print(f"— Tool call triggered to get weather information for {country}")
    weather_info = f"{country} : 25 Celcius."
    print(f"— Tool call result for get weather information {weather_info}")
    return weather_info

#create the agent for fetching user profile
userIdentityAgent = create_agent(
    tools=[get_user_details, get_membership], # Partitioning the tools for user identity agent
    model=model,
    system_prompt="You are a helpful assistant that fetches user details and membership for given user ID. Always respond with the final action taken. Membership information is important make sure it is returned as part of final response",
    debug=False
)
#create the agent for fetching and updating horoscope
horoscopeAgent = create_agent(
    tools=[get_horoscope, update_horoscope_for_user], # Partitioning the tools for horoscope agent
    model=model,
    debug=False,
    system_prompt="You are a helpful assistant who performs below steps in sequence when asked about updating horoscope : " \
    "1. Fetche today's horoscope for given astrological sign using get_horoscope tool" \
    "2. Update the horoscope fetched earlier for user using update_horoscope_for_user tool" \
    "RULES : " \
    "ONLY update horoscope after fetching it. "
)
#create the agent for fetching weather information
weatherAgent = create_agent(
    tools=[get_weather], # Partitioning the tools for weather agent
    model=model,
    system_prompt="You are a helpful assistant that fetches weather information for given country. Always respond with the final action taken",
    debug=False
)

#create tool for each agent
@tool
def user_identity_tool(user_id):
    """ Fetch user details like astrological sign and membership for a given user ID.
    Args:
        user_id (str): The ID of the user to fetch details and membership for.
    Returns:
        str: The user details and membership information for the given user ID.
    """
    print(f"— Subagent call triggered to get user identity profile for user_id: {user_id}")
    ai_msg = userIdentityAgent.invoke({"messages": [{"role": "user", "content": f"Fetch user details and membership for user_id: {user_id}"}]})
    print(f"— Subagent call result for user identity profile for user_id: {user_id}: {ai_msg['messages'][-1].content}")
    return ai_msg['messages'][-1].content
    
@tool
def horoscope_tool(user_id, astrological_sign):
    """ Fetch and update horoscope for a given astrological sign.
    Args:
        user_id (str): The ID of the user to fetch and update the horoscope for.
        astrological_sign (str): The astrological sign to fetch and update the horoscope for.
    Returns:
        str: The horoscope information for the given astrological sign.
    """
    print(f"— Subagent call triggered to fetch and update horoscope for user_id: {user_id} and astrological_sign: {astrological_sign}")
    ai_msg = horoscopeAgent.invoke({"messages": [{"role": "user", "content": f"Get today's horoscope and update it for user_id: {user_id} and astrological_sign: {astrological_sign}"}]})
    print(f"— Subagent call result for horoscope for user_id: {user_id} and astrological_sign: {astrological_sign}: {ai_msg['messages'][-1]}")
    return ai_msg['messages'][-1].content

@tool
def weather_tool(country):
    """ Fetch weather information for a given country.
    Args:
        country (str): The country to fetch weather information for.
    Returns:
        str: The weather information for the given country.
    """
    print(f"— Subagent call triggered to fetch weather information for country: {country}")
    ai_msg = weatherAgent.invoke({"messages": [{"role": "user", "content": f"Fetch weather information for country: {country}"}]})
    print(f"— Subagent call result for weather information for country: {country}: {ai_msg['messages'][-1].content}")
    return ai_msg['messages'][-1].content


checkpointer = InMemorySaver()
#create a master agent that can call other agents as tools
masterAgent = create_agent(
    tools=[user_identity_tool, horoscope_tool, weather_tool],
    model=model,
    checkpointer=checkpointer,
    system_prompt="After getting user_id from user, Do the following:\n"
        "1. Fetch user details (city and astrological sign) and membership for given user_id\n"
        "2. if membership is Premium, fetch and update horoscope based on astrological_sign retrieved as part of user details\n"
        "3. if membership is not Premium, fetch weather information based on country retrieved as part of user details\n"
        "Do not stop until you have performed all the necessary steps based on available data and conditions, including data from previous steps. Always respond with the final action taken\n",
    # debug=True
    )

config_basic = {"configurable": {"thread_id": "user_session_1"}}
config_premium = {"configurable": {"thread_id": "user_session_2"}}
def run_agent(user_message, config):
    ai_msg=masterAgent.invoke({"messages": [{
        "role": "user", 
        "content": user_message
    }]}, config=config)
    print(f"— Master agent final response: {ai_msg.get('messages',[])[-1]}")    
    return ai_msg.get("messages",[])[-1].content

if __name__ == "__main__":
    # Uncomment this.. to test with basic membership user
    # user_message = "My user ID is user123."
    # print(f"User : {user_message}")
    # response = run_agent(user_message, config_basic)
    # print(f"Agent : {response}")

    # Uncomment this.. to test with premium membership user
    print("="*60)
    user_message = "My user ID is user456."
    print(f"User : {user_message}")
    response = run_agent(user_message, config_premium)
    print(f"Agent : {response}")

    # print("="*60)
    # user_message = "what is my membership?"
    # print(f"User : {user_message}")
    # response = run_agent(user_message, config_basic)
    # print(f"Agent : {response}")
