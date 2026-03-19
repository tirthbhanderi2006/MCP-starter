import asyncio
import os
import time

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()


PROMPT = """
You are an arithmetic calculator. You MUST use the available tools to perform calculations.

For simple expressions:
- "1+2" means add 1 and 2
- "2+100" means add 2 and 100  
- "5+6" means add 5 and 6
- "10-3" means subtract 3 from 10
- "4*5" means multiply 4 and 5
- "20/4" means divide 20 by 4

ALWAYS use the tools. Never calculate in your head.
The user input is a mathematical expression that needs to be evaluated exactly as written.
"""


class MCPResponse(BaseModel):
    message: str


class ArithmeticMCPClient:
    def __init__(self):
        self.model = ChatOpenAI(
            # model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            model="gpt-5-mini",
            api_key=os.getenv("OPEN_AI_API_KEY"),
            reasoning={
                "effort": "low",  # 'low', 'medium', or 'high'
            }
            # temperature=0,
        )
    async def run_agent(self, message, thread_id="default_thread"):
        # Use MemorySaver for conversation history
        memory = MemorySaver()
        
        # MCP server configuration
        mcp_config = {
            "arithmetic": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            }
        }
        
        # Create client (not as context manager)
        client = MultiServerMCPClient(mcp_config)
        
        # Get tools from the server
        tools = await client.get_tools()
        
        # Debug: Print available tools
        print(f"Available tools: {[tool.name for tool in tools]}")
        
        config = {"configurable": {"thread_id": thread_id}}
        system_msg = SystemMessage(content=PROMPT)
        
        # Create the agent with tools from MCP server
        agent = create_agent(
            self.model,
            tools,
            system_prompt=system_msg,
            response_format=MCPResponse,
        )
        
        start = time.perf_counter()
        agent_response = await agent.ainvoke({"messages": [HumanMessage(content=message)]})
        elapsed = time.perf_counter() - start
        
        # Convert to expected format
        return {"messages": [HumanMessage(content=message), agent_response]}


async def main():
    """Main function to interact with the agent"""
    print("Starting LangChain Agent with MCP Arithmetic Server...")
    print("Type 'quit' or 'exit' to end the session\n")
    
    # Initialize client
    client = ArithmeticMCPClient()
    
    while True:
        # Get user input
        query = input("\nYou: ").strip()
        
        # Check for exit commands
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        # Skip empty queries
        if not query:
            continue
        
        print("\nProcessing...")
        
        try:
            response = await client.run_agent(query, thread_id="interactive_session")
            
            # Print the entire response
            print("\n--- Agent Response ---")
            # Extract the structured response message
            if 'structured_response' in response:
                print(f"Output: {response['structured_response'].message}")
            else:
                # Fallback for older format
                last_message = response['messages'][-1]
                if hasattr(last_message, 'content'):
                    print(f"Output: {last_message.content}")
                else:
                    print(f"Output: {last_message}")
            print("--- End Response ---\n")
            
        except Exception as e:
            print(f"\nError: {e}")
            print("Please make sure the MCP server is running on http://localhost:8000")


if __name__ == "__main__":
    asyncio.run(main())
