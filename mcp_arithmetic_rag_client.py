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
You are a helpful AI assistant with access to both arithmetic operations and a RAG (Retrieval-Augmented Generation) system for PDF documents.

You have access to the following tools:

Arithmetic Tools (for calculations ONLY):
- add(a, b) - Add two numbers together
- subtract(a, b) - Subtract second number from first number
- multiply(a, b) - Multiply two numbers together
- divide(a, b) - Divide first number by second number

RAG Tool (for ALL other questions):
- ask_pdf(pdf_path, question) - Ask any question about a PDF document. This tool automatically loads and indexes the PDF, then answers your question with source citations.

IMPORTANT ROUTING RULES:
1. For arithmetic calculations (addition, subtraction, multiplication, division):
   - Use ONLY the arithmetic tools (add, subtract, multiply, divide)
   - ALWAYS use the tools for calculations, never calculate in your head

2. For ALL other questions (questions about PDFs, general knowledge, information retrieval, etc.):
   - Use ONLY the ask_pdf tool
   - You MUST provide the full PDF path (e.g., 'c:\\Users\\deep.thakkar\\Documents\\AI Workshop\\RAG Mini project\\2. SY_Booklet_AY_2025-26.pdf')
   - If the user doesn't specify a PDF path, ask them to provide it
   - The ask_pdf tool will handle loading, indexing, and querying the PDF automatically

Examples:
- "5 + 3" → Use add(5, 3)
- "What is the exit rule after second year?" → Use ask_pdf(pdf_path, "What is the exit rule after second year?")
- "Explain machine learning" → Use ask_pdf(pdf_path, "Explain machine learning")
- "100 / 5" → Use divide(100, 5)

Always use the appropriate tool based on the user's request.
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
        
        # MCP server configuration - connecting to both arithmetic and RAG servers
        mcp_config = {
            "arithmetic": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            },
            "rag": {
                "url": "http://localhost:8001/sse",
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
    print("Starting LangChain Agent with MCP Arithmetic + RAG Servers...")
    print("Type 'quit' or 'exit' to end the session\n")
    
    # Initialize client
    client = ArithmeticMCPClient()
    
    print("Example commands:")
    print("  Arithmetic: '5 + 3', '10 * 2', '100 / 5'")
    print("  Load PDF: 'Load the PDF at c:\\path\\to\\file.pdf'")
    print("  Query PDF: 'What is the main topic in the PDF?'")
    print("  General: 'What is machine learning?'\n")
    
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
            
            # Extract the response content
            response_content = ""
            if 'structured_response' in response:
                response_content = response['structured_response'].message
            else:
                # Fallback for older format
                last_message = response['messages'][-1]
                if hasattr(last_message, 'content'):
                    response_content = last_message.content
                else:
                    response_content = str(last_message)
            
            # Check if this is a RAG response with sources
            if "Sources:" in response_content:
                # Split answer and sources
                parts = response_content.split("Sources:", 1)
                answer_part = parts[0].replace("Answer:", "").strip()
                sources_part = parts[1].strip() if len(parts) > 1 else ""
                
                print(f"Answer: {answer_part}\n")
                
                if sources_part:
                    print("📚 Sources from PDF:")
                    print(sources_part)
            else:
                # Regular output (arithmetic or other)
                print(f"Output: {response_content}")
            
            print("--- End Response ---\n")
            
        except Exception as e:
            print(f"\nError: {e}")
            print("Please make sure both MCP servers are running:")
            print("  - Arithmetic server on http://localhost:8000")
            print("  - RAG server on http://localhost:8001")


if __name__ == "__main__":
    asyncio.run(main())
