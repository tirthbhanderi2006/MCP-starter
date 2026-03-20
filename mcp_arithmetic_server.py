import logging
from fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("arithmetic-server")


@mcp.tool()
def add(a: float, b: float) -> str:
    """Add two numbers together"""
    result = a + b
    return f"Result: {result}"


@mcp.tool()
def subtract(a: float, b: float) -> str:
    """Subtract second number from first number"""
    result = a - b
    return f"Result: {result}"


@mcp.tool()
def multiply(a: float, b: float) -> str:
    """Multiply two numbers together"""
    result = a * b
    return f"Result: {result}"


@mcp.tool()
def divide(a: float, b: float) -> str:
    """Divide first number by second number"""
    if b == 0:
        return "Error: Cannot divide by zero"
    result = a / b
    return f"Result: {result}"


if __name__ == "__main__":
    logger.info("🚀 Starting MCP Arithmetic Server...")
    logger.info("📡 Server listening on 0.0.0.0:8000 with SSE transport")
    logger.info("✅ Server is ready for connections from frontend...")
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
