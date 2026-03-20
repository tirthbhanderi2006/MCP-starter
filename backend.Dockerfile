# syntax=docker/dockerfile:1.4
FROM python:3.11-slim-bullseye

WORKDIR /app

# Install fastmcp dependencies with BuildKit cache
RUN --mount=type=cache,target=/root/.cache/uv \
    pip install --no-cache-dir uv \
    && uv pip install --system --quiet fastmcp


# Copy server code
COPY mcp_arithmetic_server.py .

# Expose port (8000)
EXPOSE 8000

# Run the FastMCP server with host bind so it can be reached from the Docker network
CMD ["python", "mcp_arithmetic_server.py"]
