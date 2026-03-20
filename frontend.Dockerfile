# syntax=docker/dockerfile:1.4
FROM python:3.11-slim-bullseye

WORKDIR /app

# Install uv and dependencies with BuildKit caching
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    pip install --no-cache-dir uv \
    && uv pip install --system --quiet -r requirements.txt


# Copy the frontend application code
COPY .streamlit/ .streamlit/
COPY streamlit_app.py .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
