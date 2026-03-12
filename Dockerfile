FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# FastAPI backend
EXPOSE 8000
# MCP SSE server
EXPOSE 8001

# Default: run the FastAPI backend.
# Override CMD to run MCP: ["python", "mcp_server.py", "--sse", "--port", "8001"]
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
