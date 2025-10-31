import os
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

api_key = os.getenv("GALILEO_API_KEY")
project = os.getenv("GALILEO_PROJECT", "LinearRegression-Observability")
log_stream = os.getenv("GALILEO_LOG_STREAM", "lr-smoke-stream")

if not api_key:
    raise RuntimeError("GALILEO_API_KEY not found in environment. Ensure .env contains GALILEO_API_KEY or set it in the shell.")

import galileo

# Ensure project and log stream exist
proj = galileo.projects.get_project(name=project)
if proj is None:
    proj = galileo.projects.create_project(project)

ls = galileo.log_streams.get_log_stream(name=log_stream, project_name=project)
if ls is None:
    ls = galileo.log_streams.create_log_stream(name=log_stream, project_name=project)

# Initialize logger bound to project + log stream
logger = galileo.GalileoLogger(project=project, log_stream=log_stream)

# Start a session and a named trace (required input) so it appears in Recent Traces
logger.start_session()
trace_name = f"smoke-trace-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
logger.start_trace(
    input="smoke trace started",
    name=trace_name,
    created_at=datetime.utcnow(),
    tags=["smoke", "test"],
    metadata={"source": "galileo_smoke_test.py"}
)
span_name = f"smoke-span-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
logger.add_llm_span(
    input="smoke input",
    output="smoke output",
    model="smoke-model",
    name=span_name,
    created_at=datetime.utcnow(),
    metadata={"source": "galileo_smoke_test.py"},
    tags=["smoke", "test"]
)

logger.flush()
logger.conclude()

print("Galileo smoke test completed. Check your Galileo UI:")
print(f"  Project:    {project}")
print(f"  Log Stream: {log_stream}")
print(f"  Span Name:  {span_name}")
print(f"  Trace Name: {trace_name}")