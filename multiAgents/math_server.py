# math_server.py
import logging
from mcp.server.fastmcp import FastMCP

# Prevent logs from leaking into the pipe
logging.basicConfig(level=logging.INFO)
mcp = FastMCP("InsightMath")


@mcp.tool()
def calculate_growth(principal: float, percentage_growth: float, years: int) -> str:
    """Calculates future value. principal: start amount, percentage_growth: e.g. 12, years: time."""
    rate = float(percentage_growth)
    if 0 < rate < 1:
        rate = rate * 100

    total = float(principal) * (1 + (rate / 100)) ** int(years)
    return f"SIMULATION_SUCCESS: {total:.2f}"


if __name__ == "__main__":
    # In some versions of FastMCP, SSE is triggered like this:
    mcp.run(transport="sse")
    # Note: If it doesn't allow 'port', it will default to 8000.