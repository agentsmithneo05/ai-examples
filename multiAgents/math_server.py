from mcp.server.fastmcp import FastMCP

# 1. Initialize the MCP Server
mcp = FastMCP("AdvancedMath")

@mcp.tool()
def calculate_compound_interest(principal: float, rate: float, time: int) -> str:
    """Calculates compound interest for financial analysis."""
    amount = principal * (1 + rate/100)**time
    interest = amount - principal
    return f"After {time} years, the total is {amount:.2f} (Interest: {interest:.2f})"

if __name__ == "__main__":
    # Run using 'stdio' so our LangGraph agent can talk to it via terminal
    mcp.run(transport="stdio")