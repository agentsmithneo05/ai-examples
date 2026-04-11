import yfinance as yf
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("FinanceEngine")

@mcp.tool()
def get_stock_price(ticker: str) -> float:
    """Gets price for a ticker."""
    return float(yf.Ticker(ticker).fast_info['last_price'])

@mcp.tool()
def calculate_growth(principal: float, percentage_growth: float, years: int) -> str:
    """Calculates growth."""
    total = float(principal) * (1 + (float(percentage_growth) / 100)) ** int(years)
    return f"RESULT: {total:.2f}"

if __name__ == "__main__":
    mcp.run(transport="sse")
                            