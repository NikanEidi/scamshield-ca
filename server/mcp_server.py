import sys
import os


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.mcpserver import MCPServer
from src.classifier import classify

mcp = MCPServer("ScamShiled")

@mcp.tool()
def classify_message(text: str):
    """Classify a message as a Canadian scam type (fake_etransfer, fake_cra,
    fake_canada_post) or legit, with a confidence score per class."""
    return classify(text)


if __name__ == "__main__":
    mcp.run()


