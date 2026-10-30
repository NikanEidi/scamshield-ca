import sys
import os


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.mcpserver import MCPServer
from src.classifier import classify
from src.similarity import find_similar_scams

mcp = MCPServer("ScamShiled")

@mcp.tool()
def classify_message(text: str):
    """Classify a message as a Canadian scam type (fake_etransfer, fake_cra,
    fake_canada_post, fake_bank_alert) or legit, with a confidence score per class."""
    return classify(text)


@mcp.tool()
def find_similar(text: str, k: int = 3):
    """Find the k most semantically similar known examples to a message in the
    scamshield-ca dataset, each with its label and a similarity score. Useful as
    supporting evidence for why a message was classified a certain way -- e.g.
    showing 3 other real fake_bank_alert examples that read almost the same."""
    return find_similar_scams(text, k=k)


if __name__ == "__main__":
    mcp.run()


