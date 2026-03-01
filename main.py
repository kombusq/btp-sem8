#!/usr/bin/env python3
"""
CLI entry point to run the investment committee for a single ticker.
Usage: python main.py [TICKER]
       Default ticker: AAPL
"""

import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "src"))

from dotenv import load_dotenv
load_dotenv(root / ".env")

from trading_agent.crew import run_crew


def main():
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "AAPL").strip().upper()
    model = (os.environ.get("MODEL") or "groq/llama-3.1-70b-versatile").strip()
    api_key = (os.environ.get("API_KEY") or "").strip()
    is_local = model.lower().startswith("ollama/")
    if not api_key and not is_local:
        print("Error: API_KEY not set in .env. For Ollama use MODEL=ollama/llama3.2 (no key needed).", file=sys.stderr)
        sys.exit(1)
    print(f"Running committee for {ticker} (model: {model})...")
    result = run_crew(ticker, api_key=api_key)
    print("\n--- Recommendation ---")
    print(result["recommendation"])
    print("\n--- Justification ---")
    print(result["justification"])
    print("\n--- Researcher (sentiment) ---")
    print(result["research_output"])
    print("\n--- Quant (technical) ---")
    print(result["quant_output"])
    print("\n--- Portfolio Manager (decision) ---")
    print(result["decision_output"])


if __name__ == "__main__":
    main()
