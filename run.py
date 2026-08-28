#!/usr/bin/env python3
"""Start Murshid locally.

    pip install -r requirements.txt
    cp .env.example .env      # then add your GROQ_API_KEY
    python run.py

Then open http://127.0.0.1:7860 in a browser.
Pass --share for a temporary public link.
"""

import argparse
import sys

from app.murshid import build
from app.ui import create_app, launch_kwargs


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Murshid student services agent.")
    ap.add_argument("--share", action="store_true",
                    help="also expose a temporary public link (~72 hours)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    print("Starting Murshid…\n")
    try:
        _, info = build()
    except Exception as e:
        print(f"\nCould not start: {type(e).__name__}\n{e}\n", file=sys.stderr)
        return 1

    print("\nready — opening the interface\n")
    create_app(info).launch(server_name="127.0.0.1", server_port=args.port,
                            share=args.share, inbrowser=True, **launch_kwargs())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
