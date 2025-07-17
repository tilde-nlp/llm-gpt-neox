#!/usr/bin/env python3
"""
Skims the first five JSON-Line entries from `hay.jsonl` in the
current working directory and writes them to `hay_5s.josnl`.
"""

import json
from pathlib import Path
import sys

SRC_FILE = Path("hay.jsonl")
DST_FILE = Path("hay_5s.josnl")
N = 5

def main() -> None:
    if not SRC_FILE.exists():
        sys.exit(f"Profit blocker: '{SRC_FILE}' not found.")

    with SRC_FILE.open("r", encoding="utf-8") as src, \
         DST_FILE.open("w", encoding="utf-8") as dst:
        for idx, line in enumerate(src):
            if idx >= N:
                break
            # Validate & pretty-print each JSON dict as its own line
            data = json.loads(line)
            json.dump(data, dst, ensure_ascii=False)
            dst.write("\n")

    print(f"Top {N} lines siphoned from {SRC_FILE} → {DST_FILE}")

if __name__ == "__main__":
    main()
