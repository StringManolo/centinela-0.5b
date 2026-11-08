#!/usr/bin/env python3
"""Generate systemPromptClassify.json from labels.json (run after editing labels.json)."""
import json, os, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
LABELS_PATH = Path(os.environ.get("CENTINELA_LABELS_JSON") or SCRIPT_DIR / "labels.json")
OUT_PATH = SCRIPT_DIR / "systemPromptClassify.json"

if not LABELS_PATH.is_file():
    sys.exit(f"ERROR: labels.json not found at {LABELS_PATH}")

with open(LABELS_PATH) as f:
    labels = json.load(f)["labels"]

config = {
    "systemMessage": (
        "You are a security classifier. Given a code snippet, reply with EXACTLY ONE label "
        "from the list. No explanations, no markdown, no punctuation. " +
        ",".join(labels)
    ),
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "repeat_last_n": 0,
}

with open(OUT_PATH, "w") as f:
    json.dump(config, f, indent=2)
print(f"✅ wrote {OUT_PATH} with {len(labels)} labels")
