import os
import json
from pathlib import Path
import requests

if __name__ == "__main__":
    # Paths and API base
    ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "model"))
    META_PATH = ARTIFACT_DIR / "model_meta.json"
    API_BASE = os.getenv("API_BASE", "http://localhost:8000")

    # --- Step 1: Increment model version number ---
    try:
        if META_PATH.exists():
            meta = json.loads(META_PATH.read_text())
            version = meta.get("version", "v0")
            
            # Parse current version and increment
            if version.startswith("v") and version[1:].isdigit():
                next_version = f"v{int(version[1:]) + 1}"
            else:
                next_version = "v1"

            meta["version"] = next_version
            META_PATH.write_text(json.dumps(meta, indent=2))
            print(f"🔢 Model version incremented to {next_version}")
        else:
            # create fresh meta file if missing
            meta = {"version": "v1"}
            META_PATH.write_text(json.dumps(meta, indent=2))
            print("🆕 Created new model_meta.json with version v1")
    except Exception as e:
        print(f"⚠️ Failed to update version: {e}")

    # --- Step 2: Notify API to reload ---
    try:
        r = requests.post(f"{API_BASE}/admin/reload", timeout=10)
        r.raise_for_status()
        print(f"🔁 API reload successful → {r.json()}")
    except requests.RequestException as e:
        print(f"⚠️ Could not reach API at {API_BASE}: {e}")
