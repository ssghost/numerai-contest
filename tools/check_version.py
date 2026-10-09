import os
import json
import re
from numerapi import NumerAPI

CACHE_FILE = "version_cache.json"

def check_version():
    napi = NumerAPI()
    current_datasets = sorted(napi.list_datasets())

    version_pattern = re.compile(r"^(v\d+\.\d+)/")
    current_versions = sorted(list({
        match.group(1) for d in current_datasets if (match := version_pattern.match(d))
    }))

    current_state = {
        "versions": current_versions,
        "datasets": current_datasets
    }

    if not os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(current_state, f, indent=2, ensure_ascii=False)
        return

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        last_state = json.load(f)

    last_versions = last_state.get("versions", [])
    last_datasets = set(last_state.get("datasets", []))
    
    new_versions = [v for v in current_versions if v not in last_versions]
    new_datasets = set(current_datasets) - last_datasets

    if new_versions or new_datasets:
        print("[WARNING] Numerai dataset version updated.")
        if new_versions:
            print(f"New version: {new_versions}")
        if new_datasets:
            print(f"New files: {len(new_datasets)}")
            for item in sorted(list(new_datasets))[:10]:
                print(f"  + {item}")
        print(f"Version list: {current_versions}")

        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(current_state, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    check_version()