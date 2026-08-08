"""Regenerate private_manifest.json — the name-free snapshot the dashboard
falls back to when it has no token. Run with a token that can see the org's
private repos (HF_TOKEN in the environment):

  python make_manifest.py
"""

import json
import os
from datetime import date
from pathlib import Path

from huggingface_hub import HfApi

ORG = os.environ.get("DASHBOARD_ORG", "sapinsapin")

api = HfApi(token=os.environ["HF_TOKEN"])
manifest = {
    "as_of": date.today().isoformat(),
    "private_counts": {
        "dataset": sum(d.private for d in
                       api.list_datasets(author=ORG, expand=["private"])),
        "model": sum(m.private for m in
                     api.list_models(author=ORG, expand=["private"])),
    },
}
out = Path(__file__).parent / "private_manifest.json"
out.write_text(json.dumps(manifest, indent=2) + "\n")
print(f"{out}: {manifest}")
