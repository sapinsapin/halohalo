"""
Publish dataset cards from docs/cards/ to the Hub.

The cards are kept in-repo so they are reviewable and diffable like code
rather than edited in a web form: `docs/cards/<name>.md` is the source of
truth for `<org>/<name>`'s README.

Usage:
  python push_cards.py --dry-run          # show what would change
  python push_cards.py                    # push all cards
  python push_cards.py pld halo-livestream

Note: process_pld_parquet.py regenerates the PLD card at the end of a run.
Re-run this script afterwards to restore the hand-authored card.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

CARDS_DIR = Path(__file__).parent / "docs" / "cards"
ORG = os.environ.get("HF_ORG", "sapinsapin")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("names", nargs="*",
                    help="card stems to push (default: all in docs/cards/)")
    ap.add_argument("--org", default=ORG)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    cards = sorted(CARDS_DIR.glob("*.md"))
    if args.names:
        wanted = set(args.names)
        cards = [c for c in cards if c.stem in wanted]
        missing = wanted - {c.stem for c in cards}
        if missing:
            sys.exit(f"no card file for: {', '.join(sorted(missing))}")
    if not cards:
        sys.exit(f"no cards found in {CARDS_DIR}")

    api = HfApi(token=os.environ.get("HF_TOKEN"))

    for card in cards:
        repo_id = f"{args.org}/{card.stem}"
        body = card.read_text(encoding="utf-8")

        # compare against what is live so a no-op push is visible as a no-op
        try:
            from huggingface_hub import hf_hub_download
            live = Path(hf_hub_download(repo_id, "README.md", repo_type="dataset",
                                        token=os.environ.get("HF_TOKEN"))
                        ).read_text(encoding="utf-8")
        except Exception:
            live = None

        state = "unchanged" if live == body else ("new" if live is None else "changed")
        print(f"{repo_id:45s} {len(body):6,d} bytes  {state}")

        if args.dry_run or state == "unchanged":
            continue

        api.upload_file(
            path_or_fileobj=body.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Update dataset card ({card.name})",
        )
        print(f"  → https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
