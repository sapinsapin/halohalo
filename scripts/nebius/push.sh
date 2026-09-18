#!/usr/bin/env bash
# Copy the working tree (uncommitted edits included) and the frozen splits to
# ~/halohalo on the VM. The VM has no GitHub key, so this replaces git clone.
# Secrets are NOT copied: see push_secrets.sh.
source "$(dirname "$0")/common.sh"
cd "$REPO"

list="$(mktemp)"
trap 'rm -f "$list"' EXIT
# tracked + untracked-but-not-ignored files, minus local corpora and old run
# dirs that are not ignored but are far too big to ship (the VM pulls data from
# the Hub). splits/ is gitignored, so it is added explicitly below.
git ls-files -co --exclude-standard -z \
  | tr '\0' '\n' \
  | grep -Ev '^(FilipinoSpeechCorpus|wandb|finetune_runs[^/]*|pld|PLD[^/]*|data)/' \
  | while IFS= read -r f; do
        [ -f "$f" ] || continue
        if [ "$(stat -c %s "$f")" -gt 52428800 ]; then echo "skipping >50 MB: $f" >&2; continue; fi
        printf '%s\0' "$f"
    done > "$list"
echo "pushing $(tr -cd '\0' < "$list" | wc -c) files + splits/"

vm_ssh 'mkdir -p ~/halohalo'
tar --null -T "$list" -cf - splits | vm_ssh 'tar -xf - -C ~/halohalo'
vm_ssh 'cd ~/halohalo && find scripts -name "*.sh" -exec sed -i "s/\r$//" {} + && echo "on VM: $(du -sh . | cut -f1), splits: $(ls splits | wc -l)"'
