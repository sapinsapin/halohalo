# Sourced by the other scripts in this folder. Runs in WSL, not on the VM.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$HERE/vm.env"
export PATH="$HOME/.nebius/bin:$HOME/.local/bin:$PATH"

# public images live in project-<region code>public-images, e.g. project-e05public-images
IMAGE_PARENT="project-${NEBIUS_PROJECT:8:3}public-images"
# VMs are disposable and preemption can hand the same IP to a new host key, so
# keep their host keys out of ~/.ssh/known_hosts
KNOWN_HOSTS="$HOME/.ssh/known_hosts_nebius"
DATA_ROOT=/mnt/data

IP_CACHE="$HOME/.cache/halohalo-nebius-ip"

# The CLI's federation login expires and cannot open a browser from WSL, which
# otherwise looks exactly like "the VM is gone". Tell the two apart: an error
# here means the CLI, not the VM.
vm_field() {  # vm_field '{.status.state}' -> value; empty + rc 1 if the CLI failed
    local out rc
    out="$(nebius compute instance get-by-name --parent-id "$NEBIUS_PROJECT" \
           --name "$VM_NAME" --format "jsonpath=$1" 2>&1)"; rc=$?
    if [ $rc -ne 0 ]; then
        if grep -qi 'open browser\|auth\|token' <<<"$out"; then
            echo "!! Nebius CLI is not signed in. In a WSL terminal run:  nebius iam whoami" >&2
        elif ! grep -qi 'not found' <<<"$out"; then
            echo "!! nebius: $(head -1 <<<"$out")" >&2
        fi
        return 1
    fi
    printf '%s' "$out"
}
vm_id()    { vm_field '{.metadata.id}'; }
vm_state() { vm_field '{.status.state}'; }
vm_ip() {
    local ip
    if ip="$(vm_field '{.status.network_interfaces[0].public_ip_address.address}')" && [ -n "$ip" ]; then
        ip="${ip%%/*}"; mkdir -p "$(dirname "$IP_CACHE")"; printf '%s' "$ip" > "$IP_CACHE"
        printf '%s' "$ip"; return 0
    fi
    # SSH does not need the CLI, so fall back to the last IP we saw
    [ -n "${VM_IP:-}" ] && { printf '%s' "$VM_IP"; return 0; }
    [ -s "$IP_CACHE" ] && { cat "$IP_CACHE"; return 0; }
    return 1
}

vm_ssh() {  # vm_ssh [ssh args...] -- runs against the current IP
    local ip; ip="$(vm_ip)" || true
    [ -n "$ip" ] || { echo "!! no IP for $VM_NAME (CLI signed out? pass VM_IP=x.x.x.x)" >&2; return 1; }
    ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=15 -o LogLevel=ERROR \
        -o UserKnownHostsFile="$KNOWN_HOSTS" -o StrictHostKeyChecking=accept-new \
        "$VM_USER@$ip" "$@"
}
