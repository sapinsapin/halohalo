#!/usr/bin/env bash
# Make sure the research VM exists and is running, then wait until SSH and
# cloud-init are ready. Safe to re-run:
#   missing  -> create it (costs money: needs --create)
#   STOPPED  -> start it (preempted, or stopped by stop.sh)
#   RUNNING  -> nothing to do
#
#   bash scripts/nebius/up.sh            # start or confirm
#   bash scripts/nebius/up.sh --create   # also allowed to create a new VM
source "$(dirname "$0")/common.sh"

# vm_state returns 1 for "no such VM" as well as for a CLI failure; under
# set -e that would exit before the create branch below is ever reached
state="$(vm_state)" || true
case "$state" in
  "")
    if [ "${1:-}" != "--create" ]; then
        echo "!! $VM_NAME does not exist in $NEBIUS_PROJECT. Re-run with --create to make one:"
        echo "   $VM_PLATFORM $VM_PRESET, ${VM_DISK_GIB} GiB, preemptible=$VM_PREEMPTIBLE"
        exit 2
    fi
    subnet="$(nebius vpc subnet list --parent-id "$NEBIUS_PROJECT" --format 'jsonpath={.items[0].metadata.id}')"
    pub="$(cat "$SSH_KEY.pub")"
    userdata="$(cat <<YAML
#cloud-config
users:
  - name: $VM_USER
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys:
      - $pub
runcmd:
  - mkdir -p $DATA_ROOT
  - chown $VM_USER:$VM_USER $DATA_ROOT
YAML
)"
    preempt=()
    # preemptible VMs only accept the "fail" recovery policy
    [ "$VM_PREEMPTIBLE" = 1 ] && preempt=(--preemptible-on-preemption stop --recovery-policy fail)
    echo "creating $VM_NAME ..."
    nebius compute instance create \
      --parent-id "$NEBIUS_PROJECT" --name "$VM_NAME" \
      --labels project=halohalo,track=research \
      --resources-platform "$VM_PLATFORM" --resources-preset "$VM_PRESET" \
      "${preempt[@]}" \
      --boot-disk-attach-mode read_write \
      --boot-disk-managed-disk-name "$VM_NAME-boot" \
      --boot-disk-managed-disk-type network_ssd \
      --boot-disk-managed-disk-size-gibibytes "$VM_DISK_GIB" \
      --boot-disk-managed-disk-source-image-family-image-family "$VM_IMAGE_FAMILY" \
      --boot-disk-managed-disk-source-image-family-parent-id "$IMAGE_PARENT" \
      --network-interfaces "[{\"name\":\"eth0\",\"subnet_id\":\"$subnet\",\"ip_address\":{},\"public_ip_address\":{}}]" \
      --cloud-init-user-data "$userdata" \
      --format 'jsonpath={.metadata.id}'
    echo
    ;;
  STOPPED)
    echo "starting $VM_NAME ..."
    nebius compute instance start --id "$(vm_id)" >/dev/null
    ;;
  RUNNING) ;;
  *) echo "$VM_NAME is $state; waiting for it to settle" ;;
esac

for _ in $(seq 1 60); do
    [ "$(vm_state)" = RUNNING ] && [ -n "$(vm_ip)" ] && vm_ssh true 2>/dev/null && break
    sleep 10
done
vm_ssh 'cloud-init status --wait >/dev/null; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader' \
    || { echo "!! VM did not come up over SSH"; exit 1; }
echo "ready: $VM_NAME $(vm_ip) ($(vm_id))"
