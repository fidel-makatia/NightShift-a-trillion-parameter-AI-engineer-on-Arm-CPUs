#!/usr/bin/env bash
# NightShift VM cost control. An E96ps_v6 costs ~$6/hr while running — deallocate when idle.
# Usage: ./vm.sh {start|stop|status|ssh|watch-download}
set -euo pipefail
RG=nightshift-rg
VM=nightshift-vm
IP=20.186.16.79

case "${1:-status}" in
  start)  az vm start -g $RG -n $VM && echo "running" ;;
  stop)   az vm deallocate -g $RG -n $VM && echo "deallocated (compute billing stopped)" ;;
  status) az vm get-instance-view -g $RG -n $VM \
            --query "instanceView.statuses[?starts_with(code,'PowerState')].displayStatus" -o tsv ;;
  ssh)    exec ssh nightshift@$IP ;;
  watch-download) ssh nightshift@$IP 'watch -n 30 "tail -2 /data/download.log; df -h /data | tail -1"' ;;
  *) echo "usage: $0 {start|stop|status|ssh|watch-download}"; exit 1 ;;
esac
