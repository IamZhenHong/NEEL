#!/usr/bin/env bash
# podssh.sh HOST PORT 'remote command'
exec ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i ~/.ssh/id_ed25519 -p "$2" root@"$1" "$3"
