#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
# GPU interference generator for MegaMoE stress runs (hang_debug_plan.md 5.1).
#
# Rotates spin_occupier across all GPUs: occupy 15s, idle 15s, next device.
# Block count sweeps 30..72 to cover both partial and near-full occupancy,
# which maximizes cross-rank launch skew (the hang amplification factor).
#
# Usage: bash op_test/interfere_spin_rotate.sh [log_dir]
# Stop:  pkill -f interfere_spin_rotate; pkill -9 -f spin_occupier
# Note: a killed host process can leave the spinning kernel resident on the
# GPU for a few seconds -- verify with hy-smi (HCU% back to 0) before reuse.
cd "$(dirname "$0")/.." || exit 1
LOGD=${1:-$(pwd)/megamoe/log/stress}
mkdir -p "$LOGD"
LOG=$LOGD/interfere_$(date +%Y%m%d_%H%M%S).log

# Build the occupier on first use.
if [ ! -x op_test/spin_occupier ]; then
  hipcc -O2 --offload-arch=gfx938 op_test/spin_occupier.hip -o op_test/spin_occupier || exit 1
fi

dev=0
nb=30
echo "interfere start $(date '+%F %T')" >> "$LOG"
while true; do
  ./op_test/spin_occupier $dev $nb >> "$LOG" 2>&1 &
  SPID=$!
  sleep 15
  kill -9 $SPID 2>/dev/null
  pkill -9 -f spin_occupier 2>/dev/null
  sleep 15
  dev=$(( (dev + 1) % 8 ))
  nb=$(( nb + 1 )); [ $nb -gt 72 ] && nb=30
done
