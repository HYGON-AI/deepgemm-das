#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
# MegaMoE 8-hour random stress loop: balanced + imbalanced tokens, 3 shapes.
#
# Each round picks a RANDOM shape (default / V4 / V4.1) and then, with ~50%
# probability, either a balanced token count or an imbalanced per-rank
# config (pool borrowed from stress_imbalance_5h.sh). An interference
# program (op_test/interfere_spin_rotate.sh) loads the GPUs for the whole
# session; its output lands in <session>/interfere/. Per-run subdirs capture
# run.log + watchdog snapshots; hangs.txt is the running verdict summary.
#
# Capacity rules (same as perf matrix):
#   max tokens <= 128   -> DG_MAX_TOKENS=256   (LL; def shape overflows
#                                               RocSHMEM hint at 512)
#   max tokens <= 16384 -> DG_MAX_TOKENS=16384 (HT)
#   32768 -> 32768; 65536 -> 65536
#
# Usage: bash op_test/stress_balanced_8h.sh [hours=8] [log_root]
cd "$(dirname "$0")/.." || exit 1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export DG_HANG_WATCHDOG_S=${DG_HANG_WATCHDOG_S:-60}
export BENCH_ITER_BARRIER=${BENCH_ITER_BARRIER:-0}
export ROCSHMEM_GDR_DISABLE_XDP=${ROCSHMEM_GDR_DISABLE_XDP:-1}
export ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE:-7000000000}

HOURS=${1:-8}
ROOT=${2:-$(pwd)/megamoe/log/stress_balanced}
RUN_ID=$(date +%Y%m%d_%H%M%S)
L=$ROOT/$RUN_ID
mkdir -p "$L"

# Start GPU interference for the whole session; stop it on exit.
bash op_test/interfere_spin_rotate.sh "$L/interfere" &
INTF_PID=$!
cleanup() {
  kill -9 $INTF_PID 2>/dev/null
  pkill -9 -f interfere_spin_rotate 2>/dev/null
  pkill -9 -f spin_occupier 2>/dev/null
  pkill -9 -f test_mega_moe_fp8 2>/dev/null
  pkill -9 -f spawn_main 2>/dev/null
}
trap cleanup EXIT

echo "session $RUN_ID start $(date '+%F %T') hours=$HOURS iterbarrier=$BENCH_ITER_BARRIER heap=$ROCSHMEM_HEAP_SIZE watchdog=${DG_HANG_WATCHDOG_S}s" >> $ROOT/INDEX.txt

# name|DG_TEST_SHAPE (empty = default 7168x3072 topk6 e384)
SHAPES=(
    "def|"
    "v4|4096,2048,6,256"
    "v41|5120,2304,6,384"
)
# balanced token pool
BALANCED=(1 8 16 32 64 96 128 256 512 1024 2048 4096 8192 16384 32768 65536)
# imbalanced per-rank config pool (from stress_imbalance_5h.sh)
IMBALANCED=(
    "8,1,1,1,1,1,1,1" "1,1,1,1,1,1,1,8"
    "16,1,1,1,1,1,1,1" "1,1,1,1,1,1,1,16"
    "16,0,0,0,0,0,0,0" "16,8,4,2,1,1,1,1"
    "96,1,1,1,1,1,1,1" "1,1,1,1,1,1,1,96"
    "128,1,1,1,1,1,1,1" "1,1,1,1,1,1,1,128"
    "96,0,0,0,0,0,0,0" "128,96,64,32,16,8,4,2"
    "1,1,1,1,1,1,1,512" "512,1,1,1,1,1,1,1" "384,128,64,32,16,8,4,1"
    "1,1,1,1,1,1,1,2048" "2048,1,1,1,1,1,1,1" "2048,0,0,0,0,0,0,0"
    "2048,1024,512,256,128,64,32,16"
    "1,1,1,1,1,1,1,4096" "4096,1,1,1,1,1,1,1" "1,1,1,1,1,1,1,32"
    "1,32,1,1,1,1,1,512" "512,32,32,32,32,32,32,32"
    "512,0,0,0,0,0,0,0" "256,0,128,0,64,0,32,0"
    "1,1,1,1,1,1,1,513" "513,1,1,1,1,1,1,1"
    "1,1,1,1,1,1,1,8192" "8192,1,1,1,1,1,1,1"
    "8192,0,0,0,0,0,0,0"
    "8192,4096,2048,1024,512,256,128,64"
    "64,128,256,512,1024,2048,4096,8192"
    "8192,0,4096,0,2048,0,1024,0"
    "4096,4096,4096,4096,0,0,0,0"
    "1,1,1,1,1,1,1,16384" "16384,1,1,1,1,1,1,1"
    "16384,0,0,0,0,0,0,0"
    "16384,16384,1,1,1,1,1,1" "1,1,1,1,1,1,16384,16384"
)

cap_for() {
    local t=$1
    if [ $t -le 128 ]; then echo 256
    elif [ $t -le 16384 ]; then echo 16384
    else echo $t; fi
}

DEADLINE=$(( $(date +%s) + HOURS*3600 ))
i=0
while [ $(date +%s) -lt $DEADLINE ]; do
    i=$((i+1))
    [ $(date +%s) -ge $DEADLINE ] && break
    s=${SHAPES[$((RANDOM % ${#SHAPES[@]}))]}
    sname=${s%%|*}; sval=${s#*|}
    # ~50/50 balanced vs imbalanced
    if [ $((RANDOM % 2)) -eq 0 ]; then
        tk=${BALANCED[$((RANDOM % ${#BALANCED[@]}))]}
        cfg=$(yes "$tk" | head -8 | paste -sd,)
        kind=bal
    else
        cfg=${IMBALANCED[$((RANDOM % ${#IMBALANCED[@]}))]}
        kind=imb
    fi
    cfgmax=$(echo $cfg | tr ',' '\n' | sort -n | tail -1)
    cap=$(cap_for $cfgmax)
    if [ $cfgmax -ge 8192 ]; then TMO=900; else TMO=400; fi
    tag=$(echo $cfg | tr ',' '_')
    for attempt in 1 2; do
        [ $(date +%s) -ge $DEADLINE ] && break
        port=$(( 29500 + (i*64) % 400 + attempt*401 ))
        TD=$L/t${i}_${kind}_${sname}_${cfgmax}_try${attempt}
        mkdir -p $TD
        export DG_WATCHDOG_DIR=$TD/wd
        t0=$(date +%s)
        MASTER_PORT=$port DG_MAX_TOKENS=$cap DG_TOKEN_COUNTS="$cfg" \
        env ${sval:+DG_TEST_SHAPE=$sval} \
            timeout $TMO python3 op_test/test_mega_moe_fp8.py 8 $cfgmax \
            > $TD/run.log 2>&1
        rc=$?
        dt=$(( $(date +%s) - t0 ))
        npass=$(grep -c "async mode True PASS" $TD/run.log)
        if [ $rc -eq 124 ]; then verdict=REPRO-HANG;
        elif [ $rc -eq 0 ] && [ "$npass" -ge 8 ]; then verdict=PASS;
        else verdict=BAD; fi
        echo "verdict=$verdict rc=$rc PASS8=$npass dt=${dt}s kind=$kind shape=$sname cfg=[$cfg] cap=$cap start=$(date -d @$t0 '+%F %T')" > $TD/result.txt
        cat $TD/result.txt | sed "s|^|i=$i try=$attempt |" >> $L/hangs.txt
        pkill -9 -f test_mega_moe_fp8 2>/dev/null; pkill -9 -f spawn_main 2>/dev/null
        [ $rc -eq 124 ] && break
        [ $rc -eq 0 ] && [ "$npass" -ge 8 ] && break
        sleep 8
    done
    sleep 3
done
echo "session $RUN_ID end $(date '+%F %T')" >> $ROOT/INDEX.txt
