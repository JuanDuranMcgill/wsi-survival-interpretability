#!/bin/bash
# Starts (or restarts) the two runner services in the background. Safe to call repeatedly: each service holds a lock,
# so a second call does nothing. Used by hand and by the @reboot crontab entry. Never pushes anything.
ROOT=/data/processed/juan/pancan; LOGS=$ROOT/logs; RUN=$ROOT/run
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/home/users/juan/envs/pilot/bin/python
export TMPDIR=/data/scratch/juan/tmp HF_HOME=/data/scratch/juan/cache/huggingface HF_HUB_OFFLINE=1
# after a reboot, wait for the data disks
for i in $(seq 1 60); do [ -d /data/storage/juan ] && [ -d /data/processed/juan ] && [ -d /data/scratch/juan ] && break; sleep 10; done
[ "$1" = "--boot" ] && sleep 60
mkdir -p $LOGS $RUN $TMPDIR
for svc in download process; do
  setsid nohup flock -n $RUN/$svc.lock bash -c '
    fails=0
    while true; do
      s=$(date +%s)
      '"$PY"' -I '"$HERE"'/pancan_runner.py '"$svc"' >> '"$LOGS"'/'"$svc"'.log 2>&1
      rc=$?
      [ $rc -eq 0 ] && { echo "$(date) '"$svc"' finished (exit 0)" >> '"$LOGS"'/'"$svc"'.log; break; }
      [ $(( $(date +%s) - s )) -lt 120 ] && fails=$((fails+1)) || fails=0
      echo "$(date) '"$svc"' exited rc=$rc, restarting (quick fails $fails)" >> '"$LOGS"'/'"$svc"'.log
      sleep $(( fails > 5 ? 600 : 30 ))
    done' >/dev/null 2>&1 &
done
