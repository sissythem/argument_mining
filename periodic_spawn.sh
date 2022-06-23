#!/bin/env bash

srcdir="path/to/debatelab/sources"
if [ "$srcdir" == "path/to/debatelab/sources" ]; then
  echo "Need to set debatelab sources!" ; exit 1;
fi

# script for periodic check & (re)spawning of a container
echo "Running at $(date)"
name="debatelab-ellogon"
cd $srcdir/container
spawn_script="run_annotation_container.sh"

[[ $# -gt 0 ]] && name=$1
relevant=$(docker ps -a --filter "name=${name}" | tail -n +2)
echo "docker ps -a --filter 'name=${name}' | tail -n +2"
echo "[$relevant]"
if [ ! "${relevant}" == "" ]; then
  # something relevant exists
  # if not running
  restart_when=("created" "paused" "exited" "dead")
  echo "Relevant container exists -- checking for statuses: ${restart_when}"
  recreated=0
  for rw in ${restart_when[@]} ; do
    relevant=$(docker ps -a --filter "name=${name}" --filter "status=${rw}" | tail -n +2)
    echo "Checking $rw: [$relevant]"
    if [ ! "${relevant}" == "" ]; then
      echo "Detected container status: $rw"
      recreated=1
      if [ "$rw" == "created" ] || [ "$rw" == "exited" ]; then echo "Starting"; docker start "$name";
      elif [ "$rw" == "paused" ]; then echo "unpausing"; docker unpause "$name";
      else
        echo "running";
        bash $spawn_script;
      fi
      break
    fi
  done
  if [ $recreated -eq 0 ]; then echo "No need to recreate"; fi
else
  echo "No relevant container exists -- running new..."
  # no such container exists; just start it
  bash $spawn_script;
  echo "Run complete."
fi

