#!/bin/env bash

# run mode; either 'rest' or 'annotation'
mode='rest'
imgname='debatelab'
name="debatelab-ics"
HOST_PORT=8000
models_dirpath="$(pwd)/models/"


[[ $# -gt 0 ]] && mode="$1"
[[ $# -gt 1 ]] && imgname="$2"
[[ $# -gt 2 ]] && name="$3"
[[ $# -gt 3 ]] && HOST_PORT=$4
[[ $# -gt 4 ]] && models_dirpath="$5"


# if you have permission issues with the mounted volume, append a :z
# e.g. -v "host_path:container_path:z"
# https://stackoverflow.com/questions/24288616/permission-denied-on-accessing-host-directory-in-docker
# docker run --rm \
# docker run \
docker run --restart "unless-stopped" \
  -dit --name $name -p "$HOST_PORT:8000" -v "${models_dirpath}:/models/"  -e RUN_MODE="$mode" "$imgname"
# echo "Getting container tty..."
# docker exec -it $name bash
