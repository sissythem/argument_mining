#!/bin/env bash

# run mode; either 'rest' or 'annotation'
mode='periodic'
imgname='debatelab'
HOST_PORT=8001
models_dirpath="$(pwd)/models"
name="debatelab-${mode}"

[[ $# -gt 0 ]] && models_dirpath="$1"
[[ $# -gt 1 ]] && imgname="$2"
[[ $# -gt 2 ]] && name="$3"
[[ $# -gt 3 ]] && HOST_PORT=$4
[[ $# -gt 4 ]] && mode="$5"

# [[ ! -d "${models_dirpath}/adu" ]] && echo "Can't find ADU model in $models_dirpath" && exit 1
# [[ ! -d "${models_dirpath}/rel" ]] && echo "Can't find REL model in $models_dirpath" && exit 1
# [[ ! -d "${models_dirpath}/stance" ]] && echo "Can't find STANCE model in $models_dirpath" && exit 1

# if you have permission issues with the mounted volume, append a :z
# e.g. -v "host_path:container_path:z"
# https://stackoverflow.com/questions/24288616/permission-denied-on-accessing-host-directory-in-docker

# docker run --rm \
# docker run --restart "unless-stopped" \

# network
[[ -z $(docker network ls | grep debatelab ) ]] && echo "Creating 'debatelab' docker network." &&  docker network create debatelab

docker run \
  -dit --name $name -p "$HOST_PORT:8000" -v "${models_dirpath}:/models"  -e RUN_MODE="$mode" "$imgname"

docker network connect debatelab "${name}"
