
name='debatelab'
[[ $# -gt 0 ]] && name="$1"

docker stop "$name";
docker rm "$name"
