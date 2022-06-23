port=1344
[[ $# -gt 1 ]] && port=$1
c-icap-client -s echo -i 127.0.0.1 -p 1344 -d 10
