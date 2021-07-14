FROM tiangolo/uvicorn-gunicorn-fastapi

RUN apt-get update && apt-get install -y build-essential unzip vim git curl

# Copy the application
ARG RELEASE=main

RUN git clone --branch $RELEASE --single-branch https://docker:EMtm3PdDnpx8CX8wGs2q@gitlab.com/debatelab/argument-mining.git
RUN python3 -m pip install -r ./argument-mining/requirements.txt

EXPOSE 80
WORKDIR ./argument-mining/app

CMD docker-entrypoint.sh
#CMD tail -f /dev/null
