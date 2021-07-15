FROM tiangolo/uvicorn-gunicorn-fastapi

RUN apt-get update && apt-get install -y build-essential unzip vim git curl

# Copy the application
ARG RELEASE=main

RUN git clone --branch $RELEASE --single-branch https://docker:EMtm3PdDnpx8CX8wGs2q@gitlab.com/debatelab/argument-mining.git
RUN rm -rf ./argument-mining/.git/
RUN rm -rf ./argument-mining/docs/
RUN python3 -m pip install -r ./argument-mining/requirements.txt

EXPOSE 80
WORKDIR ./argument-mining/app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
# CMD tail -f /dev/null
