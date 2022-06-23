FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# Ellogon
#########

# Ellogon install (to test run: tclsh /opt/Ellogon/ellogon)
RUN cd /opt && \
    mkdir Ellogon && \
    chown -R root:root Ellogon && \
    cd /opt && \
    svn checkout svn://svn.code.sf.net/p/ellogon/code/runtime Ellogon --quiet && cd /opt/Ellogon && svn update


# add pretrained models
# models should be added via a mounted volume to /models during runtime

# get python 3.7
RUN cd / && echo "Fetching python3.7" && \
  wget -q https://www.python.org/ftp/python/3.7.12/Python-3.7.12.tgz && \
  tar xf Python-3.7.12.tgz &&  cd Python-3.7.12 && ./configure && make && make install

RUN echo "Using python:" && python --version

# get repo
RUN ls
RUN git clone "https://docker:31BTUPDphFu2mE_sAaZ-@gitlab.com/npit/argument-mining-v2.git" /code
RUN cd /code && git checkout develop

# yoink dependencies
RUN cd /code && python -m pip install -r requirements.txt

# ellogon python package
RUN apt-get update && apt-get -y install tcllib
ENV PYTHON_ELLOGON_PATH="/mnt/python-ellogon"
ADD python-ellogon /mnt/python-ellogon 
RUN python -m pip install -e /mnt/python-ellogon
RUN python -c "from ellogon import tokeniser; tokeniser.tokenise_spans('Ένα δύο, τρία! Τέσσερα; Ναι!')" && echo "Installed ellogon"

# cron
RUN apt install -y cron
# TODO move near model addition
# model config

# debugging stuff
RUN apt-get install -y vim tmux jq
RUN echo "set -o vi" >> /root/.bashrc

RUN mkdir /models

#######

# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
WORKDIR "/code"
CMD ["bash", "container/entrypoint.sh"]
# # CMD tail -f /dev/null
