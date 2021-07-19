FROM tiangolo/uvicorn-gunicorn-fastapi

RUN apt-get update && apt-get install -y build-essential unzip vim git curl wget \
                                        subversion tcl8.6 tcllib  tcl-tls  tcl-trf \
                                        tcl-udp tcl-vfs  tdom  tcl-tclex tcl-memchan \
                                        itcl3 tcl-signal tcl-sugar libpgtcl \
                                        libsqlite-tcl mysqltcl tcl8.6-tdbc \
                                        tcl8.6-tdbc-mysql tcl8.6-tdbc-postgres \
                                        tcl8.6-tdbc-sqlite3 critcl

RUN wget http://ports.ubuntu.com/pool/universe/c/combat/tcl-combat_0.8.1-1_all.deb
RUN dpkg -i tcl-combat_0.8.1-1_all.deb

# Compile & install rl_json library
RUN cd && \
    git clone https://github.com/RubyLane/rl_json.git && \
    cd rl_json && \
    ./configure && \
    make install && \
    rm -rf ~/rl_json

# Ellogon install (to test run: tclsh /opt/Ellogon/ellogon)
RUN cd /opt && \
    mkdir Ellogon && \
    chown -R root:root Ellogon && \
    cd /opt && \
    svn checkout svn://svn.code.sf.net/p/ellogon/code/runtime Ellogon --quiet

RUN cd /app

# Copy the application
ARG RELEASE=main

RUN git clone --branch $RELEASE --single-branch https://docker:PGAveGZ3yR7myXwZuZZe@gitlab.com/debatelab/argument-mining.git
RUN rm -rf /app/argument-mining/.git/
RUN rm -rf /app/argument-mining/docs/
RUN python3 -m pip install -r /app/argument-mining/requirements.txt

EXPOSE 80
WORKDIR /app/argument-mining/app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
# CMD tail -f /dev/null
