FROM nginx

RUN apt-get update && apt-get install -y build-essential supervisor unzip uwsgi python3 python3-pip python3-setuptools \
    procps vim git curl libxml2-dev libxslt-dev libcurl4-openssl-dev libssl-dev python-pycurl

USER root
RUN groupadd appusers
RUN usermod -a -G appusers nginx

ENV HOME=/opt/repo

# Copy the application
ARG RELEASE=master

RUN git clone --branch $RELEASE --single-branch https://docker:EMtm3PdDnpx8CX8wGs2q@gitlab.com/debatelab/argument-mining.git
RUN python3 -m pip install -r ./argument-mining/requirements.txt
RUN mkdir ${HOME}/src
RUN cp -r ./argument-mining/app/* ${HOME}/src/
RUN rm -rf argument-mining/

# Copy the configuration file from the current directory and paste
# it inside the container to use it as Nginx's default config.
COPY ./nginx.conf /etc/nginx/nginx.conf

# setup NGINX config
RUN mkdir -p /spool/nginx /run/pid && \
    chmod -R 777 /var/log/nginx /var/cache/nginx /etc/nginx /var/run /run /run/pid /spool/nginx && \
    chgrp -R 0 /var/log/nginx /var/cache/nginx /etc/nginx /var/run /run /run/pid /spool/nginx && \
    chmod -R g+rwX /var/log/nginx /var/cache/nginx /etc/nginx /var/run /run /run/pid /spool/nginx && \
    rm /etc/nginx/conf.d/default.conf

# Copy the base uWSGI ini file to enable default dynamic uwsgi process number
COPY ./uwsgi.ini /etc/uwsgi/apps-available/uwsgi.ini
RUN ln -s /etc/uwsgi/apps-available/uwsgi.ini /etc/uwsgi/apps-enabled/uwsgi.ini

COPY ./supervisord.conf /etc/supervisor/conf.d/supervisord.conf
RUN touch /var/log/supervisor/supervisord.log

EXPOSE 80:80

# setup entrypoint
COPY ./docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN rm docker-entrypoint.sh

# https://github.com/moby/moby/issues/31243#issuecomment-406879017
RUN ln -s /usr/local/bin/docker-entrypoint.sh / && \
    chmod 777 /usr/local/bin/docker-entrypoint.sh && \
    chgrp -R 0 /usr/local/bin/docker-entrypoint.sh && \
    chown -R nginx:root /usr/local/bin/docker-entrypoint.sh

# https://docs.openshift.com/container-platform/3.3/creating_images/guidelines.html
RUN chgrp -R 0 /var/log /var/cache /run/pid /spool/nginx /var/run /run /tmp /etc/uwsgi /etc/nginx && \
    chmod -R g+rwX /var/log /var/cache /run/pid /spool/nginx /var/run /run /tmp /etc/uwsgi /etc/nginx && \
    chown -R nginx:root ${HOME} && \
    chmod -R 777 ${HOME} /etc/passwd

RUN echo "nginx ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# enter
WORKDIR ${HOME}
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["supervisord"]
#CMD tail -f /dev/null
