"""
ICAP server deployment script
"""

import logging
import random
import socketserver

from pyicap import *

logging.getLogger().level = logging.INFO

class ThreadingSimpleServer(socketserver.ThreadingMixIn, ICAPServer):
    pass

class ICAPHandler(BaseICAPRequestHandler):

    def echo_OPTIONS(self):
        breakpoint()
        self.set_icap_response(200)
        self.set_icap_header('Methods', 'RESPMOD')
        self.set_icap_header('Preview', '0')
        self.send_headers(False)

    def echo_RESPMOD(self):
        breakpoint()
        self.no_adaptation_required()

    def echo_REQMOD(self):
        breakpoint()
        self.no_adaptation_required()

port = 1344

server = ThreadingSimpleServer(('', port), ICAPHandler)
try:

    logging.info(f"Launching at port {port}")
    while 1:
        server.handle_request()
except KeyboardInterrupt:
    print("Finished")
