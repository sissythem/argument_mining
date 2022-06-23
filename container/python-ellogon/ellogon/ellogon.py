# -*- coding: utf-8 -*-
from tkinter import Tcl
import re
import unicodedata
tcl = Tcl()

## Load Ellogon...
r = tcl.call("::tcl::tm::path", "add", "/opt/Ellogon/ellogon2.0/tm")
r = tcl.call("package", "require", "ellogon")

def PUL_LoadPluginsInDir(dir):
    tcl.call("::PUL_LoadPluginsInDir", dir)

def PUL_LoadPluginsInDirs(dirlist):
    for dir in dirlist: tcl.call("::PUL_LoadPluginsInDir", dir)

_nonbmp = re.compile(r'[\U00010000-\U0010FFFF]')

def _surrogatepair(match):
    char = match.group()
    assert ord(char) > 0xffff
    encoded = char.encode('utf-16-le')
    return (
        chr(int.from_bytes(encoded[:2], 'little')) +
        chr(int.from_bytes(encoded[2:], 'little')))

def text2tcl(text):
    # https://stackoverflow.com/questions/36283818/remove-characters-outside-of-the-bmp-emojis-in-python-3
    return ''.join(c for c in unicodedata.normalize('NFC', text) if c <= '\uFFFF')
    return _nonbmp.sub(_surrogatepair, text)

def package_require(pkg):
    tcl.call("package", "require", pkg)

def tcl_version():
    return tcl.call("set", "tcl_patchLevel")

## Load some additional modules...
ComponentsHome = '/opt/Intellitech/projects'
for dir in ['NERC/common', 'NERC/tag', 'Opinion/common']:
    PUL_LoadPluginsInDir(ComponentsHome + '/' + dir)
r = tcl.call("tcl::tm::path", "add", "/opt/Intellitech/projects/tm")
r = tcl.call("tcl::tm::path", "add", "/opt/Intellitech/projects/SocialWebObservatory/tm")
