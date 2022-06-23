# -*- coding: utf-8 -*-
from ellogon import tcl, package_require, text2tcl

package_require("ELEP::RunOnText::HBrill")
## Create the processor
postagger = tcl.call("::ELEP::RunOnText::HBrill", "new")

def pos(text):
    return tcl.call(postagger, "process", text2tcl(text))

def HBrill_to_universal(tag):
    if tag.startswith('JJ'):
        return 'ADJ'
    elif tag.startswith('CD'):
        return 'NUM'
    elif tag.startswith('IN') or tag.startswith('CC') or tag.startswith('RP'):
        return 'ADP'
    elif tag.startswith('RB'):
        return 'ADV'
    elif tag.startswith('MD'):
        return 'AUX'
    elif tag.startswith('CC'):
        return 'CCONJ'
    elif tag.startswith('DDT') or tag.startswith('IDT'):
        return 'DET'
    elif tag.startswith('UH'):
        return 'INTJ'
    elif tag.startswith('NN'):
        return 'NOUN'
    elif tag.startswith('VBP') or tag.startswith('VBG'):
        return 'PART'
    elif tag.startswith('PRP') or tag.startswith('PP') or tag.startswith('REP') \
      or tag.startswith('DP')  or tag.startswith('IP') or tag.startswith('WP') \
      or tag.startswith('QP')  or tag.startswith('INP'):
        return 'PRON'
    elif tag.startswith('NNP'):
        return 'PROPN'
    elif tag.startswith('.') or tag.startswith(',') or tag.startswith(':') \
      or tag.startswith(';') or tag.startswith('!') or tag.startswith('(') \
      or tag.startswith(')') or tag.startswith('"'):
        return 'PUNCT'
    elif tag.startswith('SYM'):
        return 'SYM'
    elif tag.startswith('VB'):
        return 'VERB'
    elif tag.startswith('FW') or tag.startswith('AB') or tag.startswith('LS'):
        return 'X'
    return tag

def universal_pos(text):
    ndoc = []
    for sentence in pos(text):
        s = []
        for token in sentence:
            t = {}
            t['token'] = token[0]
            t['pos']   = HBrill_to_universal(token[1])
            s.append(t)
        ndoc.append(s)
    return ndoc
