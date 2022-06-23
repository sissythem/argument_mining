# -*- coding: utf-8 -*-
import hfst

filename = "/opt/Intellitech/projects/Opinion/common/Intellitech_Opinion_OpinionPhraseIdentification/lists/greek/lemmas/minlemmas.fst"
istr = hfst.HfstInputStream(filename)
transducers = []
while not (istr.is_eof()):
    transducers.append(istr.read())
istr.close()
print("Read %i transducers in total." % len(transducers))
analyser = transducers[0]
analyser.lookup_optimize()

def lemmas_default(token, default=None):
    lemmas = analyser.lookup(token)
    if lemmas: return lemmas[0][0]
    lemmas = analyser.lookup(token.lower())
    if lemmas: return lemmas[0][0]
    return default

def lemmas(token):
    lemmas = analyser.lookup(token)
    if lemmas: return lemmas[0][0]
    lemmas = analyser.lookup(token.lower())
    if lemmas: return lemmas[0][0]
    return token

def perform_lemmatisation(words):
    return [lemmas(w) for w in words]

def process_doc(doc):
    ndoc = []
    for sentence in doc:
        s = []
        for token in sentence:
            token['lemma'] = lemmas(token['token'])
            s.append(token)
        ndoc.append(s)
    return ndoc
