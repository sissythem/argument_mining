# -*- coding: utf-8 -*-
from ellogon import tcl, package_require, text2tcl

package_require("ELEP::RunOnText::HTokenizer")
package_require("ELEP::RunOnText::OneSentencePerLine")
## Create the processor
tokeniser = tcl.call("::ELEP::RunOnText::HTokenizerExtendedUnicode", "new")
tokeniser_spans = tcl.call("::ELEP::RunOnText::HTokenizerExtendedUnicode", "new")
tcl.call(tokeniser_spans, "configure", "-report_spans", "1")
tokeniser_type_spans = tcl.call("::ELEP::RunOnText::HTokenizerExtendedUnicode", "new")
tcl.call(tokeniser_type_spans, "configure", "-report_spans", "1", "-report_type", "1")
tokeniser_no_punc = tcl.call("::ELEP::RunOnText::HTokenizer::RejectTypes", "new")
tcl.call(tokeniser_no_punc, "reject_punctuation")

one_sentence_per_line = tcl.call("ELEP::RunOnText::OneSentencePerLine", "new")

def tokenise(text):
    return tcl.call(tokeniser, "process", text2tcl(text))

def tokenise_spans(text):
    return tcl.call(tokeniser_spans, "process", text2tcl(text))

def tokenise_type_spans(text):
    return tcl.call(tokeniser_type_spans, "process", text)
    return tcl.call(tokeniser_type_spans, "process", text2tcl(text))

def tokenise_no_punc(text):
    return tcl.call(tokeniser_no_punc, "process", text2tcl(text))

def stop_words(language="greek"):
    package_require("ELEP::TokenUtilities::StopWord")
    sw = tcl.call("::ELEP::TokenUtilities::StopWord", "new", "-languages", language)
    words = tcl.call(sw, "getWords")
    tcl.call(sw, "destroy")
    return words

stop_words_el = {w:0 for w in stop_words()}

def remove_stop_words(words):
    return [w for w in words if w.lower() not in stop_words_el]

def sentences(text):
    return tcl.call(one_sentence_per_line, "process", text2tcl(text))
