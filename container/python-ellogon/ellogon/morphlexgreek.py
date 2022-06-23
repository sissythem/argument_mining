# -*- coding: utf-8 -*-
from ellogon import tcl, package_require, text2tcl

package_require("morphlex::greek")
package_require("ELEP::Morphology::Lexicon::greek")

db_lexicon = tcl.call("ELEP::Morphology::Lexicon::greek::Store", "new")
tcl.call(db_lexicon, "configure", "-driver", "mysql", "-host", "127.0.0.1",
    "-port", "3306", "-database", "MorphologicalLexicon",
    "-user", "MorphologicalLexicon", "-password", "18D1bsIN4d2FsSVj")

def lemmas(form, pos_pattern="*"):
    form = text2tcl(form)
    try:
        lemmas = tcl.call(db_lexicon, "getLemmas", form, pos_pattern)
    except:
        lemmas = None
    return lemmas
