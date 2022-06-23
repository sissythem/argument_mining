
from ellogon import tcl, package_require, text2tcl

package_require("ELEP::TokenUtilities::StopWord")

class StopWord:
    tclobj = None;

    def __init__(self, *args, **kwargs):
        arguments = list(args)
        for key in kwargs:
            arguments.append("-"+key)
            arguments.append(kwargs[key])
        self.tclobj = tcl.call("ELEP::TokenUtilities::StopWord", \
                               "new", *arguments)

    def __del__(self):
        if self.tclobj:
            tcl.call(self.tclobj, "destroy")

    def call(self, method, *args, **kwargs):
        if self.tclobj == None:
            raise "empty object"
        arguments = list(args)
        for key in kwargs:
            arguments.append("-"+key)
            arguments.append(kwargs[key])
        print(self.tclobj, method, *arguments)
        return tcl.call(self.tclobj, method, *arguments)

    def resetModel(self):
        return self.call("resetModel")

    def loadLanguage(self, language):
        return self.call("loadLanguage", language)

    def languages(self):
        return self.call("languages")

    def isStopWord(self, token):
        return self.call("stopWord?", token)

    def isNumber(self, token):
        return self.call("number?", token)

    def isNumberAndSymbols(self, token):
        return self.call("numberAndSymbols?", token)

    def getWords(self):
        return self.call("getWords")


# st = StopWord(languages="greek")
# print(st.isStopWord("και"))
# print(st.getWords())
