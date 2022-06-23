from ellogon import tokeniser, tcl_version

print("Tcl in use:", tcl_version());
text = """
😀😇🥰💠🔶.
🇬🇷.
 Η, μέχρι σήμερα, εντελώς ατεκμηρίωτη πολιτική απόφαση 😀 🌑.
"""

print(text)
print("Sentences/Tokens (with type & span offsets):")
for sentence in tokeniser.tokenise_type_spans(text):
    print(sentence)
