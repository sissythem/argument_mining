from ellogon import tokeniser, postagger, lemmatiser
text = """
Για το ellogon, πού μπορώ να βρω τα πακέτα python;
"""
print("Sentences/Tokens:")
for sentence in tokeniser.tokenise(text):
  print(sentence)
print('-----------------------------------------------------')
print("Sentences/Tokens (with span offsets):")
for sentence in tokeniser.tokenise_spans(text):
    print(sentence)
print('-----------------------------------------------------')
print("Part of speech tagging:")
for sentence in postagger.pos(text):
    print(sentence)
print("Part of speech tagging: (Universal tagset)")
for sentence in postagger.universal_pos(text):
    print(sentence)
print('-----------------------------------------------------')
print("Lemmas (LEXICON-based):")
for sentence in tokeniser.tokenise(text):
    for word in sentence:
        print(f'word: "{word}", lemmas: {lemmatiser.lemmas(word)}')
