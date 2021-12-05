from ellogon import tokeniser
from app.src.utils import inject_missing_gaps


def test_document_text_reconstruction():
    """Text on filling offset gaps introduced by omitted whitespace / punctuation splitting
        by tokenization
    """
    texts = [
        " \"  Ένα δύο, τρία ! ΤΕΣΣΕΡΑ + πέντε/έξι - επτά, έτσι:   One two three? Four! and"
    ]

    for te in texts:
        parts = tokeniser.tokenise_spans(te)
        processed_parts = []
        current_idx = 0
        for i, part in enumerate(parts):
            x = inject_missing_gaps(part, starting_idx=current_idx)
            processed_parts.append(x)
            # update starting index
            current_idx = x[-1]

            joined = utils.expanded_tokens_to_text(processed_parts)
            assert joined == te[:
                                current_idx], f"Mismatch in stiched tokens and original text, after consuming sentence # {i+1}/{len(parts)}!"

        joined = utils.expanded_tokens_to_text(processed_parts)
        assert joined == te, "Mismatch in stiched tokens and entire original text!"

        # join tokens
        reconc_sentences = []
        for pp in processed_parts:
            sentence = utils.expanded_tokens_to_text(pp)
            # tok_tuples = pp[0]
            # toks = [x[0] for x in tok_tuples]
            # sentence = "".join(toks)
            reconc_sentences.append(sentence)
        reconc_text = "".join(reconc_sentences)
        assert reconc_text == te, "Reconstrution error!"
        print()


if __name__ == "__main__":
    test_document_text_reconstruction()
