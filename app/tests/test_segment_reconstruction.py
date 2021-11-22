from ellogon import tokeniser


def inject_missing_gaps(token_idx_list, starting_idx=0):
    """Modify list of tokens so that all indices are consequtive

    Args:
        token_idx_list (list): List of tuples, each element is
        starting_idx (int): Initial index
    """
    if type(token_idx_list[0][0]) is not str:
        l = []
        for part in token_idx_list:
            if type(part) is not int:
                # it's a tuple of tokens
                part = inject_missing_gaps(part, starting_idx=starting_idx)
            else:
                pass
            l.append(part)
        # update starting index
        l[1] = l[0][0][1]
        return tuple(l)
    else:
        res = []
        current = starting_idx
        # we are at a lowest-level tuple
        for part in token_idx_list:
            txt, start, end = part
            diff = start - current
            if diff != 0:
                res.append((" "*diff, current, start))
            res.append(part)
            current = end
        return res


def test_document_text_reconstruction():
    """Text on filling offset gaps introduced by omitted whitespace / punctuation splitting
        by tokenization
    """
    texts = [
        " \"  Ένα δύο, τρία ! ΤΕΣΣΕΡΑ + πέντε/έξι - επτά  . One two three? Four! and"
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

            joined = "".join([y[0] for x in processed_parts for y in x[0]])
            assert joined == te[:
                                current_idx], f"Mismatch in stiched tokens and original text, after consuming sentence # {i+1}/{len(parts)}!"

        joined = "".join([y[0] for x in processed_parts for y in x[0]])
        assert joined == te, "Mismatch in stiched tokens and entire original text!"
