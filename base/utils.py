def wrap_and_pad_tokens(inputs, prefix, suffix, seq_len, padding, pad=True, truncate=True):
    """

    :param inputs: tokens list to process
    :param prefix: prefix to prepend
    :param suffix: suffix to append
    :param seq_len: expected final sequence length
    :param padding: symbol to use for padding
    :param pad: whether to pad or not
    :param truncate: whether to truncate or not
    :return:
    """
    out = [prefix] + inputs
    if truncate:
        out = out[:seq_len - 1]
    if pad:
        out += [padding] * (seq_len - 1 - len(out))
    out += [suffix]
    return out
