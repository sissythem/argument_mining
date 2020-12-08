import torch


def wrap_and_pad_tokens(inputs, prefix, suffix, seq_len, padding, pad=True, truncate=True):
    out = [prefix] + inputs
    if truncate:
        out = out[:seq_len - 1]
    out = out + [suffix]
    if pad:
        out += [padding] * (seq_len - 1 - len(out))
    return out


def add_padding(tokens, labels=None, max_len=512, pad_token=0):
    diff = max_len - tokens.shape[-1]
    if diff < 0:
        tokens = tokens[:, :max_len]
        if labels:
            labels = labels[:, :max_len]
    else:
        padding = torch.ones((1, diff), dtype=torch.long) * pad_token
        tokens = torch.cat([tokens, padding], dim=1)
        if labels:
            labels = torch.cat([labels, padding], dim=1)
    return tokens, labels


def remove_padding(tokens, predictions, pad_token=0):
    new_tokens, new_predictions = [], []
    for idx, token in enumerate(tokens):
        if (token != pad_token) and (token != 101) and (token != 102):
            new_tokens.append(token)
            new_predictions.append(predictions[idx])
    return new_tokens, new_predictions
