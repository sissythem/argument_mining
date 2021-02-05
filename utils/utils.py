def name_exceeds_bytes(name):
    """
    Checks if a string exceeds the 255 bytes

    Args
        name (str): the name of a file

    Returns
        bool: True/False
    """
    return utf8len(name) >= 255


def utf8len(s):
    """
    Find the length of the encoded filename

    Args
        s (str): the filename to encode

    Returns
        int: the length of the encoded filename
    """
    return len(s.encode('utf-8'))
