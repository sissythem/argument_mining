def name_exceeds_bytes(name):
    return utf8len(name) >= 255


def utf8len(s):
    return len(s.encode('utf-8'))
