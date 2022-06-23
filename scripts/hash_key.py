import hashlib
import sys


def hash_string(key):
    m = hashlib.sha256()
    m.update(key.encode())
    hash = m.hexdigest()
    return hash


if __name__ == "__main__":
    try:
        outfile = sys.argv[1]
    except IndexError:
        outfile = None
    print("Storing results to", outfile)
    key = input("key: ")
    hash = hash_string(key)
    print(hash)
    if outfile is not None:
        with open(outfile, "a") as f:
            f.write(str(hash) + "\n")
