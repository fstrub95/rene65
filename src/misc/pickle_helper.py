import pickle
import gzip
import sys

def pickle_dump(data, file_path, gz=False):
    open_fct = open
    if gz:
        open_fct = gzip.open
        file_path += ".gz"

    with open_fct(file_path, "wb") as f:
        pickle.dump(data, f)

def pickle_loader(file_path, gz=False):
    open_fct = open
    if gz:
        open_fct = gzip.open

    with open_fct(file_path, "rb") as f:
        if sys.version_info > (3, 0):  # Workaround to load pickle data python2 -> python3
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            return u.load()
        else:
            return pickle.load(f)