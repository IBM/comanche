import dawn
import pickle

"""
Save arbitrary Python data to KV store
"""
def pickle_put(pool, keyname, item):
    pickled_item = pickle.dumps(item)
    pool.put_direct(keyname, bytearray(pickled_item))

"""
Load arbitrary Python data from KV store
"""
def pickle_get(pool, keyname):
    bytearray_item = pool.get_direct(keyname)
    return pickle.loads(bytes(bytearray_item))
