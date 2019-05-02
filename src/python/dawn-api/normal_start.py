import dawn
import sys
import numpy as np

def get_keys(pool, expr):
    result = []
    offset=0
    print(expr)
    (k,offset)=pool.find_key(expr, offset)
    while k != None:
        result.append(k)
        offset += 1
        (k, offset) = pool.find_key(expr, offset)
    return result

pool_name = 'foo'
#session = dawn.Session(ip="10.0.0.92",port=11900)
session = dawn.Session(ip="10.0.0.22")
pool = session.create_pool(pool_name,int(2e9),100)


pool.put('key9','k-nine')
pool.put('key0','hello world!')
pool.put('key1','goodbye world')

pool.configure("AddIndex::VolatileTree")

pool.put('tree0','oak')
pool.put('tree1','maple')
pool.put('key2','enjoy life')


print(get_keys(pool, "regex:.*"))
print('----')

print(get_keys(pool, "next:"))
print('----')

print(get_keys(pool, "regex:key[12]"))
print('----')

print(get_keys(pool, "prefix:tre"))
print('----')

#pool.configure("RemoveIndex::");

x = pool.get('key0')

print('>>' + x + '<<')
print(pool.count())

pool.erase('key2')

#arr = bytearray('byte array', 'utf-8')
arr = bytearray(int(1e3))
arr[0] = 3;
arr[1] = 2;
arr[2] = 1;
pool.put_direct('array0', arr)

y = pool.get_direct('array0')
#y = pool.get_direct('key1')
#y = pool.get('array0')
print('First part...')
print(y[0:20])

print('Len: %d' % pool.get_attribute('array0','length'))
print('Crc: %x' % pool.get_attribute('array0','crc32'))

print('Size enquiry:%d' % pool.get_size('array0'))


