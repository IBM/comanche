import dawn
import sys
import numpy as np

pool_name = 'foo'
#session = dawn.Session(ip="10.0.0.92",port=11900)
session = dawn.Session(ip="10.0.0.22")
pool = session.create_pool(pool_name,int(2e9),100)



pool.put('key0','hello world!')
pool.put('key1','goodbye world')

pool.configure("{ \"index\" : \"volatile_rbtree\" }")

pool.put('key2','bad data')

x = pool.get('key0')

print('>>' + x + '<<')
print(pool.count())

pool.erase('key2')

k = pool.find("regex:k.*")
print(k)


#arr = bytearray('byte array', 'utf-8')
arr = bytearray(int(1e6))
arr[0] = 3;
arr[1] = 2;
arr[2] = 1;
pool.put_direct('array0', arr)

y = pool.get_direct('array0')
#y = pool.get_direct('key1')
#y = pool.get('array0')
print('First part...')
print(y[0:20])

print('Size enquiry:%d' % pool.get_size('array0'))
      
#pool.close()
#session.delete_pool(pool_name)
