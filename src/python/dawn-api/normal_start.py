import dawn
import sys

pool_name = 'foo'
session = dawn.Session(ip="10.0.0.22")
pool = session.create_pool(pool_name,int(1e9),100)

pool.put('key0','hello world!')
pool.put('key1','goodbye world')

x = pool.get('key0')


print('>>' + x + '<<')
print(pool.count())

pool.close()
session.delete_pool(pool_name)
