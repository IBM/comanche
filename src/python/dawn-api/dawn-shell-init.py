import dawn

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

