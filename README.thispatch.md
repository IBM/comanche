Notes for fixing:
## 2018-3
[bug fix]:

### issue #1 
- description: two asynchonous writes cannot be issues in the same time, see [here](73933e)
- reason: async_write will pass the **buffer** address directly to aiocb(which is used by adio_write and return; this **buffer** is not protected, and will be overwritten if you issues another async_write using the same **buffer** without wait_completion.

- fix(two approach):
        1. can have another buffer allocated in async_write, and data will be copied to this buffer and return, the callback function will free this buffer 
        2. let the callers of async_write create extra buffer and do the copy(of course we can also have a "buffered async_write")
    
    I use 2 first. reasons:
        1. I think it's better to follow the philosophy of aio_write: let user control the buffer.
        2. if i change block_posix, that means I have to change nmve_posix also.


