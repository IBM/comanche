#ifndef __TYPES_H__
#define __TYPES_H__

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL DAWN_ARRAY_API

#include <common/logging.h>
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include "pool_type.h"

extern PyTypeObject PoolType;

static PyObject * pool_close(Pool* self);
static PyObject * pool_count(Pool* self);
static PyObject * pool_put(Pool* self, PyObject *args, PyObject *kwds);
static PyObject * pool_get(Pool* self, PyObject *args, PyObject *kwds);
static PyObject * pool_put_direct(Pool* self, PyObject *args, PyObject *kwds);
static PyObject * pool_get_direct(Pool* self, PyObject *args, PyObject *kwds);
static PyObject * pool_get_size(Pool* self, PyObject *args, PyObject *kwds);
static PyObject * pool_erase(Pool* self, PyObject *args, PyObject *kwds);
static PyObject * pool_configure(Pool* self, PyObject *args, PyObject *kwds);
static PyObject * pool_find_key(Pool* self, PyObject *args, PyObject *kwds);

static PyObject *
Pool_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  auto self = (Pool *)type->tp_alloc(type, 0);
  assert(self);
  return (PyObject*)self;
}

Pool * Pool_new()
{
  return (Pool *) PyType_GenericAlloc(&PoolType,1);
}

/** 
 * tp_dealloc: Called when reference count is 0
 * 
 * @param self 
 */
static void
Pool_dealloc(Pool *self)
{
  assert(self);
  Py_TYPE(self)->tp_free((PyObject*)self);
  PLOG("Pool: dealloc");
}

static PyMemberDef Pool_members[] = {
  //  {"port", T_ULONG, offsetof(Pool, _port), READONLY, "Port"},
  {NULL}
};

PyDoc_STRVAR(put_doc,"Pool.put(key,value) -> Write key-value pair to pool.");
PyDoc_STRVAR(put_direct_doc,"Pool.put_direct(key,value) -> Write bytearray value to pool using zero-copy.");
PyDoc_STRVAR(get_doc,"Pool.get(key) -> Read value from pool.");
PyDoc_STRVAR(get_size_doc,"Pool.get_size(key) -> Get size of a value.");
PyDoc_STRVAR(get_direct_doc,"Pool.get_direct(key) -> Read bytearray value from pool using zero-copy.");
PyDoc_STRVAR(close_doc,"Pool.close() -> Forces pool closure. Otherwise close happens on deletion.");
PyDoc_STRVAR(count_doc,"Pool.count() -> Get number of objects in the pool.");
PyDoc_STRVAR(erase_doc,"Pool.erase(key) -> Erase object from the pool.");
PyDoc_STRVAR(configure_doc,"Pool.configure(jsoncmd) -> Configure pool.");
PyDoc_STRVAR(find_key_doc,"Pool.find(expr, [limit]) -> Find keys using expression.");

static PyMethodDef Pool_methods[] = {
  {"close",(PyCFunction) pool_close, METH_NOARGS, close_doc},
  {"count",(PyCFunction) pool_count, METH_NOARGS, count_doc},
  {"put",(PyCFunction) pool_put, METH_VARARGS | METH_KEYWORDS, put_doc},
  {"put_direct",(PyCFunction) pool_put_direct, METH_VARARGS | METH_KEYWORDS, put_direct_doc},
  {"get",(PyCFunction) pool_get, METH_VARARGS | METH_KEYWORDS, get_doc},
  {"get_direct",(PyCFunction) pool_get_direct, METH_VARARGS | METH_KEYWORDS, get_direct_doc},
  {"get_size",(PyCFunction) pool_get_size, METH_VARARGS | METH_KEYWORDS, get_size_doc},
  {"erase",(PyCFunction) pool_erase, METH_VARARGS | METH_KEYWORDS, erase_doc},
  {"configure",(PyCFunction) pool_configure, METH_VARARGS | METH_KEYWORDS, configure_doc},
  {"find_key",(PyCFunction) pool_find_key, METH_VARARGS | METH_KEYWORDS, find_key_doc},
  {NULL}
};



PyTypeObject PoolType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "dawn.Pool",           /* tp_name */
  sizeof(Pool)   ,      /* tp_basicsize */
  0,                       /* tp_itemsize */
  (destructor) Pool_dealloc,      /* tp_dealloc */
  0,                       /* tp_print */
  0,                       /* tp_getattr */
  0,                       /* tp_setattr */
  0,                       /* tp_reserved */
  0,                       /* tp_repr */
  0,                       /* tp_as_number */
  0,                       /* tp_as_sequence */
  0,                       /* tp_as_mapping */
  0,                       /* tp_hash */
  0,                       /* tp_call */
  0,                       /* tp_str */
  0,                       /* tp_getattro */
  0,                       /* tp_setattro */
  0,                       /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  "Pool",              /* tp_doc */
  0,                       /* tp_traverse */
  0,                       /* tp_clear */
  0,                       /* tp_richcompare */
  0,                       /* tp_weaklistoffset */
  0,                       /* tp_iter */
  0,                       /* tp_iternext */
  Pool_methods,         /* tp_methods */
  Pool_members,         /* tp_members */
  0,                       /* tp_getset */
  0,                       /* tp_base */
  0,                       /* tp_dict */
  0,                       /* tp_descr_get */
  0,                       /* tp_descr_set */
  0,                       /* tp_dictoffset */
  0, //(initproc)Pool_init,  /* tp_init */
  0,            /* tp_alloc */
  Pool_new,             /* tp_new */
  0, /* tp_free */
};

  

static PyObject * pool_put(Pool* self, PyObject *args, PyObject *kwds)
{
  static const char *kwlist[] = {"key",
                                 "value",
                                 "no_stomp",
                                 NULL};

  const char * key = nullptr;
  PyObject * value = nullptr;
  int do_not_stomp = 0;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "sO|p",
                                    const_cast<char**>(kwlist),
                                    &key,
                                    &value,
                                    &do_not_stomp)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  assert(self->_kvstore);
  assert(self->_pool != IKVStore::POOL_ERROR);

  if(self->_pool == 0) {
    PyErr_SetString(PyExc_RuntimeError,"already closed");
    return NULL;
  }

  void * p = nullptr;
  size_t p_len = 0;
  if(PyByteArray_Check(value)) {
    p = PyByteArray_AsString(value);
    p_len = PyByteArray_Size(value);
  }
  else if(PyUnicode_Check(value)) {
    p = PyUnicode_DATA(value);
    p_len = PyUnicode_GET_DATA_SIZE(value);
  }

  unsigned int flags = 0;
  auto hr = self->_dawn->put(self->_pool,
                             key,
                             p,
                             p_len,
                             flags);
                                    
  if(hr != S_OK) {
    std::stringstream ss;
    ss << "pool.put failed [status:" << hr << "]";
    PyErr_SetString(PyExc_RuntimeError,ss.str().c_str());
    return NULL;
  }
                                    
  Py_INCREF(self);
  return (PyObject *) self;
}


static PyObject * pool_put_direct(Pool* self, PyObject *args, PyObject *kwds)
{
  static const char *kwlist[] = {"key",
                                 "value",
                                 NULL};

  const char * key = nullptr;
  PyObject * value = nullptr;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "sO|",
                                    const_cast<char**>(kwlist),
                                    &key,
                                    &value)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  void * p = nullptr;
  size_t p_len = 0;
  
  if(PyByteArray_Check(value)) {
    p = PyByteArray_AsString(value);
    p_len = PyByteArray_Size(value);
  }
  else {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  unsigned int flags = 0;
  status_t hr;
  Component::IKVStore::memory_handle_t handle = self->_dawn->register_direct_memory(p, p_len);

  if(handle == nullptr) {
    PyErr_SetString(PyExc_RuntimeError,"RDMA memory registration failed");
    return NULL;
  }
  
  hr = self->_dawn->put_direct(self->_pool,
                               key,
                               p,
                               p_len,
                               handle,
                               flags);
                                    
  if(hr != S_OK) {
    std::stringstream ss;
    ss << "pool.put_direct failed [status:" << hr << "]";
    PyErr_SetString(PyExc_RuntimeError,ss.str().c_str());
    return NULL;
  }

  /* unregister memory */
  self->_dawn->unregister_direct_memory(handle);
  
  Py_INCREF(self);
  return (PyObject *) self;
}


static PyObject * pool_get(Pool* self, PyObject *args, PyObject *kwds)
{
  static const char *kwlist[] = {"key",
                                 NULL};

  const char * key = nullptr;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "s",
                                    const_cast<char**>(kwlist),
                                    &key)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  assert(self->_kvstore);
  assert(self->_pool != IKVStore::POOL_ERROR);

  if(self->_pool == 0) {
    PyErr_SetString(PyExc_RuntimeError,"already closed");
    return NULL;
  }

  void * out_p = nullptr;
  size_t out_p_len = 0;
  auto hr = self->_dawn->get(self->_pool,
                             key,
                             out_p,
                             out_p_len);

  if(hr == Component::IKVStore::E_KEY_NOT_FOUND) {
    Py_RETURN_NONE;
  }
  else if(hr != S_OK || out_p == nullptr) {
    std::stringstream ss;
    ss << "pool.get failed [status:" << hr << "]";
    PyErr_SetString(PyExc_RuntimeError,ss.str().c_str());
    return NULL;
  }

  /* copy value string */
  auto result = PyUnicode_FromString(static_cast<const char*>(out_p));
  self->_dawn->free_memory(out_p);

  return result;
}


static PyObject * pool_get_direct(Pool* self, PyObject *args, PyObject *kwds)
{
  static const char *kwlist[] = {"key",
                                 NULL};

  const char * key = nullptr;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "s",
                                    const_cast<char**>(kwlist),
                                    &key)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  assert(self->_pool);
    
  std::vector<uint64_t> v;
  std::string k(key);
  
  auto hr = self->_dawn->get_attribute(self->_pool,
                                       Component::IKVStore::Attribute::VALUE_LEN,
                                       v,
                                       &k);

  if(hr != S_OK || v.size() != 1) {
    std::stringstream ss;
    ss << "pool.get_direct failed [status:" << hr << "]";
    PyErr_SetString(PyExc_RuntimeError,ss.str().c_str());
    return NULL;
  }
  
  /* now we have the buffer size, we can allocate accordingly */
  size_t p_len = v[0];
  PyObject * result = PyBytes_FromStringAndSize(NULL, p_len);

  void * p = (void *) PyBytes_AsString(result);

  /* register memory */
  Component::IKVStore::memory_handle_t handle = self->_dawn->register_direct_memory(p, p_len);

  if(handle == nullptr) {
    PyErr_SetString(PyExc_RuntimeError,"RDMA memory registration failed");
    return NULL;
  }
  
  /* now perform get_direct */
  hr = self->_dawn->get_direct(self->_pool, k, p, p_len, handle);
  if(hr != S_OK) {
    std::stringstream ss;
    ss << "pool.get_direct failed [status:" << hr << "]";
    PyErr_SetString(PyExc_RuntimeError,ss.str().c_str());
    return NULL;
  }

  self->_dawn->unregister_direct_memory(handle);
  
  return result;
}



static PyObject * pool_get_size(Pool* self, PyObject *args, PyObject *kwds)
{
  static const char *kwlist[] = {"key",
                                 NULL};

  const char * key = nullptr;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "s",
                                    const_cast<char**>(kwlist),
                                    &key)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  assert(self->_pool);
    
  std::vector<uint64_t> v;
  std::string k(key);
  
  auto hr = self->_dawn->get_attribute(self->_pool,
                                       Component::IKVStore::Attribute::VALUE_LEN,
                                       v,
                                       &k);

  if(hr != S_OK || v.size() != 1) {
    std::stringstream ss;
    ss << "pool.get_size failed [status:" << hr << "]";
    PyErr_SetString(PyExc_RuntimeError,ss.str().c_str());
    return NULL;
  }
  
  return PyLong_FromSize_t(v[0]);
}


static PyObject * pool_close(Pool* self)
{
  
  assert(self->_dawn);
  assert(self->_pool != IKVStore::POOL_ERROR);

  if(self->_pool == 0) {
    PyErr_SetString(PyExc_RuntimeError,"Dawn.Pool.close failed. Already closed.");
    return NULL;
  }

  status_t hr = self->_dawn->close_pool(self->_pool);
  self->_pool = 0;

  if(hr != S_OK) {
    std::stringstream ss;
    ss << "pool.close failed [status:" << hr << "]";
    PyErr_SetString(PyExc_RuntimeError,ss.str().c_str());
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyObject * pool_count(Pool* self)
{
  assert(self->_dawn);
  assert(self->_pool != IKVStore::POOL_ERROR);

  if(self->_pool == 0) {
    PyErr_SetString(PyExc_RuntimeError,"Dawn.Pool.count failed. Already closed.");
    return NULL;
  }

  return PyLong_FromSize_t(self->_dawn->count(self->_pool));
}

static PyObject * pool_erase(Pool* self, PyObject *args, PyObject *kwds)
{
  static const char *kwlist[] = {"key",
                                 NULL};

  const char * key = nullptr;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "s",
                                    const_cast<char**>(kwlist),
                                    &key)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  assert(self->_pool);
    
  std::string k(key);
  
  auto hr = self->_dawn->erase(self->_pool, k);

  if(hr != S_OK) {
    std::stringstream ss;
    ss << "pool.erase [status:" << hr << "]";
    PyErr_SetString(PyExc_RuntimeError,ss.str().c_str());    
    return NULL;
  }

  Py_RETURN_TRUE;
}


static PyObject * pool_configure(Pool* self, PyObject *args, PyObject *kwds)
{
  static const char *kwlist[] = {"command",
                                 NULL};

  const char * command = nullptr;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "s",
                                    const_cast<char**>(kwlist),
                                    &command)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  assert(self->_pool);
    
  const std::string cmd(command);
  
  auto hr = self->_dawn->configure_pool(self->_pool, cmd);

  if(hr != S_OK) {
    std::stringstream ss;
    ss << "pool.configure [status:" << hr << "]";
    PyErr_SetString(PyExc_RuntimeError,ss.str().c_str());    
    return NULL;
  }

  Py_RETURN_TRUE;
}



static PyObject * pool_find_key(Pool* self, PyObject *args, PyObject *kwds)
{
  static const char *kwlist[] = {"expr",
                                 "offset",
                                 NULL};

  const char * expr_param = nullptr;
  int offset_param = 0;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "s|i",
                                    const_cast<char**>(kwlist),
                                    &expr_param,
                                    &offset_param)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  assert(self->_pool);
    
  const std::string expr(expr_param);

  std::string out_key;
  offset_t out_pos = 0;
  auto hr = self->_dawn->find(self->_pool,
                              expr,                              
                              offset_param,
                              out_pos,
                              out_key);

  if(hr == S_OK) {
    auto tuple = PyTuple_New(2);
    PyTuple_SetItem(tuple, 0, PyUnicode_FromString(out_key.c_str()));
    PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLong(out_pos));
    return tuple;
  }
  else if(hr == E_FAIL) {
    auto tuple = PyTuple_New(2);
    PyTuple_SetItem(tuple, 0, Py_None);
    PyTuple_SetItem(tuple, 1, Py_None);
    return tuple;
  }
  else {
    std::stringstream ss;
    ss << "pool.find [status:" << hr << "]";
    PyErr_SetString(PyExc_RuntimeError,ss.str().c_str());    
    return NULL;
  }

  return NULL;
}

#endif
