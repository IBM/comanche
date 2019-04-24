#ifndef __TYPES_H__
#define __TYPES_H__

#define DEFAULT_DEVICE "mlx5_0"
#define DEFAULT_PORT 11911

#include <sstream>
#include <common/logging.h>
#include <api/dawn_itf.h>
#include <api/kvstore_itf.h>
#include <Python.h>
#include <structmember.h>

#include "config.h"
#include "pool_type.h"

using namespace Component;

typedef struct {
  PyObject_HEAD
  Component::IDawn * _dawn;
  int                _port;
} Session;

static PyObject * open_pool(Session* self, PyObject *args, PyObject *kwds);
static PyObject * create_pool(Session* self, PyObject *args, PyObject *kwds);
static PyObject * delete_pool(Session* self, PyObject *args, PyObject *kwds);
  
static PyObject *
Session_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  auto self = (Session *)type->tp_alloc(type, 0);
  assert(self);
  return (PyObject*)self;
}



/** 
 * tp_dealloc: Called when reference count is 0
 * 
 * @param self 
 */
static void
Session_dealloc(Session *self)
{
  assert(_dawn);
  self->_dawn->release_ref();
  
  assert(self);
  Py_TYPE(self)->tp_free((PyObject*)self);
  PLOG("Session: dealloc");
}


static int Session_init(Session *self, PyObject *args, PyObject *kwds)
{
  PLOG("Session init");
  static const char *kwlist[] = {"ip",
                                 "port",
                                 "device",
                                 "debug",
                                 NULL};

  const char * p_ip = nullptr;
  const char * p_device = nullptr;
  int port = DEFAULT_PORT;
  int debug_level = 0;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "s|is",
                                    const_cast<char**>(kwlist),
                                    &p_ip,
                                    &port,
                                    &p_device,
                                    &debug_level)) {
    PyErr_SetString(PyExc_RuntimeError, "bad arguments");
    return -1;
  }

  std::string device = DEFAULT_DEVICE;
  if(p_device)
    device = p_device;

  std::stringstream addr;
  addr << p_ip << ":" << port;


  using namespace Component;
  
  /* create object instance through factory */
  std::string path = COMANCHE_LIB_PATH;
  path += "/libcomanche-dawn-client.so";
  
  IBase *comp = load_component(path.c_str(), dawn_client_factory);

  if(comp == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Dawn.Session failed to load libcomanche-dawn-client.so");
    return -1;
  }
  
  auto fact = (IDawn_factory *) comp->query_interface(IDawn_factory::iid());
  if(fact == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Dawn.Session failed to get IDawn_factory");
    return -1;
  }

  char * p_env_user_name = getenv("USER");
  std::string user_name;
  if(p_env_user_name) user_name = p_env_user_name;
  else user_name = "unknown";
  
  self->_dawn = fact->dawn_create(debug_level,
                                  user_name,
                                  addr.str(),
                                  device);
  self->_port = port;
  
  fact->release_ref();

  PLOG("session: (%s)(%s) %p", addr.str().c_str(), device.c_str(), self->_dawn);
  return 0;
}

static PyMemberDef Session_members[] = {
  {"port", T_ULONG, offsetof(Session, _port), READONLY, "Port"},
  {NULL}
};

PyDoc_STRVAR(open_pool_doc,"Session.open_pool(name,[readonly=True]) -> Open pool.");
PyDoc_STRVAR(create_pool_doc,"Session.create_pool(name,pool_size,objcount) -> Create pool.");
PyDoc_STRVAR(delete_pool_doc,"Session.delete_pool(name) -> Delete pool.");

static PyMethodDef Session_methods[] = {
  {"open_pool",  (PyCFunction) open_pool, METH_VARARGS | METH_KEYWORDS, open_pool_doc},
  {"create_pool",  (PyCFunction) create_pool, METH_VARARGS | METH_KEYWORDS, create_pool_doc},
  {"delete_pool",  (PyCFunction) delete_pool, METH_VARARGS | METH_KEYWORDS, delete_pool_doc},
  {NULL}
};



PyTypeObject SessionType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "dawn.Session",           /* tp_name */
  sizeof(Session)   ,      /* tp_basicsize */
  0,                       /* tp_itemsize */
  (destructor) Session_dealloc,      /* tp_dealloc */
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
  "Session",              /* tp_doc */
  0,                       /* tp_traverse */
  0,                       /* tp_clear */
  0,                       /* tp_richcompare */
  0,                       /* tp_weaklistoffset */
  0,                       /* tp_iter */
  0,                       /* tp_iternext */
  Session_methods,         /* tp_methods */
  Session_members,         /* tp_members */
  0,                       /* tp_getset */
  0,                       /* tp_base */
  0,                       /* tp_dict */
  0,                       /* tp_descr_get */
  0,                       /* tp_descr_set */
  0,                       /* tp_dictoffset */
  (initproc)Session_init,  /* tp_init */
  0,            /* tp_alloc */
  Session_new,             /* tp_new */
  0, /* tp_free */
};


static PyObject * open_pool(Session* self, PyObject *args, PyObject *kwds)
{
  static const char *kwlist[] = {"name",
                                 "readonly",
                                 NULL};

  const char * pool_name = nullptr;
  int read_only = 0;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "s|p",
                                    const_cast<char**>(kwlist),
                                    &pool_name,
                                    &read_only)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  assert(self->_dawn);
  Pool * p = Pool_new();
  uint32_t flags = 0;
  if(read_only) flags |= Component::IKVStore::FLAGS_READ_ONLY;

  self->_dawn->add_ref();

  assert(self->_dawn);
  p->_dawn = self->_dawn;
  p->_pool = self->_dawn->open_pool(pool_name, flags);  

  if(p->_pool == IKVStore::POOL_ERROR) {
    PyErr_SetString(PyExc_RuntimeError,"Dawn.Session.open_pool failed");
    return NULL;
  }

  return (PyObject *) p;
}

static PyObject * create_pool(Session* self, PyObject *args, PyObject *kwds)
{
  static const char *kwlist[] = {"name",
                                 "size",
                                 "objcount",
                                 "create_only",
                                 NULL};

  const char * pool_name = nullptr;
  unsigned long size = 0;
  unsigned long objcount = 0;
  int create_only = 0;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "sk|kp",
                                    const_cast<char**>(kwlist),
                                    &pool_name,
                                    &size,
                                    &objcount,
                                    &create_only)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  assert(self->_dawn);
  Pool * p = Pool_new();
  
  uint32_t flags = 0;
  if(create_only) flags |= Component::IKVStore::FLAGS_CREATE_ONLY;

  assert(self->_dawn);
  p->_dawn = self->_dawn;
  p->_pool = self->_dawn->create_pool(pool_name,
                                         size,
                                         flags,
                                         objcount);

  if(p->_pool == IKVStore::POOL_ERROR) {
    PyErr_SetString(PyExc_RuntimeError,"Dawn.Session.create_pool failed");
    return NULL;
  }

  return (PyObject *) p;
}


static PyObject * delete_pool(Session* self, PyObject *args, PyObject *kwds)
{
  static const char *kwlist[] = {"name",
                                 NULL};

  const char * pool_name = nullptr;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "s",
                                    const_cast<char**>(kwlist),
                                    &pool_name)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  assert(self->_dawn);

  status_t hr = self->_dawn->delete_pool(pool_name);

  if(hr != S_OK) {
    std::stringstream ss;
    ss << "Dawn.Session.create_pool failed [status:" << hr << "]";
    PyErr_SetString(PyExc_RuntimeError,ss.str().c_str());
    return NULL;
  }

  Py_RETURN_NONE;
}


#endif
