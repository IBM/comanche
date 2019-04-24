#ifndef __TYPES_H__
#define __TYPES_H__

#define DEFAULT_DEVICE "mlx5_0"
#define DEFAULT_PORT 11911

#include "config.h"
#include <common/logging.h>
#include <api/kvstore_itf.h>
#include <Python.h>
#include <structmember.h>


typedef struct {
  PyObject_HEAD
  Component::IKVStore * _kvstore;
  int                   _port;
} Session;

static PyObject * session_open_pool(Session* self, PyObject *args, PyObject *kwds);

  
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
    PyErr_SetString(PyExc_RuntimeError, "Dawn.Session constructor requires ip, port, device fields");
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
  
  auto fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  self->_kvstore = fact->create(debug_level,
                                "dwaddington",
                                addr.str(),
                                device);
  self->_port = port;
  
  fact->release_ref();

  PLOG("session: (%s)(%s) %p", addr.str().c_str(), device.c_str(), self->_kvstore);
  return 0;
}

static PyMemberDef Session_members[] = {
  {"port", T_ULONG, offsetof(Session, _port), READONLY, "Port"},
  {NULL}
};

PyDoc_STRVAR(open_pool_doc,"Session.open_pool(name) -> Open pool.");

static PyMethodDef Session_methods[] = {
  {"open_pool",  (PyCFunction) session_open_pool, METH_VARARGS | METH_KEYWORDS, open_pool_doc},
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


static PyObject * session_open_pool(Session* self, PyObject *args, PyObject *kwds)
{
  Py_RETURN_NONE;
}


#endif
