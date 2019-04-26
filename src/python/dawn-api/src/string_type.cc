#ifndef __TYPES_H__
#define __TYPES_H__

#include <Python.h>

typedef struct {
  PyObject_HEAD
  void * p;
} ZcString;

static PyObject *
ZcString_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  auto self = (ZcString *)type->tp_alloc(type, 0);
  assert(self);
  return (PyObject*)self;
}
/** 
 * tp_dealloc: Called when reference count is 0
 * 
 * @param self 
 */
static void
ZcString_dealloc(ZcString *self)
{
  assert(self);
  Py_TYPE(self)->tp_free((PyObject*)self);
}



static int
ZcString_init(ZcString *self, PyObject *args, PyObject *kwds)
{
  // static const char *kwlist[] = {"param",
  //                                NULL};
  
  // PyObject * object = nullptr;
  // PyObject * kid_list = nullptr;
  // unsigned int K = 0;
  // int collect_origins = 0;
  // int canonical = 1; /* by default convert to canonical form */
  
  // if (! PyArg_ParseTupleAndKeywords(args,
  //                                   kwds,
  //                                   "|OOIpp",
  //                                   const_cast<char**>(kwlist),
  //                                   &refdata_object,
  //                                   &kid_list,
  //                                   &K,
  //                                   &collect_origins,
  //                                   &canonical)) {
  //   PyErr_SetString(PyExc_RuntimeError, "Kmer_map ctor unable to parse args");
  //   return -1;
  // }
  return 0;
}


PyTypeObject ZcStringType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "dawn.ZcString",           /* tp_name */
  sizeof(ZcString)   ,      /* tp_basicsize */
  0,                       /* tp_itemsize */
  (destructor) ZcString_dealloc,      /* tp_dealloc */
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
  "ZcString",              /* tp_doc */
  0,                       /* tp_traverse */
  0,                       /* tp_clear */
  0,                       /* tp_richcompare */
  0,                       /* tp_weaklistoffset */
  0,                       /* tp_iter */
  0,                       /* tp_iternext */
  0, //ZcString_methods,         /* tp_methods */
  0, //ZcString_members,         /* tp_members */
  0,                       /* tp_getset */
  0,                       /* tp_base */
  0,                       /* tp_dict */
  0,                       /* tp_descr_get */
  0,                       /* tp_descr_set */
  0,                       /* tp_dictoffset */
  (initproc)ZcString_init,  /* tp_init */
  0,            /* tp_alloc */
  ZcString_new,             /* tp_new */
  0, /* tp_free */
};



#endif
