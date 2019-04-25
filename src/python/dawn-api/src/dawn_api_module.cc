
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL DAWN_ARRAY_API
#include <numpy/arrayobject.h>

//now, everything is setup, just include the numpy-arrays:
#include <numpy/arrayobject.h>
#include <Python.h>
#include <structmember.h>
#include <objimpl.h>
#include <pythread.h>
//#include <numpy/npy_math.h>

#include <api/components.h>
#include <api/kvstore_itf.h>

#if defined(__linux__)
#include <execinfo.h>
#endif

#include <list>
#include <common/logging.h>

static PyObject * open_connection(PyObject * self,
                                  PyObject * args,
                                  PyObject * kwargs);


PyDoc_STRVAR(open_connection_doc,"open_connection(ipaddr:port) -> open connection to Dawn server");


// forward declaration of custom types
//
extern PyTypeObject ZcStringType;
extern PyTypeObject SessionType;
extern PyTypeObject PoolType;

static PyMethodDef dawn_methods[] = {
  {"open_connection", (PyCFunction) open_connection, METH_VARARGS | METH_KEYWORDS, open_connection_doc},
  // {"compare_set_sorted_hashes",
  //  (PyCFunction) bio_compare_set_sorted_hashes, METH_VARARGS | METH_KEYWORDS, bio_compare_set_sorted_hashes_doc},
  // {"parallel_compare_set_sorted_hashes",
  //  (PyCFunction) bio_parallel_compare_set_sorted_hashes, METH_VARARGS | METH_KEYWORDS, bio_parallel_compare_set_sorted_hashes_doc},  
  {NULL, NULL, 0, NULL}        /* Sentinel */
};

static PyModuleDef dawn_module = {
    PyModuleDef_HEAD_INIT,
    "dawn",
    "Dawn client API extension module",
    -1,
    dawn_methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_dawn(void)
{  
  PyObject *m;

  PLOG("Init dawn Python extension");

  import_array();
  
  ZcStringType.tp_base = 0; // no inheritance
  if(PyType_Ready(&ZcStringType) < 0) {
    assert(0);
    return NULL;
  }

  SessionType.tp_base = 0; // no inheritance
  if(PyType_Ready(&SessionType) < 0) {
    assert(0);
    return NULL;
  }

  PoolType.tp_base = 0; // no inheritance
  if(PyType_Ready(&PoolType) < 0) {
    assert(0);
    return NULL;
  }


  /* register module */
#if PY_MAJOR_VERSION >= 3
  m = PyModule_Create(&dawn_module);
#else
#error "Extension for Python 3 only."
#endif

  if (m == NULL)
    return NULL;

  /* add types */
  int rc;

  Py_INCREF(&ZcStringType);
  rc = PyModule_AddObject(m, "ZcString", (PyObject *) &ZcStringType);
  if(rc) return NULL;

  Py_INCREF(&SessionType);
  rc = PyModule_AddObject(m, "Session", (PyObject *) &SessionType);
  if(rc) return NULL;
  
  Py_INCREF(&PoolType);
  rc = PyModule_AddObject(m, "Pool", (PyObject *) &PoolType);
  if(rc) return NULL;


  return m;
}

static PyObject * open_connection(PyObject * self,
                                  PyObject * args,
                                  PyObject * kwargs)
{
  Py_RETURN_NONE;
}

// static PyObject * bio_parallel_compare_set_sorted_hashes(PyObject * self,
//                                                          PyObject * args,
//                                                          PyObject * kwargs)
// {
//   static const char *kwlist[] = {"filelist", NULL};
//   PyObject * filelist_obj = nullptr;
  
//   assert(kwargs);
//   if(! PyArg_ParseTupleAndKeywords(args,
//                                    kwargs,
//                                    "O|",
//                                    const_cast<char**>(kwlist),
//                                    &filelist_obj)) {
//     PyErr_SetString(PyExc_AttributeError, "bio_parallel_compare_set_sorted_hashes: unable to parse args");
//     return NULL;
//   }

//   if(!PyList_Check(filelist_obj)) {
//     PyErr_SetString(PyExc_AttributeError, "bio_parallel_compare_set_sorted_hashes: invalid type(s)");
//     return NULL;
//   }

//   std::list<std::string> filenames;
  
//   auto size = PyList_Size(filelist_obj);
//   for(ssize_t i=0;i<size;i++) {
//     auto item = PyList_GetItem(filelist_obj, i);
//     assert(item);
//     if(!PyUnicode_Check(item)) {
//       PyErr_SetString(PyExc_AttributeError, "bio_parallel_compare_set_sorted_hashes: invalid filename type");
//       return NULL;
//     }
//     Py_ssize_t size;
//     char *ptr = PyUnicode_AsUTF8AndSize(item, &size);
//     filenames.push_back(std::string(ptr));
//   }

//   double * out_jaccards = nullptr;
//   size_t out_jaccards_length = 0;

//   size_t * out_intersections = nullptr;
//   size_t out_intersections_length = 0;

//   if(Genetics::parallel_compare_set_sorted_hashes(filenames,
//                                                   out_jaccards,
//                                                   out_jaccards_length,
//                                                   out_intersections,
//                                                   out_intersections_length) != S_OK) {
//     PyErr_SetString(PyExc_AttributeError,
//                     "bio_parallel_compare_set_sorted_hashes: parallel_compare_set_sorted_hashes failed.");
//     return NULL;
//   }

//   assert(out_jaccards_length == out_intersections_length);
//   PLOG("jaccards=%p %lu  intersections=%p %lu",
//        out_jaccards,
//        out_jaccards_length,
//        out_intersections,
//        out_intersections_length);
  
//   auto dim = filenames.size();

//   int nd = 2;
//   npy_intp dims[2] = { (signed long) dim, (signed long) dim };

//   /* allocate ndarray memory and do a copy*/
//   auto out_jaccard_matrix = (PyArrayObject *) PyArray_SimpleNew(nd, dims, NPY_FLOAT64);
//   memcpy(PyArray_DATA(out_jaccard_matrix), out_jaccards, out_jaccards_length * sizeof(double));

//   auto out_isect_matrix = (PyArrayObject *) PyArray_SimpleNew(nd, dims, NPY_UINT64);
//   memcpy(PyArray_DATA(out_isect_matrix), out_intersections, out_intersections_length * sizeof(uint64_t));

//   PyObject* result = PyDict_New();
//   if(PyDict_SetItemString(result, "jaccard_matrix",  (PyObject*) out_jaccard_matrix) != 0) {
//     PyErr_SetString(PyExc_RuntimeError, "bio_compare_set_sorted_hashes: unable construct result");
//     Py_DECREF(result);
//     Py_DECREF(out_jaccard_matrix);
//     Py_DECREF(out_isect_matrix);
//     return NULL;
//   }

//   if(PyDict_SetItemString(result, "intersection_matrix",  (PyObject*) out_isect_matrix) != 0) {
//     PyErr_SetString(PyExc_RuntimeError, "bio_compare_set_sorted_hashes: unable construct result");
//     Py_DECREF(result);
//     Py_DECREF(out_jaccard_matrix);
//     Py_DECREF(out_isect_matrix);
//     return NULL;
//   }

//   /* clean up memory */
//   free(out_jaccards);
//   free(out_intersections);
  
//   return (PyObject *) result;
// }
