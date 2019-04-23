
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <structmember.h>
#include <objimpl.h>
#include <pythread.h>
#include <numpy/arrayobject.h>
//#include <numpy/npy_math.h>

#if defined(__linux__)
#include <execinfo.h>
#endif

#include <list>
#include <common/logging.h>

// static PyObject * bio_relative_complement(ReferenceData * self,
//                                           PyObject * args,
//                                           PyObject * kwargs);


// PyDoc_STRVAR(bio_relative_complement_doc,"relative_complement(a,b) -> Returns relative complement of 'a' and 'b'.");


// forward declaration of custom types
//
extern PyTypeObject ReferenceDataType;
extern PyTypeObject KmerMapType;
extern PyTypeObject KmerMapIteratorType;
extern PyTypeObject KmerLshType;
extern PyTypeObject KmerLshIteratorType;
extern PyTypeObject DebruijnGraphType;
extern PyTypeObject DebruijnGraphIteratorType;

static PyMethodDef dawn_methods[] = {
  // {"load_set_sorted_hashes",
  //  (PyCFunction) bio_load_set_sorted_hashes, METH_VARARGS | METH_KEYWORDS, bio_load_set_sorted_hashes_doc},
  // {"compare_set_sorted_hashes",
  //  (PyCFunction) bio_compare_set_sorted_hashes, METH_VARARGS | METH_KEYWORDS, bio_compare_set_sorted_hashes_doc},
  // {"parallel_compare_set_sorted_hashes",
  //  (PyCFunction) bio_parallel_compare_set_sorted_hashes, METH_VARARGS | METH_KEYWORDS, bio_parallel_compare_set_sorted_hashes_doc},  
  {NULL, NULL, 0, NULL}        /* Sentinel */
};

static PyModuleDef dawn_module = {
    PyModuleDef_HEAD_INIT,
    "dawn",
    "Dawn client API extension module.",
    -1,
    dawn_methods,
    NULL, NULL, NULL, NULL
};



PyMODINIT_FUNC
PyInit_dawn(void)
{  
  PyObject *m;

  import_array(); /* using NumPy C-API */
  
  // ReferenceDataType.tp_base = 0; // no inheritance
  // if(PyType_Ready(&ReferenceDataType) < 0) {
  //   assert(0);
  //   return NULL;
  // }

  /* register module */
#if PY_MAJOR_VERSION >= 3
  m = PyModule_Create(&dawn_module);
#else
#error "Extension for Python 3 only."
#endif

  if (m == NULL)
    return NULL;

  /* add types */
  // int rc;

  // Py_INCREF(&ReferenceDataType);
  // rc = PyModule_AddObject(m, "ReferenceData", (PyObject *) &ReferenceDataType);
  // if(rc) {
  //   assert(rc==0);
  //   return NULL;
  // }

  return m;
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
