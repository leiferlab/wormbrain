//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "dsmm.hpp" //This is inside dsmm/ in the root of the repository

// Define functions below

static PyObject *_dsmmc_bare(PyObject *self, PyObject *args);

/////// Python-module-related functions and tables

// The module's method table
static PyMethodDef _dsmm_cMethods[] = {
    {"_dsmmc_bare", _dsmmc_bare, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

// The module definition function
static struct PyModuleDef _dsmm_c = {
    PyModuleDef_HEAD_INIT,
    "_dsmm_c",
    NULL, // Module documentation
    -1,
    _dsmm_cMethods
};

// The module initialization function
PyMODINIT_FUNC PyInit__dsmm_c(void) { 
        import_array(); //Numpy
        return PyModule_Create(&_dsmm_c);
};

static PyObject *_dsmmc_bare(PyObject *self, PyObject *args) {

    int M, N, D;
    double beta, lambda, neighbor_cutoff, alpha, gamma0, conv_epsilon, eq_tol;
    //bool releaseGIL;
    PyObject *X_o, *Y_o, *pwise_dist_o, *pwise_distYY_o, *Gamma_o, *CDE_term_o;
    PyObject *w_o, *F_t_o, *wF_t_o, *wF_t_sum_o, *p_o, *u_o, *Match_o;
    PyObject *hatP_o, *hatPI_diag_o, *hatPIG_o, *hatPX_o, *hatPIY_o;
    PyObject *G_o, *W_o, *GW_o, *sumPoverN_o, *expAlphaSumPoverN_o;
    
    if(!PyArg_ParseTuple(args, "OOiiidddddddOOOOOOOOOOOOOOOOOOOOO", 
        &X_o, &Y_o, &M, &N, &D, 
        &beta, &lambda, &neighbor_cutoff,
        &alpha, &gamma0,
        &conv_epsilon, &eq_tol,
        &pwise_dist_o, &pwise_distYY_o,
        &Gamma_o, &CDE_term_o,
        &w_o, &F_t_o, &wF_t_o, &wF_t_sum_o,
        &p_o, &u_o, &Match_o,
        &hatP_o, &hatPI_diag_o, &hatPIG_o, &hatPX_o, &hatPIY_o,
        &G_o, &W_o, &GW_o,
        &sumPoverN_o, &expAlphaSumPoverN_o)) return NULL;
    
    PyObject *X_a = PyArray_FROM_OTF(X_o, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject *Y_a = PyArray_FROM_OTF(Y_o, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject *pwise_dist_a = PyArray_FROM_OTF(pwise_dist_o, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject *pwise_distYY_a = PyArray_FROM_OTF(pwise_distYY_o, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject *Gamma_a = PyArray_FROM_OTF(Gamma_o, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject *CDE_term_a = PyArray_FROM_OTF(CDE_term_o, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject *w_a = PyArray_FROM_OTF(w_o, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject *F_t_a = PyArray_FROM_OTF(F_t_o, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject *wF_t_a = PyArray_FROM_OTF(wF_t_o, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject *wF_t_sum_a = PyArray_FROM_OTF(wF_t_sum_o, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject *p_a = PyArray_FROM_OTF(p_o, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject *u_a = PyArray_FROM_OTF(u_o, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject *Match_a = PyArray_FROM_OTF(Match_o, NPY_INT32, NPY_IN_ARRAY);
    PyObject *hatP_a = PyArray_FROM_OTF(hatP_o, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject *hatPI_diag_a = PyArray_FROM_OTF(hatPI_diag_o, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject *hatPIG_a = PyArray_FROM_OTF(hatPIG_o, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject *hatPX_a = PyArray_FROM_OTF(hatPX_o, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject *hatPIY_a = PyArray_FROM_OTF(hatPIY_o, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject *G_a = PyArray_FROM_OTF(G_o, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject *W_a = PyArray_FROM_OTF(W_o, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject *GW_a = PyArray_FROM_OTF(GW_o, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject *sumPoverN_a = PyArray_FROM_OTF(sumPoverN_o, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject *expAlphaSumPoverN_a = PyArray_FROM_OTF(expAlphaSumPoverN_o, NPY_FLOAT64, NPY_IN_ARRAY);
        
    // Check that the above conversion worked, otherwise decrease the reference
    // count and return NULL.                                 
    if (X_a == NULL || Y_a == NULL || pwise_dist_a == NULL || pwise_distYY_a == NULL
        || Gamma_a == NULL || CDE_term_a == NULL || w_a == NULL || F_t_a == NULL
        || wF_t_a == NULL || wF_t_sum_a == NULL || p_a == NULL || u_a == NULL
        || Match_a == NULL
        || hatP_a == NULL || hatPI_diag_a == NULL || hatPIG_a == NULL || hatPX_a == NULL
        || hatPIY_a == NULL || G_a == NULL || W_a == NULL || GW_a == NULL
        || sumPoverN_a == NULL || expAlphaSumPoverN_a == NULL
        ) {
        Py_XDECREF(X_a);
        Py_XDECREF(Y_a);
        Py_XDECREF(pwise_dist_a);
        Py_XDECREF(pwise_distYY_a);
        Py_XDECREF(Gamma_a);
        Py_XDECREF(CDE_term_a);
        Py_XDECREF(w_a);
        Py_XDECREF(F_t_a);
        Py_XDECREF(wF_t_a);
        Py_XDECREF(wF_t_sum_a);
        Py_XDECREF(p_a);
        Py_XDECREF(u_a);
        Py_XDECREF(Match_a);
        Py_XDECREF(hatP_a);
        Py_XDECREF(hatPI_diag_a);
        Py_XDECREF(hatPIG_a);
        Py_XDECREF(hatPX_a);
        Py_XDECREF(hatPIY_a);
        Py_XDECREF(G_a);
        Py_XDECREF(W_a);
        Py_XDECREF(GW_a);
        Py_XDECREF(sumPoverN_a);
        Py_XDECREF(expAlphaSumPoverN_a);
        return NULL;
    }
    
    // Get pointers to the data in the numpy arrays.
    double *X = (double*)PyArray_DATA(X_a);
    double *Y = (double*)PyArray_DATA(Y_a);
    double *pwise_dist = (double*)PyArray_DATA(pwise_dist_a);
    double *pwise_distYY = (double*)PyArray_DATA(pwise_distYY_a);
    double *Gamma = (double*)PyArray_DATA(Gamma_a);
    double *CDE_term = (double*)PyArray_DATA(CDE_term_a);
    double *w = (double*)PyArray_DATA(w_a);
    double *F_t = (double*)PyArray_DATA(F_t_a);
    double *wF_t = (double*)PyArray_DATA(wF_t_a);
    double *wF_t_sum = (double*)PyArray_DATA(wF_t_sum_a);
    double *p = (double*)PyArray_DATA(p_a);
    double *u = (double*)PyArray_DATA(u_a);
    int *Match = (int*)PyArray_DATA(Match_a);
    double *hatP = (double*)PyArray_DATA(hatP_a);
    double *hatPI_diag = (double*)PyArray_DATA(hatPI_diag_a);
    double *hatPIG = (double*)PyArray_DATA(hatPIG_a);
    double *hatPX = (double*)PyArray_DATA(hatPX_a);
    double *hatPIY = (double*)PyArray_DATA(hatPIY_a);
    double *G = (double*)PyArray_DATA(G_a);
    double *W = (double*)PyArray_DATA(W_a);
    double *GW = (double*)PyArray_DATA(GW_a);
    double *sumPoverN = (double*)PyArray_DATA(sumPoverN_a);
    double *expAlphaSumPoverN = (double*)PyArray_DATA(expAlphaSumPoverN_a);
    
    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////
    // Actual C code
    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////
    
    dsmm::_dsmm(X,Y,M,N,D,beta,lambda,neighbor_cutoff,alpha,gamma0,
           conv_epsilon,eq_tol,
           pwise_dist,pwise_distYY,Gamma,CDE_term,
           w,F_t,wF_t,wF_t_sum,p,u,Match,
           hatP,hatPI_diag,hatPIG,hatPX,hatPIY,
           G,W,GW,sumPoverN,expAlphaSumPoverN);
    
    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////
    // End of C code
    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////
    
    
    // Decrease the reference count for the python objects that have been 
    // declared in this function.
    Py_XDECREF(X_a);
    Py_XDECREF(Y_a);
    Py_XDECREF(pwise_dist_a);
    Py_XDECREF(pwise_distYY_a);
    Py_XDECREF(Gamma_a);
    Py_XDECREF(CDE_term_a);
    Py_XDECREF(w_a);
    Py_XDECREF(F_t_a);
    Py_XDECREF(wF_t_a);
    Py_XDECREF(wF_t_sum_a);
    Py_XDECREF(p_a);
    Py_XDECREF(u_a);
    Py_XDECREF(Match_a);
    Py_XDECREF(hatP_a);
    Py_XDECREF(hatPI_diag_a);
    Py_XDECREF(hatPIG_a);
    Py_XDECREF(hatPX_a);
    Py_XDECREF(hatPIY_a);
    Py_XDECREF(G_a);
    Py_XDECREF(W_a);
    Py_XDECREF(GW_a);
    Py_XDECREF(sumPoverN_a);
    Py_XDECREF(expAlphaSumPoverN_a);
    
    // Return the python object none. Its reference count has to be increased.
    Py_INCREF(Py_None);
    return Py_None;
}
