
#pragma once
#include <Python.h>

#define PY_RUN(...) PyRun_SimpleString( (const char*)#__VA_ARGS__ )

static PyThreadState* __py_ts = 0;

inline void PY_BEGIN()
{
	PyEval_AcquireLock();
	__py_ts = Py_NewInterpreter();
	PyThreadState_Swap(__py_ts);
}

inline void PY_END()
{
	PyThreadState_Swap(0);

	PyThreadState_Clear(__py_ts);
	//PyThreadState_Delete(__py_ts);
	//PyEval_ReleaseLock();
}

inline void setPyInt( const char* name, const int v )
{
	PyObject* main = PyImport_AddModule("__main__");
	PyObject* globals = PyModule_GetDict(main);
	PyObject* value = PyLong_FromLong(v);

	PyDict_SetItemString(globals, name, value);
}

inline int getPyInt( const char* name )
{
	PyObject* main = PyImport_AddModule("__main__");
	PyObject* value = PyObject_GetAttrString(main, name);

	return PyLong_AsLong(value);
}

/*
inline void python_get()
{
	PyObject* main = PyImport_AddModule("__main__");
	PyObject* globals = PyModule_GetDict(main);

	PyObject* value = PyLong_FromLong(2323);

	if (PyDict_SetItemString(globals, "n", value) < 0)
	{
		std::cout << "errorr" << std::endl;
	}
}

inline void python_set()
{
	PyObject* main = PyImport_AddModule("__main__");
	PyObject* vvv = PyObject_GetAttrString(main, "n");

	if (vvv == 0)
	{
		std::cout << "efefef" << std::endl;
	}

	std::cout << PyLong_AsLong(vvv) << std::endl;
}
*/

