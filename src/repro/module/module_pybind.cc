

#include <pybind11/pybind11.h>

#include "test.hpp"

PYBIND11_MODULE(module, m) {
  m.def("pybench", [](int N = 10, int M = 10, int S = 10, int nrepeat = 10) {
    bench(N, M, S, nrepeat);
  });
}
