
#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(utils, m) {
  m.def("init", []() {
    auto settings =
        Kokkos::InitializationSettings(); /* use default constructor */
    auto num_threads_chars = std::getenv("OMP_NUM_THREADS");
    if (num_threads_chars != nullptr) {
      settings.set_num_threads(std::stoi(num_threads_chars));
    }
    Kokkos::initialize(settings);
  });

  m.def("finalize", []() { Kokkos::finalize(); });
}
