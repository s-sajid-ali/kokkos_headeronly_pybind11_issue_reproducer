configure_file(__init__.py __init__.py COPYONLY)

pybind11_add_module(module module_pybind.cc)
target_link_libraries(module
  PRIVATE
  ${kokkos_libs})
