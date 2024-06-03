configure_file(__init__.py __init__.py COPYONLY)

pybind11_add_module(utils utils_pywrap.cc)
target_link_libraries(utils 
  PRIVATE
  ${kokkos_libs}
  )
