set(LLVM_BUILD_BINARY_DIRS "${LLVM_BINARY_DIR}/bin/")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

list(APPEND LIBSYCL_TEST_DEPS
  libsycl-runtime-libraries
  FileCheck
  llvm-readobj
  )

add_lit_testsuite(check-sycl-dumps "Running ABI dump tests only"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS "-v"
  PARAMS "LIBSYCL_LIB_DUMPS_ONLY=True"
  DEPENDS ${LIBSYCL_TEST_DEPS}
  EXCLUDE_FROM_CHECK_ALL
  )

add_lit_testsuite(check-sycl "Running the SYCL regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS "-v"
  DEPENDS ${LIBSYCL_TEST_DEPS}
  EXCLUDE_FROM_CHECK_ALL
  )
set_target_properties(check-sycl PROPERTIES FOLDER "SYCL tests")
