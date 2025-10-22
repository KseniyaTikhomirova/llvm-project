add_executable(sycl-ls sycl-ls.cpp)
add_dependencies(sycl-ls sycl)
target_include_directories(sycl-ls PRIVATE ${LIBSYCL_BUILD_INCLUDE_DIR})

target_link_libraries(sycl-ls
  PRIVATE
    ${LIBSYCL_SHARED_OUTPUT_NAME}
)

if (WIN32)
  # 0x900: Search for the dependency DLLs only in the System32 directory and in the directory with sycl-ls.exe
  target_link_options(sycl-ls PRIVATE LINKER:/DEPENDENTLOADFLAG:0x900)
endif()
install(TARGETS sycl-ls
  RUNTIME DESTINATION "bin" COMPONENT sycl-ls)
