add_executable(sycl-ls sycl-ls.cpp)
add_dependencies(sycl-ls sycl)
target_include_directories(sycl-ls PRIVATE ${LIBSYCL_BUILD_INCLUDE_DIR})

# ktikhomi to move to common
set(LIB_NAME "sycl")
set(LIB_OUTPUT_NAME "${LIB_NAME}")
if (CMAKE_SYSTEM_NAME STREQUAL Windows)
  if (CMAKE_MSVC_RUNTIME_LIBRARY AND (NOT CMAKE_MSVC_RUNTIME_LIBRARY MATCHES "DLL$"))
      message(FATAL_ERROR "libsycl requires a DLL version of the MSVC CRT.")
  endif()
  if ((NOT CMAKE_MSVC_RUNTIME_LIBRARY AND uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
      OR (CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL "MultiThreadedDebugDLL"))
    set(LIB_OUTPUT_NAME "${LIB_OUTPUT_NAME}d")
  endif()
endif()

target_link_libraries(sycl-ls
  PRIVATE
    ${LIB_NAME}
)

if (WIN32)
  # 0x900: Search for the dependency DLLs only in the System32 directory and in the directory with sycl-ls.exe
  target_link_options(sycl-ls PRIVATE LINKER:/DEPENDENTLOADFLAG:0x900)
endif()
install(TARGETS sycl-ls
  RUNTIME DESTINATION "bin" COMPONENT sycl-ls)