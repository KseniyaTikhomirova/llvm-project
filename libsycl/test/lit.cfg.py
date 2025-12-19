# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "SYCL"

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest()

# suffixes: A list of file extensions to treat as test files.
dump_only_tests = bool(lit_config.params.get("LIBSYCL_LIB_DUMPS_ONLY", False))
if dump_only_tests:
    config.suffixes = [".dump"]  # Only run dump testing
else:
    config.suffixes = [
        ".cpp",
        ".dump",
    ]
config.excludes = ["Inputs"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# allow expanding substitutions that are based on other substitutions
config.recursiveExpansionLimit = 10

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.libsycl_obj_root, "test")

# Propagate some variables from the host environment.
llvm_config.with_system_environment(
    ["PATH", "OCL_ICD_FILENAMES"]
)

config.substitutions.append(("%python", '"%s"' % (sys.executable)))

# Propagate extra environment variables
if config.extra_environment:
    lit_config.note("Extra environment variables")
    for env_pair in config.extra_environment.split(","):
        [var, val] = env_pair.split("=")
        if val:
            llvm_config.with_environment(var, val)
            lit_config.note("\t" + var + "=" + val)
        else:
            lit_config.note("\tUnset " + var)
            llvm_config.with_environment(var, "")

# Configure LD_LIBRARY_PATH or corresponding os-specific alternatives
# Add 'libcxx' feature to filter out all SYCL abi tests when SYCL runtime
# is built with llvm libcxx. This feature is added for Linux only since MSVC
# CL compiler doesn't support to use llvm libcxx instead of MSVC STL.
if platform.system() == "Linux":
    config.available_features.add("linux")
    llvm_config.with_system_environment("LD_LIBRARY_PATH")
    llvm_config.with_environment(
        "LD_LIBRARY_PATH", config.libsycl_libs_dir, append_path=True
    )

llvm_config.with_environment("PATH", config.libsycl_tools_dir, append_path=True)

config.substitutions.append(("%sycl_libs_dir", config.libsycl_libs_dir))
config.substitutions.append(("%sycl_include", config.libsycl_include))
config.substitutions.append(("%sycl_source_dir", config.libsycl_source_dir))
config.substitutions.append(("%test_include_path", config.test_include_path))
config.substitutions.append(("%llvm_build_bin_dir", config.llvm_build_bin_dir))

config.substitutions.append(
    (
        "%sycl_options",
        " -lsycl"
        + " -isystem "
        + config.libsycl_include
        + " -isystem "
        + os.path.join(config.libsycl_include, "sycl")
        + " -L"
        + config.libsycl_libs_dir,
    )
)

additional_flags = config.cxx_flags.split(" ")

# Dump-only tests do not have clang available
if not dump_only_tests:
    llvm_config.use_clang(additional_flags=additional_flags)

# Set timeout for test = 10 mins
try:
    import psutil

    lit_config.maxIndividualTestTime = 600
except ImportError:
    pass