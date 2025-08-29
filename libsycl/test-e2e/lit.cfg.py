# -*- Python -*-

import os
import platform
import copy
import re
import subprocess
import textwrap
import shlex
import shutil

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst, FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "SYCL"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".cpp"]

config.excludes = ["Inputs"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = config.sycl_obj_root

# allow expanding substitutions that are based on other substitutions
config.recursiveExpansionLimit = 10

# To be filled by lit.local.cfg files.
config.required_features = []
config.unsupported_features = []

# Dummy substitution to indicate line should be a run line
config.substitutions.append(("%{run-aux}", ""))

# Cleanup environment variables which may affect tests
possibly_dangerous_env_vars = [
    "COMPILER_PATH",
    "RC_DEBUG_OPTIONS",
    "CINDEXTEST_PREAMBLE_FILE",
    "LIBRARY_PATH",
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "OBJC_INCLUDE_PATH",
    "OBJCPLUS_INCLUDE_PATH",
    "LIBCLANG_TIMING",
    "LIBCLANG_OBJTRACKING",
    "LIBCLANG_LOGGING",
    "LIBCLANG_BGPRIO_INDEX",
    "LIBCLANG_BGPRIO_EDIT",
    "LIBCLANG_NOTHREADS",
    "LIBCLANG_RESOURCE_USAGE",
    "LIBCLANG_CODE_COMPLETION_LOGGING",
]

# Clang/Win32 may refer to %INCLUDE%. vsvarsall.bat sets it.
if platform.system() != "Windows":
    possibly_dangerous_env_vars.append("INCLUDE")

for name in possibly_dangerous_env_vars:
    if name in llvm_config.config.environment:
        del llvm_config.config.environment[name]

# Propagate some variables from the host environment.
llvm_config.with_system_environment(
    [
        "PATH",
        "OCL_ICD_FILENAMES",
        "OCL_ICD_VENDORS",
        "CL_CONFIG_DEVICES"
    ]
)

# Take into account extra system environment variables if provided via parameter.
if config.extra_system_environment:
    lit_config.note(
        "Extra system variables to propagate value from: "
        + config.extra_system_environment
    )
    extra_env_vars = config.extra_system_environment.split(",")
    for var in extra_env_vars:
        if var in os.environ:
            llvm_config.with_system_environment(var)

llvm_config.with_environment("PATH", config.lit_tools_dir, append_path=True)

# Configure LD_LIBRARY_PATH or corresponding os-specific alternatives
if platform.system() == "Linux":
    config.available_features.add("linux")
    llvm_config.with_system_environment(
        ["LD_LIBRARY_PATH", "LIBRARY_PATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"]
    )
    llvm_config.with_environment(
        "LD_LIBRARY_PATH", config.sycl_libs_dir, append_path=True
    )

elif platform.system() == "Windows":
    config.available_features.add("windows")
    llvm_config.with_system_environment(
        ["LIB", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "INCLUDE"]
    )
    llvm_config.with_environment("LIB", config.sycl_libs_dir, append_path=True)
    llvm_config.with_environment("PATH", config.sycl_libs_dir, append_path=True)
    llvm_config.with_environment(
        "LIB", os.path.join(config.dpcpp_root_dir, "lib"), append_path=True
    )

elif platform.system() == "Darwin":
    # FIXME: surely there is a more elegant way to instantiate the Xcode directories.
    llvm_config.with_system_environment(["C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"])
    llvm_config.with_environment(
        "CPLUS_INCLUDE_PATH",
        "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1",
        append_path=True,
    )
    llvm_config.with_environment(
        "C_INCLUDE_PATH",
        "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/",
        append_path=True,
    )
    llvm_config.with_environment(
        "CPLUS_INCLUDE_PATH",
        "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/",
        append_path=True,
    )
    llvm_config.with_environment("DYLD_LIBRARY_PATH", config.sycl_libs_dir)

llvm_config.with_environment("PATH", config.sycl_tools_dir, append_path=True)

if config.extra_environment:
    lit_config.note("Extra environment variables")
    for env_pair in config.extra_environment.split(","):
        [var, val] = env_pair.split("=", 1)
        if val:
            llvm_config.with_environment(var, val)
            lit_config.note("\t" + var + "=" + val)
        else:
            lit_config.note("\tUnset " + var)
            llvm_config.with_environment(var, "")

# Disable the UR logger callback sink during test runs as output to SYCL RT can interfere with some tests relying on standard input/output
llvm_config.with_environment("UR_LOG_CALLBACK", "disabled")


# Temporarily modify environment to be the same that we use when running tests
class test_env:
    def __enter__(self):
        self.old_environ = dict(os.environ)
        os.environ.clear()
        os.environ.update(config.environment)
        self.old_dir = os.getcwd()
        os.chdir(config.sycl_obj_root)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        os.environ.clear()
        os.environ.update(self.old_environ)
        os.chdir(self.old_dir)


config.substitutions.append(("%sycl_libs_dir", config.sycl_libs_dir))
if platform.system() == "Windows":
    config.substitutions.append(
        ("%sycl_static_libs_dir", config.sycl_libs_dir + "/../lib")
    )
    config.substitutions.append(("%obj_ext", ".obj"))
    config.substitutions.append(
        ("%sycl_include", "-Xclang -isystem -Xclang " + config.sycl_include)
    )
elif platform.system() == "Linux":
    config.substitutions.append(("%sycl_static_libs_dir", config.sycl_libs_dir))
    config.substitutions.append(("%obj_ext", ".o"))
    config.substitutions.append(("%sycl_include", "-isystem " + config.sycl_include))

def quote_path(path):
    if not path:
        return ""
    if platform.system() == "Windows":
        return f'"{path}"'
    return shlex.quote(path)

# Use this to make sure that any dynamic checks below are done in the build
# directory and not where the sources are located. This is important for the
# in-tree configuration (as opposite to the standalone one).
def open_check_file(file_name):
    return open(os.path.join(config.sycl_obj_root, file_name), "w")


# check if compiler supports CL command line options
cl_options = False
with test_env():
    sp = subprocess.getstatusoutput(config.sycl_compiler + " /help")
    if sp[0] == 0:
        cl_options = True
        config.available_features.add("cl_options")

# check if the compiler was built in NDEBUG configuration
has_ndebug = False
ps = subprocess.Popen(
    [config.sycl_compiler, "-mllvm", "-debug", "-x", "c", "-", "-S", "-o", os.devnull],
    stdin=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
_ = ps.communicate(input=b"int main(){}\n")
if ps.wait() == 0:
    config.available_features.add("has_ndebug")

# Check for Level Zero SDK
check_l0_file = "l0_include.cpp"
with open_check_file(check_l0_file) as fp:
    print(
        textwrap.dedent(
            """
        #include <level_zero/ze_api.h>
        int main() { uint32_t t; zeDriverGet(&t, nullptr); return t; }
        """
        ),
        file=fp,
    )

config.level_zero_libs_dir = quote_path(
    lit_config.params.get("level_zero_libs_dir", config.level_zero_libs_dir)
)
config.level_zero_include = quote_path(
    lit_config.params.get(
        "level_zero_include",
        (
            config.level_zero_include
            if config.level_zero_include
            else config.sycl_include
        ),
    )
)

level_zero_options = level_zero_options = (
    (" -L" + config.level_zero_libs_dir if config.level_zero_libs_dir else "")
    + " -lze_loader "
    + " -I"
    + config.level_zero_include
)
if cl_options:
    level_zero_options = (
        " "
        + (
            config.level_zero_libs_dir + "/ze_loader.lib "
            if config.level_zero_libs_dir
            else "ze_loader.lib"
        )
        + " /I"
        + config.level_zero_include
    )

config.substitutions.append(("%level_zero_options", level_zero_options))

with test_env():
    sp = subprocess.getstatusoutput(
        config.sycl_compiler + " -fsycl  " + check_l0_file + level_zero_options
    )
    if sp[0] == 0:
        config.available_features.add("level_zero_dev_kit")
        config.substitutions.append(("%level_zero_options", level_zero_options))
    else:
        config.substitutions.append(("%level_zero_options", ""))

# Check if clang is built with ZSTD and compression support.
fPIC_opt = "-fPIC" if platform.system() != "Windows" else ""
# -shared is invalid for icx on Windows, use /LD instead.
dll_opt = "/LD" if cl_options else "-shared"

ps = subprocess.Popen(
    [
        config.sycl_compiler,
        "-fsycl",
        "--offload-compress",
        dll_opt,
        fPIC_opt,
        "-x",
        "c++",
        "-",
        "-o",
        os.devnull,
    ],
    stdin=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
op = ps.communicate(input=b"")
if ps.wait() == 0:
    config.available_features.add("zstd")

# Check for CUDA SDK
check_cuda_file = "cuda_include.cpp"
with open_check_file(check_cuda_file) as fp:
    print(
        textwrap.dedent(
            """
        #include <cuda.h>
        int main() { CUresult r = cuInit(0); return r; }
        """
        ),
        file=fp,
    )

config.cuda_libs_dir = quote_path(
    lit_config.params.get("cuda_libs_dir", config.cuda_libs_dir)
)
config.cuda_include = quote_path(
    lit_config.params.get(
        "cuda_include",
        (config.cuda_include if config.cuda_include else config.sycl_include),
    )
)

cuda_options = (
    (" -L" + config.cuda_libs_dir if config.cuda_libs_dir else "")
    + " -lcuda "
    + " -I"
    + config.cuda_include
)
if cl_options:
    cuda_options = (
        " "
        + (config.cuda_libs_dir + "/cuda.lib " if config.cuda_libs_dir else "cuda.lib")
        + " /I"
        + config.cuda_include
    )
if platform.system() == "Windows":
    cuda_options += (
        " --cuda-path=" + os.path.dirname(os.path.dirname(config.cuda_libs_dir)) + f'"'
    )

config.substitutions.append(("%cuda_options", cuda_options))

with test_env():
    sp = subprocess.getstatusoutput(
        config.sycl_compiler + " -fsycl  " + check_cuda_file + cuda_options
    )
    if sp[0] == 0:
        config.available_features.add("cuda_dev_kit")
        config.substitutions.append(("%cuda_options", cuda_options))
    else:
        config.substitutions.append(("%cuda_options", ""))

# Check for HIP SDK
check_hip_file = "hip_include.cpp"
with open_check_file(check_hip_file) as fp:
    print(
        textwrap.dedent(
            """
        #define __HIP_PLATFORM_AMD__ 1
        #include <hip/hip_runtime.h>
        int main() {  hipError_t r = hipInit(0); return r; }
        """
        ),
        file=fp,
    )
config.hip_libs_dir = quote_path(
    lit_config.params.get("hip_libs_dir", config.hip_libs_dir)
)
config.hip_include = quote_path(
    lit_config.params.get(
        "hip_include",
        (config.hip_include if config.hip_include else config.sycl_include),
    )
)

hip_options = (
    (" -L" + config.hip_libs_dir if config.hip_libs_dir else "")
    + " -lamdhip64 "
    + " -I"
    + config.hip_include
)
if cl_options:
    hip_options = (
        " "
        + (
            config.hip_libs_dir + "/amdhip64.lib "
            if config.hip_libs_dir
            else "amdhip64.lib"
        )
        + " /I"
        + config.hip_include
    )
if platform.system() == "Windows":
    hip_options += " --rocm-path=" + os.path.dirname(config.hip_libs_dir) + f'"'
with test_env():
    sp = subprocess.getstatusoutput(
        config.sycl_compiler + " -fsycl  " + check_hip_file + hip_options
    )
    if sp[0] == 0:
        config.available_features.add("hip_dev_kit")
        config.substitutions.append(("%hip_options", hip_options))
    else:
        config.substitutions.append(("%hip_options", ""))

# Add ROCM_PATH from system environment, this is used by clang to find ROCm
# libraries in non-standard installation locations.
llvm_config.with_system_environment("ROCM_PATH")

# Check for OpenCL ICD
if config.opencl_libs_dir:
    config.opencl_libs_dir = quote_path(config.opencl_libs_dir)
    config.opencl_include_dir = quote_path(config.opencl_include_dir)
    if cl_options:
        config.substitutions.append(
            ("%opencl_lib", " " + config.opencl_libs_dir + "/OpenCL.lib")
        )
    else:
        config.substitutions.append(
            ("%opencl_lib", "-L" + config.opencl_libs_dir + " -lOpenCL")
        )
    config.available_features.add("opencl_icd")
config.substitutions.append(("%opencl_include_dir", config.opencl_include_dir))

if cl_options:
    config.substitutions.append(
        (
            "%sycl_options",
            " "
            + os.path.normpath(os.path.join(config.sycl_libs_dir + "/../lib/sycl.lib"))
            + " -Xclang -isystem -Xclang "
            + config.sycl_include
            + " -Xclang -isystem -Xclang "
            + os.path.join(config.sycl_include, "sycl"),
        )
    )
    config.substitutions.append(("%include_option", "/FI"))
    config.substitutions.append(("%debug_option", "/Zi /DEBUG"))
    config.substitutions.append(("%cxx_std_option", "/clang:-std="))
    config.substitutions.append(("%fPIC", ""))
    config.substitutions.append(("%shared_lib", "/LD"))
    config.substitutions.append(("%O0", "/Od"))
    config.substitutions.append(("%fp-model-", "/fp:"))
else:
    config.substitutions.append(
        (
            "%sycl_options",
            (" -lsycl")
            + " -isystem "
            + config.sycl_include
            + " -isystem "
            + os.path.join(config.sycl_include, "sycl")
            + " -L"
            + config.sycl_libs_dir,
        )
    )
    config.substitutions.append(("%include_option", "-include"))
    config.substitutions.append(("%debug_option", "-g"))
    config.substitutions.append(("%cxx_std_option", "-std="))
    # Position-independent code does not make sence on Windows. At the same
    # time providing this option for compilation targeting
    # x86_64-pc-windows-msvc will cause compile time error on some
    # configurations
    config.substitutions.append(
        ("%fPIC", ("" if platform.system() == "Windows" else "-fPIC"))
    )
    config.substitutions.append(("%shared_lib", "-shared"))
    config.substitutions.append(("%O0", "-O0"))
    config.substitutions.append(("%fp-model-", "-ffp-model="))

# Check if user passed verbose-print parameter, if yes, add VERBOSE_PRINT macro
if "verbose-print" in lit_config.params:
    config.substitutions.append(("%verbose_print", "-DVERBOSE_PRINT"))
else:
    config.substitutions.append(("%verbose_print", ""))

config.substitutions.append(("%threads_lib", config.sycl_threads_lib))

if config.run_launcher:
    config.substitutions.append(("%e2e_tests_root", config.test_source_root))

# Try and find each of these tools in the DPC++ bin directory, in the llvm tools directory
# or the PATH, in that order. If found, they will be added as substitutions with the full path
# to the tool. This allows us to support both in-tree builds and standalone
# builds, where the tools may be externally defined.
# The DPC++ bin directory is different from the LLVM bin directory when using
# the Intel Compiler (icx), which puts tools into $dpcpp_root_dir/bin/compiler
llvm_config.add_tool_substitutions(
    tools, [config.dpcpp_bin_dir, config.llvm_tools_dir, os.environ.get("PATH", "")]
)
for tool in feature_tools:
    if tool.was_resolved:
        config.available_features.add(tool.key)
    else:
        lit_config.warning("Can't find " + tool.key)

if shutil.which("cmc") is not None:
    config.available_features.add("cm-compiler")

# Clear build targets when not in build-only, to populate according to devices
if config.test_mode != "build-only":
    config.sycl_build_targets = set()

if lit_config.params.get("compatibility_testing", "False") != "False":
    config.substitutions.append(("%clangxx", " true "))
    config.substitutions.append(("%clang", " true "))
else:
    clangxx = " " + config.sycl_compiler + " "
    clangxx += config.cxx_flags
    config.substitutions.append(("%clangxx", clangxx))

# Set timeout for a single test
try:
    import psutil

    if config.test_mode == "run-only":
        lit_config.maxIndividualTestTime = 300
    else:
        lit_config.maxIndividualTestTime = 600

except ImportError:
    pass