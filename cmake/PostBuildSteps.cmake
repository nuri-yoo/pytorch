# Post-build steps previously handled by setup.py's build_ext.run().
# These run as CMake install(SCRIPT) or install(CODE) commands.

if(NOT TORCH_INSTALL_LIB_DIR)
  set(TORCH_INSTALL_LIB_DIR lib)
endif()
if(NOT TORCH_INSTALL_INCLUDE_DIR)
  set(TORCH_INSTALL_INCLUDE_DIR include)
endif()

# --- Header wrapping with TORCH_STABLE_ONLY guards ---
# Wrap installed headers so they error when included with TORCH_STABLE_ONLY
# or TORCH_TARGET_VERSION defined. This is done at install time via a script.
install(CODE "
  set(_include_dir \"\${CMAKE_INSTALL_PREFIX}/${TORCH_INSTALL_INCLUDE_DIR}\")
  if(EXISTS \"\${_include_dir}\")
    message(STATUS \"Wrapping headers with TORCH_STABLE_ONLY guards...\")
    set(_header_extensions h hpp cuh)
    set(_exclude_patterns
      \"torch/headeronly/\"
      \"torch/csrc/stable/\"
      \"torch/csrc/inductor/aoti_torch/c/\"
      \"torch/csrc/inductor/aoti_torch/generated/\"
    )
    set(_wrap_marker \"#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)\")

    foreach(_ext IN ITEMS h hpp cuh)
      file(GLOB_RECURSE _headers \"\${_include_dir}/*.\${_ext}\")
      foreach(_header IN LISTS _headers)
        file(RELATIVE_PATH _rel \"\${_include_dir}\" \"\${_header}\")

        # Check exclusion patterns
        set(_excluded FALSE)
        foreach(_pat IN LISTS _exclude_patterns)
          string(FIND \"\${_rel}\" \"\${_pat}\" _pos)
          if(NOT _pos EQUAL -1)
            set(_excluded TRUE)
            break()
          endif()
        endforeach()
        if(_excluded)
          continue()
        endif()

        file(READ \"\${_header}\" _content)
        string(FIND \"\${_content}\" \"\${_wrap_marker}\" _already_wrapped)
        if(_already_wrapped EQUAL 0)
          continue()
        endif()

        set(_wrapped \"\${_wrap_marker}\\n\${_content}\\n#else\\n\")
        string(APPEND _wrapped
          \"#error \\\"This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined.\\\"\\n\")
        string(APPEND _wrapped
          \"#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)\\n\")
        file(WRITE \"\${_header}\" \"\${_wrapped}\")
      endforeach()
    endforeach()
  endif()
")

# --- Compile commands merging ---
# Merge compile_commands.json from build subdirectories.
# Write the script to a file to avoid CMake stripping newlines from multiline
# command arguments when passed through Ninja.
file(WRITE "${CMAKE_BINARY_DIR}/merge_compile_commands.py"
"import json, pathlib, itertools\n\
build = pathlib.Path('${CMAKE_BINARY_DIR}')\n\
ninja = list(build.glob('*compile_commands.json'))\n\
cmake_sub = list((build / 'torch' / 'lib' / 'build').glob('*/compile_commands.json')) if (build / 'torch' / 'lib' / 'build').exists() else []\n\
cmds = [e for f in itertools.chain(ninja, cmake_sub) for e in json.loads(f.read_text())]\n\
for c in cmds:\n\
    if c.get('command', '').startswith('gcc '):\n\
        c['command'] = 'g++ ' + c['command'][4:]\n\
out = pathlib.Path('${PROJECT_SOURCE_DIR}/compile_commands.json')\n\
new = json.dumps(cmds, indent=2)\n\
if not out.exists() or out.read_text() != new:\n\
    out.write_text(new)\n\
")
add_custom_target(merge_compile_commands ALL
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_BINARY_DIR}/merge_compile_commands.py"
  COMMENT "Merging compile_commands.json..."
  VERBATIM
)

# --- License concatenation ---
# Build the bundled license file for wheel distribution.
file(WRITE "${CMAKE_BINARY_DIR}/bundle_licenses.py"
"import sys, pathlib\n\
third_party = pathlib.Path('${PROJECT_SOURCE_DIR}/third_party')\n\
sys.path.insert(0, str(third_party))\n\
from build_bundled import create_bundled\n\
license_file = pathlib.Path('${PROJECT_SOURCE_DIR}/LICENSE')\n\
bsd_text = license_file.read_text()\n\
with license_file.open('a') as f:\n\
    f.write('\\n\\n')\n\
    create_bundled(str(third_party.resolve()), f, include_files=True)\n\
bundled = license_file.read_text()\n\
license_file.write_text(bsd_text)\n\
pathlib.Path('${CMAKE_BINARY_DIR}/LICENSES_BUNDLED.txt').write_text(bundled)\n\
")
add_custom_target(bundle_licenses ALL
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_BINARY_DIR}/bundle_licenses.py"
  COMMENT "Generating bundled license file..."
  VERBATIM
)
install(FILES "${CMAKE_BINARY_DIR}/LICENSES_BUNDLED.txt"
  DESTINATION "."
  RENAME "LICENSE"
  OPTIONAL
)

# --- Windows export library ---
if(WIN32 AND BUILD_PYTHON AND NOT BUILD_LIBTORCH_WHL)
  install(CODE "
    set(_export_lib \"\${CMAKE_INSTALL_PREFIX}/${TORCH_INSTALL_LIB_DIR}/_C.lib\")
    # The export lib is generated alongside the _C module
    if(EXISTS \"${CMAKE_BINARY_DIR}/torch/csrc/_C.lib\")
      file(INSTALL \"${CMAKE_BINARY_DIR}/torch/csrc/_C.lib\"
           DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${TORCH_INSTALL_LIB_DIR}\")
    endif()
  ")
endif()

# --- Runtime DLL bundling (Windows) ---
# The old CI scripts (copy.bat / copy_cpu.bat) copied runtime DLLs into the
# source tree before setuptools ran.  With scikit-build-core the wheel is
# built from the cmake install prefix, so we install them via cmake instead.
if(WIN32 AND BUILD_PYTHON)
  # OpenMP runtime (libiomp5md.dll) — required by torch_cpu.dll when MKL
  # threading uses Intel OpenMP.
  if(MKL_OPENMP_LIBRARY AND MKL_OPENMP_LIBRARY MATCHES "libiomp5md\\.lib$")
    get_filename_component(_omp_lib_dir "${MKL_OPENMP_LIBRARY}" DIRECTORY)
    get_filename_component(_omp_prefix "${_omp_lib_dir}" DIRECTORY)
    # The DLL lives in bin/ next to the lib/ that contains the import library.
    set(_omp_dll "${_omp_prefix}/bin/libiomp5md.dll")
    if(EXISTS "${_omp_dll}")
      install(FILES "${_omp_dll}" DESTINATION "${TORCH_INSTALL_LIB_DIR}")
    else()
      # Fallback: DLL in the same directory as the import library.
      file(GLOB _omp_dll_fallback "${_omp_lib_dir}/libiomp5md.dll")
      if(_omp_dll_fallback)
        install(FILES ${_omp_dll_fallback} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
      endif()
    endif()
    # Also install the stubs library if present (libiompstubs5md.dll).
    file(GLOB _omp_stubs "${_omp_prefix}/bin/libiompstubs5md.dll")
    if(NOT _omp_stubs)
      file(GLOB _omp_stubs "${_omp_lib_dir}/libiompstubs5md.dll")
    endif()
    if(_omp_stubs)
      install(FILES ${_omp_stubs} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
    endif()
  endif()

  # libuv (uv.dll) — required by torch distributed (gloo transport).
  if(USE_DISTRIBUTED)
    if(libuv_DLL_PATH AND EXISTS "${libuv_DLL_PATH}")
      install(FILES "${libuv_DLL_PATH}" DESTINATION "${TORCH_INSTALL_LIB_DIR}")
    elseif(DEFINED ENV{libuv_ROOT})
      file(GLOB _uv_dll "$ENV{libuv_ROOT}/bin/uv.dll")
      if(_uv_dll)
        install(FILES ${_uv_dll} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
      endif()
    endif()
  endif()

  # CUDA runtime DLLs — only for CUDA builds.
  if(USE_CUDA AND CUDA_TOOLKIT_ROOT_DIR)
    # CUDA 13+ moves DLLs to bin/x64.
    if(IS_DIRECTORY "${CUDA_TOOLKIT_ROOT_DIR}/bin/x64")
      set(_cuda_bin "${CUDA_TOOLKIT_ROOT_DIR}/bin/x64")
    else()
      set(_cuda_bin "${CUDA_TOOLKIT_ROOT_DIR}/bin")
    endif()
    set(_cuda_dll_patterns
      "${_cuda_bin}/cusparse*64_*.dll"
      "${_cuda_bin}/cublas*64_*.dll"
      "${_cuda_bin}/cudart*64_*.dll"
      "${_cuda_bin}/curand*64_*.dll"
      "${_cuda_bin}/cufft*64_*.dll"
      "${_cuda_bin}/cusolver*64_*.dll"
      "${_cuda_bin}/nvrtc*64_*.dll"
      "${_cuda_bin}/nvJitLink_*.dll"
      "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudnn*64_*.dll"
      "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64/cupti64_*.dll"
      "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64/nvperf_host*.dll"
    )
    foreach(_pattern ${_cuda_dll_patterns})
      file(GLOB _dlls "${_pattern}")
      if(_dlls)
        install(FILES ${_dlls} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
      endif()
    endforeach()

    # NvToolsExt (legacy, may not exist on all systems).
    set(_nvtoolsext "C:/Program Files/NVIDIA Corporation/NvToolsExt/bin/x64/nvToolsExt64_1.dll")
    if(EXISTS "${_nvtoolsext}")
      install(FILES "${_nvtoolsext}" DESTINATION "${TORCH_INSTALL_LIB_DIR}")
    endif()

    # zlibwapi (needed by some CUDA libraries).
    if(EXISTS "C:/Windows/System32/zlibwapi.dll")
      install(FILES "C:/Windows/System32/zlibwapi.dll"
              DESTINATION "${TORCH_INSTALL_LIB_DIR}")
    endif()
  endif()
endif()

# --- macOS OpenMP embedding ---
# Copy libomp.dylib / libiomp5.dylib into the wheel and fix rpaths so the
# wheel is self-contained (replicates setup.py's _embed_libomp).
if(APPLE AND BUILD_PYTHON AND OPENMP_FOUND)
  # OpenMP_libomp_LIBRARY is set by our FindOpenMP module to the full path
  # of the OpenMP shared library (e.g. /path/to/libomp.dylib).
  if(OpenMP_libomp_LIBRARY AND EXISTS "${OpenMP_libomp_LIBRARY}")
    get_filename_component(_omp_name "${OpenMP_libomp_LIBRARY}" NAME)
    install(FILES "${OpenMP_libomp_LIBRARY}"
            DESTINATION "${TORCH_INSTALL_LIB_DIR}")
    # Install omp.h so Inductor's C++ backend can find it at runtime.
    # The header lives at <prefix>/include/omp.h next to <prefix>/lib/libomp.dylib.
    get_filename_component(_omp_lib_dir "${OpenMP_libomp_LIBRARY}" DIRECTORY)
    get_filename_component(_omp_prefix "${_omp_lib_dir}" DIRECTORY)
    if(EXISTS "${_omp_prefix}/include/omp.h")
      install(FILES "${_omp_prefix}/include/omp.h"
              DESTINATION "${TORCH_INSTALL_INCLUDE_DIR}")
    endif()
    # Fix libtorch_cpu's rpath so it finds the bundled library at load time.
    install(CODE "
      set(_lib_dir \"\${CMAKE_INSTALL_PREFIX}/${TORCH_INSTALL_LIB_DIR}\")
      set(_libtorch_cpu \"\${_lib_dir}/libtorch_cpu.dylib\")
      if(EXISTS \"\${_libtorch_cpu}\")
        execute_process(
          COMMAND install_name_tool -add_rpath @loader_path \"\${_libtorch_cpu}\"
          ERROR_QUIET
        )
        # Point the load command at @rpath so it picks up the bundled copy.
        execute_process(
          COMMAND install_name_tool -change
            \"${OpenMP_libomp_LIBRARY}\" \"@rpath/${_omp_name}\"
            \"\${_libtorch_cpu}\"
          ERROR_QUIET
        )
      endif()
    ")
  endif()
endif()
