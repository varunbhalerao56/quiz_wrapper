#!/bin/bash

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <llama_cpp_path> <dev_team>"
    exit 1
fi

LLAMA_CPP_PATH="$1"
DEV_TEAM="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/bin/macos-arm64"
BUILD_DIR="${LLAMA_CPP_PATH}/build_macos_arm64"

echo "Building llama.cpp for macOS ARM64..."
echo "Source: ${LLAMA_CPP_PATH}"
echo "Output: ${OUTPUT_DIR}"

# Clean previous build
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
mkdir -p "${OUTPUT_DIR}"

cd "${BUILD_DIR}"

# Configure with CMake
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_METAL=ON \
  -DLLAMA_METAL_EMBED_LIBRARY=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 \
  -DCMAKE_INSTALL_RPATH="@loader_path" \
  -DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM="${DEV_TEAM}" \
  -DCMAKE_INSTALL_PREFIX="./install"

# Build
cmake --build . --config Release --parallel

# Install to temp location
cmake --install . --config Release

# Copy libraries to output directory
echo "Copying libraries..."
LIBS=(
  "install/lib/libllama.dylib"
  "install/lib/libggml.dylib"
  "install/lib/libggml-base.dylib"
  "install/lib/libggml-metal.dylib"
  "install/lib/libggml-cpu.dylib"
  "install/lib/libggml-blas.dylib"
  "install/lib/libmtmd.dylib"
)

for lib in "${LIBS[@]}"; do
  if [ -f "$lib" ]; then
    cp "$lib" "${OUTPUT_DIR}/"
    lib_name=$(basename "$lib")
    echo "  ✓ Copied ${lib_name}"
    
    # Get the base name without lib prefix and .dylib extension for framework naming
    base_name="${lib_name#lib}"
    base_name="${base_name%.dylib}"
    
    # Fix install name - use the framework style name
    install_name_tool -id "@rpath/${base_name}.framework/${base_name}" "${OUTPUT_DIR}/${lib_name}"
    
    # Update dependencies to use framework paths
    for dep_lib in "${LIBS[@]}"; do
      dep_name=$(basename "$dep_lib")
      dep_base="${dep_name#lib}"
      dep_base="${dep_base%.dylib}"
      
      if [ "$dep_name" != "$lib_name" ]; then
        # Change from @rpath/libXXX.dylib to @rpath/XXX.framework/XXX
        install_name_tool -change \
          "@rpath/${dep_name}" \
          "@rpath/${dep_base}.framework/${dep_base}" \
          "${OUTPUT_DIR}/${lib_name}" 2>/dev/null || true
        install_name_tool -change \
          "@loader_path/${dep_name}" \
          "@rpath/${dep_base}.framework/${dep_base}" \
          "${OUTPUT_DIR}/${lib_name}" 2>/dev/null || true
      fi
    done
    
    # Code sign
    codesign --force --sign - --timestamp=none "${OUTPUT_DIR}/${lib_name}"
    
  else
    echo "  ✗ Not found: $lib"
  fi
done

# Make libraries executable
chmod +x "${OUTPUT_DIR}"/*.dylib

echo ""
echo "Build completed successfully!"
echo "Libraries are in: ${OUTPUT_DIR}"