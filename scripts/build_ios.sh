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

# Convert LLAMA_CPP_PATH to absolute path
LLAMA_CPP_PATH="$(cd "${LLAMA_CPP_PATH}" && pwd)"

IOS_TOOLCHAIN="${PROJECT_ROOT}/darwin/ios.toolchain.cmake"

# Auto-download iOS toolchain if missing
if [ ! -f "$IOS_TOOLCHAIN" ]; then
    echo "iOS toolchain not found. Downloading..."
    mkdir -p "${PROJECT_ROOT}/darwin"
    curl -L -o "$IOS_TOOLCHAIN" \
        "https://raw.githubusercontent.com/leetal/ios-cmake/master/ios.toolchain.cmake"
    
    if [ ! -f "$IOS_TOOLCHAIN" ]; then
        echo "Error: Failed to download iOS toolchain"
        exit 1
    fi
    echo "✓ iOS toolchain downloaded successfully"
    echo ""
fi

echo "Building llama.cpp for iOS (all platforms)..."
echo "Source: ${LLAMA_CPP_PATH}"
echo "Toolchain: ${IOS_TOOLCHAIN}"
echo ""

# Common CMake flags
CMAKE_COMMON_FLAGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=ON
  -DLLAMA_METAL=ON
  -DLLAMA_METAL_EMBED_LIBRARY=ON
  -DLLAMA_CURL=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_SERVER=OFF
  -DCMAKE_INSTALL_RPATH="@loader_path"
  -DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM="${DEV_TEAM}"
  -DCMAKE_TOOLCHAIN_FILE="${IOS_TOOLCHAIN}"
)

# Libraries to copy
LIBS=(
  "libllama.dylib"
  "libggml.dylib"
  "libggml-base.dylib"
  "libggml-metal.dylib"
  "libggml-cpu.dylib"
  "libggml-blas.dylib"
  "libmtmd.dylib"
)

# Function to build for a specific platform
build_platform() {
  local PLATFORM=$1
  local OUTPUT_DIR="${PROJECT_ROOT}/bin/${PLATFORM}"
  local BUILD_DIR="${LLAMA_CPP_PATH}/build_ios_${PLATFORM}"
  
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Building for ${PLATFORM}..."
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  
  # Clean previous build
  rm -rf "${BUILD_DIR}"
  mkdir -p "${BUILD_DIR}"
  mkdir -p "${OUTPUT_DIR}"
  
  cd "${BUILD_DIR}"
  
  # Configure with CMake
  cmake "${LLAMA_CPP_PATH}" \
    "${CMAKE_COMMON_FLAGS[@]}" \
    -DPLATFORM="${PLATFORM}" \
    -DDEPLOYMENT_TARGET=13.0 \
    -DCMAKE_INSTALL_PREFIX="./install"
  
  # Build
  cmake --build . --config Release --parallel
  
  # Install to temp location
  cmake --install . --config Release
  
  # Copy libraries to output directory
  echo "Copying libraries to ${OUTPUT_DIR}..."
  for lib in "${LIBS[@]}"; do
    if [ -f "install/lib/$lib" ]; then
      cp "install/lib/$lib" "${OUTPUT_DIR}/"
      lib_name=$(basename "$lib")
      echo "  ✓ Copied ${lib_name}"
      
      # Get the base name without lib prefix and .dylib extension
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
      echo "  ✗ Not found: install/lib/$lib"
    fi
  done
  
  # Make libraries executable
  chmod +x "${OUTPUT_DIR}"/*.dylib
  
  echo ""
}

# Build for all iOS platforms
echo "🍎 Starting iOS builds for all platforms..."
echo ""

build_platform "OS64"
build_platform "SIMULATORARM64"
build_platform "SIMULATOR64"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All iOS builds completed successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Libraries are in:"
echo "  - bin/OS64/           (iOS Device arm64)"
echo "  - bin/SIMULATORARM64/ (iOS Simulator arm64)"
echo "  - bin/SIMULATOR64/    (iOS Simulator x86_64)"
echo ""
echo "Next step: Run the XCFramework builder:"
echo "  ./darwin/build_xcframework.sh"