#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 🛠  CONFIGURATION – Adjust names/paths for your project
###############################################################################
FRAMEWORK_NAME="Llama"
# NOTE: MAIN_LIB_NAME is the dylib that Dart FFI will load (DynamicLibrary.open)
MAIN_LIB_NAME="libmtmd.dylib" 
FRAMEWORK_VERSION="0.1.0"
FRAMEWORK_ID="com.example.quiz_wrapper"

MIN_IOS_VERSION="13.0"
MIN_MACOS_VERSION="12.0"

# Library dependencies - ORDER MATTERS for proper path fixing!
DEPENDENCY_LIBS=(
  "libggml-base.dylib"
  "libggml-blas.dylib"
  "libggml-cpu.dylib"
  "libggml-metal.dylib"
  "libggml.dylib"
  "libmtmd.dylib" # Included here only for fix_dylib_paths in macOS
  "libllama.dylib" # Included here for fix_dylib_paths if needed
)

# Paths where YOUR build system placed the thin architectures
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IOS_DEVICE_LIB_DIR="${PROJECT_ROOT}/bin/OS64"
IOS_SIM_ARM64_LIB_DIR="${PROJECT_ROOT}/bin/SIMULATORARM64"
IOS_SIM_X86_64_LIB_DIR="${PROJECT_ROOT}/bin/SIMULATOR64"
# Using macOS-arm64 directory from File A's naming
MACOS_ARM64_LIB_DIR="${PROJECT_ROOT}/bin/macos-arm64" 

HEADER_SOURCE_DIRS=(
  "${PROJECT_ROOT}/third_party/llama.cpp/include"
  "${PROJECT_ROOT}/third_party/llama.cpp/ggml/include"
)

PLIST_TEMPLATE="${PROJECT_ROOT}/darwin/Info.plist" # Assumes File B's PLIST template location

###############################################################################
# 🔑 CODE SIGNING & PATHS
###############################################################################
IDENTITY="${SIGN_IDENTITY:-${EXPANDED_CODE_SIGN_IDENTITY:-}}"
if [[ -z "$IDENTITY" ]]; then
  echo "🔓 No signing identity supplied – will output *UNSIGNED* XCFramework (Ad-hoc signing will be applied)"
else
  echo "🔐 Using signing identity: $IDENTITY"
fi
echo

TMP_DIR="${PROJECT_ROOT}/build/xcframework_temp"
OUTPUT_DIR="${PROJECT_ROOT}/dist"
OUTPUT_XCFW="${OUTPUT_DIR}/${FRAMEWORK_NAME}.xcframework"

EXECUTABLE_NAME="$FRAMEWORK_NAME"
# Use @rpath/Frameworks/ for dependencies, @rpath/FrameworkName/ExecutableName for main binary
INSTALL_NAME="@rpath/${FRAMEWORK_NAME}.framework/$EXECUTABLE_NAME"
DEPENDENCY_PATH="@loader_path/Frameworks/"

IOS_DEVICE_FW="$TMP_DIR/ios-arm64/${FRAMEWORK_NAME}.framework"
IOS_SIM_FW="$TMP_DIR/ios-arm64_x86_64-simulator/${FRAMEWORK_NAME}.framework"
MACOS_FW="$TMP_DIR/macos-arm64/${FRAMEWORK_NAME}.framework"

rm -rf "$TMP_DIR" "$OUTPUT_XCFW"
mkdir -p "$TMP_DIR" "$OUTPUT_DIR"

###############################################################################
# 🔧 HELPER FUNCTIONS (COMBINED & IMPROVED)
###############################################################################

codesign_bin() {
  local bin="$1"
  # File B's robust signing with Hardened Runtime option
  if [[ -z "$IDENTITY" ]]; then 
    /usr/bin/codesign --force --sign - "$bin" &>/dev/null || true # Ad-hoc sign for unsigned build
  else
    if /usr/bin/codesign --force --options=runtime --timestamp \
                         --sign "$IDENTITY" "$bin" &>/dev/null; then
      echo "✅  codesigned  $(basename "$bin")"
    else
      echo "❌  failed to sign $(basename "$bin")" >&2
      exit 1
    fi
  fi
}

strip_signature() {
  /usr/bin/codesign --remove-signature "$1" 2>/dev/null || true
}

# --- File A's Comprehensive Path Fixing ---
# This fixes internal references within the dylibs themselves
fix_internal_dylib_paths() {
  local binary="$1"
  
  for lib in "${DEPENDENCY_LIBS[@]}"; do
    # File B places all dependencies in Frameworks/
    local target_path="${DEPENDENCY_PATH}$(basename "$lib")"

    # Fix direct @loader_path or absolute paths to the main framework path
    install_name_tool -id "@loader_path/$(basename "$lib")" "$binary" 2>/dev/null || true

    # 1. Change from framework reference (e.g., @rpath/ggml.framework/ggml)
    local lib_basename=$(basename "$lib" .dylib)
    install_name_tool -change "@rpath/${lib_basename}.framework/${lib_basename}" "$target_path" \
      "$binary" 2>/dev/null || true
    
    # 2. Change from direct dylib reference (e.g., @rpath/libggml.dylib)
    install_name_tool -change "@rpath/$lib" "$target_path" \
      "$binary" 2>/dev/null || true
    
    # 3. Change from internal @loader_path reference (e.g., @loader_path/libggml.dylib)
    # Note: We only change this inside the *main* framework binary. 
    # For dependencies, we only set the ID (below).
    if [[ "$(basename "$binary")" == "$EXECUTABLE_NAME" ]]; then
      install_name_tool -change "@loader_path/$lib" "$target_path" \
        "$binary" 2>/dev/null || true
    fi

  done
}

# --- File B's Generalized Build Logic, Enhanced ---
build_slice() {
  local TARGET_DIR=$1 PLATFORM=$2 MIN_OS=$3 MAIN_LIB=$4 DEP_DIR=$5

  echo "▶️  Building slice: $PLATFORM"
  mkdir -p "$TARGET_DIR/Headers" "$TARGET_DIR/Frameworks"

  # 1. Main dylib copy
  cp "$MAIN_LIB" "$TARGET_DIR/$EXECUTABLE_NAME"
  install_name_tool -id "$INSTALL_NAME" "$TARGET_DIR/$EXECUTABLE_NAME"
  
  # Set the install name for the main binary to reference dependencies in Frameworks/
  for DEP in "${DEPENDENCY_LIBS[@]}"; do
    if [[ "$DEP" != "$MAIN_LIB_NAME" ]]; then
        install_name_tool -change "@loader_path/$DEP" "${DEPENDENCY_PATH}$DEP" \
            "$TARGET_DIR/$EXECUTABLE_NAME" 2>/dev/null || true
    fi
  done
  
  # 2. Dependencies copy, path fix, and sign
  for DEP in "${DEPENDENCY_LIBS[@]}"; do
    [[ "$DEP" == "$MAIN_LIB_NAME" ]] && continue # Skip main lib
    [[ -f "$DEP_DIR/$DEP" ]] || { echo "⚠️   missing $DEP in $DEP_DIR"; continue; }
    
    local DEP_PATH="$TARGET_DIR/Frameworks/$DEP"
    cp "$DEP_DIR/$DEP" "$DEP_PATH"
    
    # Set install name for the dependency itself to @loader_path/lib.dylib (File A logic)
    install_name_tool -id "@loader_path/$DEP" "$DEP_PATH"

    # Fix internal references *within* the dependency dylib (File A logic)
    fix_internal_dylib_paths "$DEP_PATH"
    
    strip_signature "$DEP_PATH"
    codesign_bin "$DEP_PATH"
  done

  # 3. Headers (File B)
  for H in "${HEADER_SOURCE_DIRS[@]}"; do cp -R "$H/." "$TARGET_DIR/Headers/" 2>/dev/null || true; done

  # 4. Info.plist (File B)
  sed -e "s/__NAME__/${FRAMEWORK_NAME}/g" \
      -e "s/__EXECUTABLE__/${EXECUTABLE_NAME}/g" \
      -e "s/__IDENTIFIER__/${FRAMEWORK_ID}/g" \
      -e "s/__VERSION__/${FRAMEWORK_VERSION}/g" \
      -e "s/__MIN_OS_VERSION__/${MIN_OS}/g" \
      "$PLIST_TEMPLATE" > "$TARGET_DIR/Info.plist"

  # 5. Final Code Sign
  strip_signature "$TARGET_DIR/$EXECUTABLE_NAME"
  codesign_bin     "$TARGET_DIR/$EXECUTABLE_NAME"
  echo
}

###############################################################################
# 🛠  BUILD SLICES
###############################################################################

# --- iOS Device arm64 ---
build_slice "$IOS_DEVICE_FW" "iOS Device arm64" \
            "$MIN_IOS_VERSION" "$IOS_DEVICE_LIB_DIR/$MAIN_LIB_NAME" "$IOS_DEVICE_LIB_DIR"

# --- macOS arm64 ---
# NOTE: File B's logic is used here, ensuring dependencies go into Frameworks/
# unlike File A's macos logic. This maintains structure consistency.
build_slice "$MACOS_FW" "macOS arm64" \
            "$MIN_MACOS_VERSION" "$MACOS_ARM64_LIB_DIR/$MAIN_LIB_NAME" "$MACOS_ARM64_LIB_DIR"

# --- iOS Simulator universal (arm64 + x86_64) ---
echo "▶️  Building slice: iOS Simulator universal"
mkdir -p "$IOS_SIM_FW/Headers" "$IOS_SIM_FW/Frameworks"

# 1. Create universal binary for main library (File B)
lipo -create "$IOS_SIM_ARM64_LIB_DIR/$MAIN_LIB_NAME" \
              "$IOS_SIM_X86_64_LIB_DIR/$MAIN_LIB_NAME" \
     -output "$IOS_SIM_FW/$EXECUTABLE_NAME"
install_name_tool -id "$INSTALL_NAME" "$IOS_SIM_FW/$EXECUTABLE_NAME"

# 2. Universal dependencies (File B logic + File A's fix_internal_dylib_paths)
for DEP in "${DEPENDENCY_LIBS[@]}"; do
  [[ "$DEP" == "$MAIN_LIB_NAME" ]] && continue
  if [[ -f "$IOS_SIM_ARM64_LIB_DIR/$DEP" && -f "$IOS_SIM_X86_64_LIB_DIR/$DEP" ]]; then
    DEP_PATH="$IOS_SIM_FW/Frameworks/$DEP"
    lipo -create "$IOS_SIM_ARM64_LIB_DIR/$DEP" "$IOS_SIM_X86_64_LIB_DIR/$DEP" \
         -output "$DEP_PATH"

    # Change main binary to reference the dependency
    install_name_tool -change "@rpath/$DEP" "${DEPENDENCY_PATH}$DEP" \
                      "$IOS_SIM_FW/$EXECUTABLE_NAME" 2>/dev/null || true
    
    # Set install name for the dependency itself
    install_name_tool -id "@loader_path/$DEP" "$DEP_PATH"
    
    # Fix internal references *within* the dependency dylib
    fix_internal_dylib_paths "$DEP_PATH"
    
    strip_signature "$DEP_PATH"
    codesign_bin "$DEP_PATH"
  fi
done

# 3. Headers, Info.plist, and Final Sign (File B)
for H in "${HEADER_SOURCE_DIRS[@]}"; do cp -R "$H/." "$IOS_SIM_FW/Headers/" 2>/dev/null || true; done
sed -e "s/__NAME__/${FRAMEWORK_NAME}/g" \
    -e "s/__EXECUTABLE__/${EXECUTABLE_NAME}/g" \
    -e "s/__IDENTIFIER__/${FRAMEWORK_ID}/g" \
    -e "s/__VERSION__/${FRAMEWORK_VERSION}/g" \
    -e "s/__MIN_OS_VERSION__/${MIN_IOS_VERSION}/g" \
    "$PLIST_TEMPLATE" > "$IOS_SIM_FW/Info.plist"

strip_signature "$IOS_SIM_FW/$EXECUTABLE_NAME"; codesign_bin "$IOS_SIM_FW/$EXECUTABLE_NAME"
echo

###############################################################################
# 🎁  CREATE XCFRAMEWORK (File B)
###############################################################################
echo "📦  Assembling XCFramework…"
xcodebuild -quiet -create-xcframework \
  -framework "$IOS_DEVICE_FW" \
  -framework "$IOS_SIM_FW" \
  -framework "$MACOS_FW" \
  -output "$OUTPUT_XCFW"
echo "✅  XCFramework written to $OUTPUT_XCFW"
echo

###############################################################################
# 🔍  FINAL VERIFICATION TABLE (File B)
###############################################################################
printf "┌%-38s┬%-12s┐\n" " Binary " "Team ID"
find "$OUTPUT_XCFW" -type f \( -name "$EXECUTABLE_NAME" -o -name "*.dylib" \) | while read -r BIN; do
  TEAM=$(/usr/bin/codesign -dv "$BIN" 2>&1 | grep -Eo 'TeamIdentifier=[A-Z0-9]+' || echo "TeamIdentifier=adhoc")
  printf "│ %-37s│ %-11s│\n" "$(basename "$BIN")" "${TEAM#TeamIdentifier=}"
done
printf "└%s┴%s┘\n" "$(printf '─%.0s' {1..38})" "$(printf '─%.0s' {1..12})"

[[ -z "$IDENTITY" ]] \
  && echo "🔓  Framework is UNSIGNED (Ad-hoc); Xcode will sign it when you embed it." \
  || echo "🔏  All binaries signed by: $IDENTITY"