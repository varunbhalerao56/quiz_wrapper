#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THIRD_PARTY_DIR="${PROJECT_ROOT}/third_party"
LLAMA_CPP_DIR="${THIRD_PARTY_DIR}/llama.cpp"

echo "Setting up llama.cpp..."

# Create third_party directory if it doesn't exist
mkdir -p "${THIRD_PARTY_DIR}"

# Clone llama.cpp if it doesn't exist
if [ ! -d "${LLAMA_CPP_DIR}" ]; then
  echo "Cloning llama.cpp..."
  git clone https://github.com/ggml-org/llama.cpp.git "${LLAMA_CPP_DIR}"
else
  echo "llama.cpp already exists"
fi

cd "${LLAMA_CPP_DIR}"

# Get current commit hash
CURRENT_COMMIT=$(git rev-parse HEAD)
echo "Current llama.cpp commit: ${CURRENT_COMMIT}"

# Save commit hash to version file
echo "${CURRENT_COMMIT}" > "${PROJECT_ROOT}/LLAMA_VERSION"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Setup complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo ""
echo "1. Build for macOS:"
echo "   ./scripts/build_macos.sh third_party/llama.cpp \${VF63K2KVXG}"
echo ""
echo "2. Build for iOS (all platforms):"
echo "   ./scripts/build_ios.sh third_party/llama.cpp VF63K2KVXG"
echo ""
echo "3. Create XCFramework (after iOS build):"
echo "   ./darwin/build_xcframework.sh"
echo ""
echo "Note: Replace \${TEAM_ID} with your Apple Developer Team ID"
echo "      Find it at: https://developer.apple.com/account"