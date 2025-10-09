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
echo "Setup complete!"
echo "To build: ./scripts/build_macos.sh third_party/llama.cpp ${TEAM_ID}"