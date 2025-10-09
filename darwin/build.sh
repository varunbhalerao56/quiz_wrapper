#!/bin/bash

# Exit on error
set -e

# Get the directory where the script is located
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to build and fix rpaths for a platform
build_platform() {
    local platform=$1
    local dev_team=$2
    
    echo "Building for ${platform}..."
    bash "${script_dir}/run_build.sh" llama.cpp $dev_team ${platform}
    
    cd "${script_dir}"
    echo "Fixing rpaths for ${platform}..."
    bash "fix_rpath.sh" ${platform}
    cd "${script_dir}/.."
}

# Replace YOUR_DEVELOPER_TEAM_ID with your actual Apple Developer Team ID
# Uncomment the platforms you need to build
build_platform "MAC_ARM64" "VF63K2KVXG"
# build_platform "OS64" "YOUR_DEVELOPER_TEAM_ID"
# build_platform "SIMULATOR64" "YOUR_DEVELOPER_TEAM_ID"
# build_platform "SIMULATORARM64" "YOUR_DEVELOPER_TEAM_ID"
# build_platform "MAC_CATALYST_ARM64" "YOUR_DEVELOPER_TEAM_ID"

echo "Build completed successfully for all platforms."