#!/bin/bash
# ------------------------------------------------------------------------------
# This script performs the following steps:
#   1. Creates a temporary directory.
#   2. Recursively clones the 'develop' branch of the Partitura repository into that directory.
#   3. Installs the cloned package using pip.
#   4. Cleans up by deleting the temporary directory.
#
# Requirements:
#   - git
#   - pip
#   - A Unix-like shell environment
# ------------------------------------------------------------------------------

# Exit immediately if a command exits with a non-zero status.
set -e
set -o pipefail

# Create a temporary directory for cloning the repository.
TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: ${TEMP_DIR}"

# Set up a trap to automatically remove the temporary directory on script exit.
# shellcheck disable=SC2064
trap "rm -rf '${TEMP_DIR}'" EXIT

# Recursively clone the 'develop' branch of the Partitura repository into the temporary directory.
# The repository will be cloned into ${TEMP_DIR}/partitura.
echo "Cloning the 'develop' branch of the Partitura repository recursively..."
git clone -b develop --recursive https://github.com/CPJKU/partitura.git "${TEMP_DIR}/partitura"

# Install the package from the cloned repository using pip.
echo "Installing the Partitura package via pip..."
pip install "${TEMP_DIR}/partitura"

# If using Python 3 and pip3 is required, uncomment the following line:
# pip3 install "${TEMP_DIR}/partitura"

echo "Installation complete. The temporary clone has been removed."