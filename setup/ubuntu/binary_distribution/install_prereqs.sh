#!/bin/bash
#
# Install development and runtime prerequisites for binary distributions of
# Drake on Ubuntu 18.04 (Bionic) or 20.04 (Focal).

set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo 'ERROR: This script must be run as root' >&2
  exit 1
fi

if command -v conda &>/dev/null; then
  echo 'WARNING: Anaconda is NOT supported. Please remove the Anaconda bin directory from the PATH.' >&2
fi

apt-get update
apt-get install --no-install-recommends lsb-release

codename=$(lsb_release -sc)

if [[ "${codename}" != 'bionic' && "${codename}" != 'focal' ]]; then
  echo 'ERROR: This script requires Ubuntu 18.04 (Bionic) or 20.04 (Focal)' >&2
  exit 2
fi
if [[ "${codename}" == 'focal' ]]; then
  echo 'WARNING: Drake is not officially supported on Ubuntu 20.04 (Focal)' >&2
fi

apt-get install --no-install-recommends $(tr '\n' ' ' <<EOF
build-essential
cmake
pkg-config
EOF
)

apt-get install --no-install-recommends $(cat "${BASH_SOURCE%/*}/packages-${codename}.txt" | tr '\n' ' ')
