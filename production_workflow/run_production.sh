#!/usr/bin/env bash
set -euo pipefail

source /home/firedrake/firedrake/bin/activate

workflow_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JOG_DOCKER_IMAGE="${JOG_DOCKER_IMAGE:-icepack-image-correct}"
export JOG_DOCKER_IMAGE_ID="${JOG_DOCKER_IMAGE_ID:-sha256:75366f110d592001f6c04ec36e9d93ecf9a6596082c3d467d36b0c2354919359}"

exec python "${workflow_dir}/production_amundsen.py" "$@"
