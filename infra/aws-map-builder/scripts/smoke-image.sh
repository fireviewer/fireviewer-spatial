#!/usr/bin/env bash
set -euo pipefail

image="${1:-}"
if [[ -z "${image}" ]]; then
  echo "usage: smoke-image.sh <image>" >&2
  exit 2
fi

os="$(docker image inspect "${image}" --format '{{.Os}}')"
architecture="$(docker image inspect "${image}" --format '{{.Architecture}}')"
if [[ "${os}/${architecture}" != "linux/amd64" ]]; then
  echo "expected linux/amd64, observed ${os}/${architecture}" >&2
  exit 3
fi

docker run --rm --platform linux/amd64 --entrypoint /bin/sh "${image}" -ec '
  test -x /usr/local/bin/map-builder
  test -x /usr/local/bin/aws-map-builder-entrypoint
  test -x /opt/fireviewer/aws/execute-batch.sh
  test -d /opt/fireviewer/assets/simready_final_0001_0294
  command -v aws >/dev/null
  map-builder --help >/dev/null
'

docker run --rm --platform linux/amd64 "${image}" --help >/dev/null
docker run --rm --platform linux/amd64 "${image}" /bin/sh -ec '
  test "$1" = "batch-command-dispatch-pass"
' _ batch-command-dispatch-pass

echo "Map Builder image smoke test: PASS (${image}, ${os}/${architecture})"
