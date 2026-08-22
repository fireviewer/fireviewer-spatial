#!/usr/bin/env bash
set -euo pipefail

request_s3_uri=""
shards_s3_uri=""
expected_image_digest=""
shard_count=""

while (($#)); do
  case "$1" in
    --request-s3-uri) request_s3_uri="${2:-}"; shift 2 ;;
    --shards-s3-uri) shards_s3_uri="${2:-}"; shift 2 ;;
    --image-digest) expected_image_digest="${2:-}"; shift 2 ;;
    --shard-count) shard_count="${2:-}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 20 ;;
  esac
done

if [[ ! "${request_s3_uri}" =~ ^s3://[^/]+/.+ ]] || [[ ! "${shards_s3_uri}" =~ ^s3://[^/]+/.+ ]]; then
  echo "request and shards must be complete S3 URIs" >&2
  exit 20
fi
if [[ ! "${expected_image_digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "image digest must be an immutable sha256 digest" >&2
  exit 20
fi

job_id="${AWS_BATCH_JOB_ID:-}"
shard_index="${AWS_BATCH_JOB_ARRAY_INDEX:-}"
if [[ ! "${job_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] \
  || [[ ! "${shard_index}" =~ ^[0-9]+$ ]] \
  || [[ ! "${shard_count}" =~ ^[1-9][0-9]*$ ]] \
  || ((shard_index >= shard_count)); then
  echo "AWS Batch array identity is missing or invalid" >&2
  exit 20
fi

run_root="/scratch/${job_id}"
request_path="${run_root}/inputs/request.json"
output_path="${run_root}/export"
mkdir -p "${run_root}/inputs" "${output_path}" "${run_root}/tmp" "${run_root}/cache"
export TMPDIR="${run_root}/tmp" TMP="${run_root}/tmp" TEMP="${run_root}/tmp"
export XDG_CACHE_HOME="${run_root}/cache"

if ! aws s3 cp "${request_s3_uri}" "${request_path}" --only-show-errors; then
  echo "Request download failed" >&2
  exit 30
fi
python - "${request_path}" "${expected_image_digest}" <<'PY'
import json
import sys

request = json.load(open(sys.argv[1], encoding="utf-8"))
if request.get("builder", {}).get("image_digest") != sys.argv[2]:
    raise SystemExit("request image digest differs from the job definition")
PY

shard_uri="${shards_s3_uri%/}/${shard_index}"
shard_without_scheme="${shard_uri#s3://}"
shard_bucket="${shard_without_scheme%%/*}"
shard_prefix="${shard_without_scheme#*/}"
done_key="${shard_prefix}/shard.done.json"
if aws s3api head-object --bucket "${shard_bucket}" --key "${done_key}" >/dev/null 2>&1; then
  echo "Refusing to overwrite a sealed tile shard: ${shard_uri}" >&2
  exit 30
fi

/usr/local/bin/map-builder \
  --request "${request_path}" \
  --scratch-root "${run_root}/builder" \
  --output "${output_path}" \
  --mode tile-shard \
  --shard-index "${shard_index}" \
  --shard-count "${shard_count}"

done_path="${output_path}/shard.done.json"
if [[ ! -f "${done_path}" ]] || find "${output_path}" -name zone.done.json -print -quit | grep -q .; then
  echo "Shard output is incomplete or contains a forbidden final map marker" >&2
  exit 31
fi
local_object_count="$(find "${output_path}" -type f ! -name shard.done.json | wc -l | tr -d ' ')"
if ! aws s3 cp "${output_path}" "${shard_uri}" --recursive --exclude shard.done.json \
  --checksum-algorithm SHA256 --only-show-errors; then
  exit 75
fi
remote_object_count="$(aws s3api list-objects-v2 --bucket "${shard_bucket}" \
  --prefix "${shard_prefix}/" --query 'length(Contents)' --output text)"
if [[ "${remote_object_count}" != "${local_object_count}" ]]; then
  echo "Shard artifact count mismatch: local=${local_object_count}, s3=${remote_object_count}" >&2
  exit 31
fi
read -r done_sha256 done_checksum_b64 < <(python - "${done_path}" <<'PY'
import base64
import hashlib
import sys

digest = hashlib.sha256(open(sys.argv[1], "rb").read()).digest()
print(digest.hex(), base64.b64encode(digest).decode("ascii"))
PY
)
aws s3api put-object --bucket "${shard_bucket}" --key "${done_key}" --body "${done_path}" \
  --checksum-algorithm SHA256 --checksum-sha256 "${done_checksum_b64}" \
  --metadata "sha256=${done_sha256},publication-order=last" >/dev/null
printf 'Tile shard complete: index=%s/%s, objects=%s, output=%s\n' \
  "${shard_index}" "${shard_count}" "${local_object_count}" "${shard_uri}"
