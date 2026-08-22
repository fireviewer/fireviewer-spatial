#!/usr/bin/env bash
set -euo pipefail

request_s3_uri=""
output_s3_uri=""
expected_image_digest=""
shards_s3_uri=""
shard_count=""

while (($#)); do
  case "$1" in
    --request-s3-uri)
      request_s3_uri="${2:-}"
      shift 2
      ;;
    --output-s3-uri)
      output_s3_uri="${2:-}"
      shift 2
      ;;
    --image-digest)
      expected_image_digest="${2:-}"
      shift 2
      ;;
    --shards-s3-uri)
      shards_s3_uri="${2:-}"
      shift 2
      ;;
    --shard-count)
      shard_count="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 20
      ;;
  esac
done

if [[ ! "${request_s3_uri}" =~ ^s3://[^/]+/.+ ]] || [[ ! "${output_s3_uri}" =~ ^s3://[^/]+/.+ ]]; then
  echo "request and output must be complete S3 URIs" >&2
  exit 20
fi
if [[ ! "${expected_image_digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "image digest must be an immutable sha256 digest" >&2
  exit 20
fi
if [[ "${shards_s3_uri}" == "disabled" && "${shard_count}" == "0" ]]; then
  shards_s3_uri=""
  shard_count=""
elif [[ -n "${shards_s3_uri}" || -n "${shard_count}" ]]; then
  if [[ ! "${shards_s3_uri}" =~ ^s3://[^/]+/.+ ]] || [[ ! "${shard_count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "shards URI and positive shard count must be supplied together" >&2
    exit 20
  fi
fi

job_id="${AWS_BATCH_JOB_ID:-}"
if [[ ! "${job_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "AWS_BATCH_JOB_ID is missing or invalid" >&2
  exit 20
fi

run_root="/scratch/${job_id}"
request_path="${run_root}/inputs/request.json"
output_path="${run_root}/export"
mkdir -p "${run_root}/inputs" "${output_path}" "${run_root}/tmp" "${run_root}/cache"

export TMPDIR="${run_root}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export XDG_CACHE_HOME="${run_root}/cache"

if ! aws s3 cp "${request_s3_uri}" "${request_path}" --only-show-errors; then
  echo "Request download failed; treating a missing or inaccessible request as deterministic" >&2
  exit 30
fi

python - "${request_path}" "${expected_image_digest}" <<'PY'
import json
import sys

request = json.load(open(sys.argv[1], encoding="utf-8"))
observed = request.get("builder", {}).get("image_digest")
if observed != sys.argv[2]:
    raise SystemExit(
        f"request image digest {observed!r} does not match job definition {sys.argv[2]!r}"
    )
PY

checkpoint_args=()
if [[ -n "${shards_s3_uri}" ]]; then
  shard_download_root="${run_root}/shards"
  checkpoint_seed="${run_root}/checkpoint-seed"
  mkdir -p "${shard_download_root}"
  if ! aws s3 cp "${shards_s3_uri}" "${shard_download_root}" --recursive --only-show-errors; then
    echo "Tile shard download failed" >&2
    exit 75
  fi
  read -r request_build_id request_zone_id < <(python - "${request_path}" <<'PY'
import json
import sys

request = json.load(open(sys.argv[1], encoding="utf-8"))
print(request["build_id"], request["zone_id"])
PY
)
  python /opt/fireviewer/runtime/merge_tile_shards.py \
    --source "${shard_download_root}" \
    --output "${checkpoint_seed}" \
    --build-id "${request_build_id}" \
    --zone-id "${request_zone_id}" \
    --shard-count "${shard_count}"
  checkpoint_args=(--checkpoint-seed "${checkpoint_seed}")
fi

output_without_scheme="${output_s3_uri#s3://}"
output_bucket="${output_without_scheme%%/*}"
output_prefix="${output_without_scheme#*/}"
output_prefix="${output_prefix%/}"
done_key="${output_prefix}/zone.done.json"

if aws s3api head-object --bucket "${output_bucket}" --key "${done_key}" >/dev/null 2>&1; then
  echo "Refusing to overwrite an already sealed build: ${output_s3_uri}" >&2
  exit 30
fi

set +e
/usr/local/bin/map-builder \
  --request "${request_path}" \
  --scratch-root "${run_root}" \
  --output "${output_path}" \
  "${checkpoint_args[@]}"
builder_status=$?
set -e
if ((builder_status != 0)); then
  exit "${builder_status}"
fi

done_path="${output_path}/zone.done.json"
if [[ ! -f "${done_path}" ]]; then
  echo "Builder completed without zone.done.json" >&2
  exit 31
fi

local_object_count="$(find "${output_path}" -type f ! -name zone.done.json | wc -l | tr -d ' ')"
if ((local_object_count < 1)); then
  echo "Builder output contains no publishable artifacts" >&2
  exit 31
fi

if ! aws s3 cp "${output_path}" "${output_s3_uri}" \
  --recursive \
  --exclude zone.done.json \
  --checksum-algorithm SHA256 \
  --only-show-errors; then
  echo "Bulk artifact upload failed" >&2
  exit 75
fi

if ! remote_object_count="$(aws s3api list-objects-v2 \
  --bucket "${output_bucket}" \
  --prefix "${output_prefix}/" \
  --query 'length(Contents)' \
  --output text)"; then
  echo "Unable to count uploaded artifacts" >&2
  exit 75
fi
if [[ "${remote_object_count}" != "${local_object_count}" ]]; then
  echo "Artifact count mismatch: local=${local_object_count}, s3=${remote_object_count}" >&2
  exit 31
fi

read -r done_sha256 done_checksum_b64 < <(
  python - "${done_path}" <<'PY'
import base64
import hashlib
import sys

digest = hashlib.sha256(open(sys.argv[1], "rb").read()).digest()
print(digest.hex(), base64.b64encode(digest).decode("ascii"))
PY
)

if ! aws s3api put-object \
  --bucket "${output_bucket}" \
  --key "${done_key}" \
  --body "${done_path}" \
  --checksum-algorithm SHA256 \
  --checksum-sha256 "${done_checksum_b64}" \
  --metadata "sha256=${done_sha256},publication-order=last" \
  >/dev/null; then
  echo "Final zone.done.json publication failed" >&2
  exit 75
fi

if ! remote_done_checksum="$(aws s3api head-object \
  --bucket "${output_bucket}" \
  --key "${done_key}" \
  --checksum-mode ENABLED \
  --query ChecksumSHA256 \
  --output text)"; then
  echo "Unable to verify final zone.done.json checksum" >&2
  exit 75
fi
if [[ "${remote_done_checksum}" != "${done_checksum_b64}" ]]; then
  echo "Final zone.done.json checksum mismatch" >&2
  exit 31
fi

printf 'Batch publication complete: objects=%s, done_sha256=%s, output=%s\n' \
  "${local_object_count}" "${done_sha256}" "${output_s3_uri}"
