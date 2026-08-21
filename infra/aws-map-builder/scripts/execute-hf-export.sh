#!/usr/bin/env bash
set -euo pipefail

source_s3_uri=""
receipt_s3_uri=""
repo_id=""
map_job_id=""
expected_image_digest=""

while (($#)); do
  case "$1" in
    --source-s3-uri) source_s3_uri="${2:-}"; shift 2 ;;
    --receipt-s3-uri) receipt_s3_uri="${2:-}"; shift 2 ;;
    --repo-id) repo_id="${2:-}"; shift 2 ;;
    --job-id) map_job_id="${2:-}"; shift 2 ;;
    --image-digest) expected_image_digest="${2:-}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 20 ;;
  esac
done

if [[ ! "${source_s3_uri}" =~ ^s3://[^/]+/maps/.+ ]] \
  || [[ ! "${receipt_s3_uri}" =~ ^s3://[^/]+/maps/.+/provenance/hf-viewer-publication.json$ ]]; then
  echo "source and receipt must be confined Map Builder S3 URIs" >&2
  exit 20
fi
if [[ ! "${repo_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]]; then
  echo "invalid Hugging Face dataset id" >&2
  exit 20
fi
if [[ ! "${map_job_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$ ]]; then
  echo "invalid map job id" >&2
  exit 20
fi
if [[ ! "${expected_image_digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "image digest must be immutable" >&2
  exit 20
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is missing" >&2
  exit 20
fi

receipt_without_scheme="${receipt_s3_uri#s3://}"
receipt_bucket="${receipt_without_scheme%%/*}"
receipt_key="${receipt_without_scheme#*/}"
if aws s3api head-object --bucket "${receipt_bucket}" --key "${receipt_key}" >/dev/null 2>&1; then
  echo "Hugging Face publication receipt already exists"
  exit 0
fi

batch_job_id="${AWS_BATCH_JOB_ID:-}"
if [[ ! "${batch_job_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$ ]]; then
  echo "AWS_BATCH_JOB_ID is missing or invalid" >&2
  exit 20
fi
run_root="/scratch/${batch_job_id}"
output_receipt="${run_root}/hf-viewer-publication.json"
mkdir -p "${run_root}/viewer-tiled" "${run_root}/tmp" "${run_root}/cache" "${run_root}/hf-xet"
export TMPDIR="${run_root}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export XDG_CACHE_HOME="${run_root}/cache"
export HF_XET_CACHE="${run_root}/hf-xet"
export HF_XET_HIGH_PERFORMANCE=1

if ! aws s3 cp "${source_s3_uri%/}/zone.done.json" "${run_root}/zone.done.json" --only-show-errors; then
  echo "sealed zone receipt download failed" >&2
  exit 30
fi
if ! aws s3 cp "${source_s3_uri%/}/runtime/viewer-tiled" "${run_root}/viewer-tiled" \
  --recursive --only-show-errors; then
  echo "tiled viewer download failed" >&2
  exit 75
fi

viewer_object_count="$(find "${run_root}/viewer-tiled" -type f | wc -l | tr -d ' ')"
if ((viewer_object_count < 3)); then
  echo "tiled viewer is incomplete" >&2
  exit 31
fi

source_without_scheme="${source_s3_uri#s3://}"
source_prefix="${source_without_scheme#*/}"
remote_root="${source_prefix%/}/runtime"
python /opt/fireviewer/runtime/hf_viewer_exporter.py \
  --input "${run_root}" \
  --repo-id "${repo_id}" \
  --remote-root "${remote_root}" \
  --job-id "${map_job_id}" \
  --image-digest "${expected_image_digest}" \
  --output-receipt "${output_receipt}"

read -r receipt_sha256 receipt_checksum_b64 < <(
  python - "${output_receipt}" <<'PY'
import base64
import hashlib
import sys

digest = hashlib.sha256(open(sys.argv[1], "rb").read()).digest()
print(digest.hex(), base64.b64encode(digest).decode("ascii"))
PY
)

if ! aws s3api put-object \
  --bucket "${receipt_bucket}" \
  --key "${receipt_key}" \
  --body "${output_receipt}" \
  --if-none-match '*' \
  --checksum-algorithm SHA256 \
  --checksum-sha256 "${receipt_checksum_b64}" \
  --metadata "sha256=${receipt_sha256},publication-stage=hf-final" \
  >/dev/null; then
  echo "immutable Hugging Face publication receipt write failed" >&2
  exit 31
fi

remote_checksum="$(aws s3api head-object \
  --bucket "${receipt_bucket}" \
  --key "${receipt_key}" \
  --checksum-mode ENABLED \
  --query ChecksumSHA256 \
  --output text)"
if [[ "${remote_checksum}" != "${receipt_checksum_b64}" ]]; then
  echo "Hugging Face publication receipt checksum mismatch" >&2
  exit 31
fi

printf 'Hugging Face viewer publication complete: files=%s receipt=%s\n' \
  "${viewer_object_count}" "${receipt_s3_uri}"
